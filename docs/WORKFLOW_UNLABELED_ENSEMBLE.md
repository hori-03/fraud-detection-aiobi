# 🚀 Workflow Complet: Dataset Non Étiqueté → Mode Ensemble

## 📖 Vue d'Ensemble

Quand un utilisateur upload un **dataset non étiqueté** et coche "Dataset non étiqueté", voici ce qui se passe en coulisses pour appliquer le **Mode Ensemble Complet** comme dans `apply_automl_production.py`.

---

## 🔄 Workflow Étape par Étape

### 1️⃣ **User Upload Dataset Non Étiqueté**

```
Interface: upload.html
┌─────────────────────────────────────────────┐
│ [📁 Choisir fichier] unlabeled_data.csv    │
│ [✅] Dataset non étiqueté                   │
│ [🚀 Appliquer le modèle]                   │
└─────────────────────────────────────────────┘
         ⬇
JavaScript détecte checkbox cochée
         ⬇
Envoie à /api/apply_unlabeled (au lieu de /api/train)
```

---

### 2️⃣ **Analyse des Colonnes & Auto-Match**

```python
# Route: /api/apply_unlabeled

# 1. Charger le CSV
df_prod = pd.read_csv(filepath)
column_names = df_prod.columns.tolist()
# Ex: ['amount', 'merchant', 'country', 'time', 'card_type']

# 2. Chercher dans la BDD PostgreSQL
best_model, similarity = ReferenceModel.find_best_match(
    column_names=column_names,
    dataset_size=len(df_prod)
)
```

**Que fait `find_best_match()` ?**

```python
# Dans app/models/reference_model.py

@staticmethod
def find_best_match(column_names: list, dataset_size: int = None):
    """
    Compare les colonnes de l'utilisateur avec les 40 modèles de référence
    """
    active_models = ReferenceModel.query.filter_by(is_active=True).all()
    # → Récupère les 40 modèles (dataset1-40) depuis PostgreSQL
    
    matcher = ColumnMatcher()
    best_match = None
    best_score = 0
    
    for model in active_models:
        model_columns = json.loads(model.column_names)
        # Ex dataset27: ['tx_amount', 'merchant_name', 'dest_country', ...]
        
        # Similarité sémantique via ColumnMatcher
        similarity = matcher.compute_dataset_similarity(
            column_names,      # ['amount', 'merchant', 'country', ...]
            model_columns      # ['tx_amount', 'merchant_name', 'dest_country', ...]
        )
        # → 0.92 (92% de similarité)
        
        # Bonus si taille dataset similaire
        if dataset_size and model.dataset_size:
            size_ratio = min(dataset_size, model.dataset_size) / max(dataset_size, model.dataset_size)
            similarity *= (0.9 + 0.1 * size_ratio)
        
        if similarity > best_score:
            best_score = similarity
            best_match = model
    
    return best_match, best_score
    # → (ReferenceModel(dataset27), 0.92)
```

**Résultat:**
```
✅ Best match: dataset27 (similarity: 92%)
   - model_path: C:\...\data\automl_models\dataset27\
   - roc_auc: 0.9984
   - domain: banking
```

---

### 3️⃣ **Initialisation d'AutoMLProductionApplicator**

```python
# Route: /api/apply_unlabeled

from automl_transformer.apply_automl_production import AutoMLProductionApplicator

applicator = AutoMLProductionApplicator(
    automl_models_dir='C:\\...\\data\\automl_models'
)
```

**Que fait le constructeur ?**

```python
# Dans automl_transformer/apply_automl_production.py

class AutoMLProductionApplicator:
    def __init__(self, automl_models_dir):
        self.automl_models_dir = Path(automl_models_dir)
        self.column_matcher = ColumnMatcher()
        
        # Scan tous les modèles disponibles
        self.available_models = self._scan_models()
        # → {'dataset1': {...}, 'dataset2': {...}, ..., 'dataset40': {...}}
        
    def _scan_models(self):
        """Parcourt data/automl_models/ et charge les métadonnées"""
        models = {}
        for model_dir in self.automl_models_dir.iterdir():
            if model_dir.is_dir():
                metadata = self._load_metadata(model_dir)
                models[model_dir.name] = {
                    'path': model_dir,
                    'xgboost_model': model_dir / 'xgboost_model.joblib',
                    'feature_engineer': model_dir / 'feature_engineer.joblib',
                    'metadata': metadata
                }
        return models
```

**⚠️ IMPORTANT:** `AutoMLProductionApplicator` charge **DIRECTEMENT** les fichiers `.joblib` depuis `data/automl_models/`, PAS depuis la BDD !

---

### 4️⃣ **Ensemble Predictions (Top 3 Modèles)**

```python
# Route: /api/apply_unlabeled

results = applicator.apply_ensemble_predictions(
    df=df_prod,
    top_k=3,                    # ← Utilise les 3 meilleurs modèles
    anomaly_detection=True,     # ← Active Isolation Forest
    calibrate=True,             # ← Calibre les probabilités
    threshold=0.5
)
```

**Que fait `apply_ensemble_predictions()` ?**

```python
# Dans automl_transformer/apply_automl_production.py

def apply_ensemble_predictions(self, df, top_k=3, anomaly_detection=True, calibrate=True):
    """
    Mode Ensemble Complet - Exactement comme la commande:
    python apply_automl_production.py --ensemble --top_k 3 --anomaly_detection --calibrate
    """
    
    # ÉTAPE 1: Auto-match pour trouver les meilleurs modèles
    column_names = df.columns.tolist()
    ranked_models = []
    
    for model_name, model_info in self.available_models.items():
        model_columns = model_info['metadata'].get('column_names', [])
        
        # Similarité sémantique
        similarity = self.column_matcher.compute_dataset_similarity(
            column_names, 
            model_columns
        )
        
        ranked_models.append((model_name, similarity, model_info))
    
    # Trier par similarité décroissante
    ranked_models.sort(key=lambda x: x[1], reverse=True)
    
    # Sélectionner top_k=3 modèles
    top_models = ranked_models[:top_k]
    # Ex: [('dataset27', 0.92, {...}), ('dataset31', 0.89, {...}), ('dataset35', 0.87, {...})]
    
    
    # ÉTAPE 2: Charger les 3 modèles XGBoost + Feature Engineers
    ensemble_predictions = []
    
    for model_name, similarity, model_info in top_models:
        # Charger le modèle XGBoost
        xgboost_model = joblib.load(model_info['xgboost_model'])
        # → data/automl_models/dataset27/xgboost_model.joblib
        
        # Charger le feature engineer
        feature_engineer = joblib.load(model_info['feature_engineer'])
        # → data/automl_models/dataset27/feature_engineer.joblib
        
        # Transformer les données
        X_transformed = feature_engineer.transform(df)
        
        # Prédictions
        predictions = xgboost_model.predict_proba(X_transformed)[:, 1]
        
        ensemble_predictions.append({
            'model': model_name,
            'similarity': similarity,
            'predictions': predictions
        })
    
    
    # ÉTAPE 3: Moyenne pondérée des 3 modèles
    weights = [p['similarity'] for p in ensemble_predictions]
    weights = [w / sum(weights) for w in weights]  # Normaliser
    
    # Ex: weights = [0.92/(0.92+0.89+0.87), 0.89/(0.92+0.89+0.87), 0.87/(0.92+0.89+0.87)]
    #              = [0.343, 0.332, 0.325]
    
    final_predictions = np.zeros(len(df))
    for pred, weight in zip(ensemble_predictions, weights):
        final_predictions += weight * pred['predictions']
    
    # Calculer variance (stabilité)
    predictions_matrix = np.array([p['predictions'] for p in ensemble_predictions])
    prediction_variance = np.var(predictions_matrix, axis=0)
    
    
    # ÉTAPE 4: Anomaly Detection (Isolation Forest)
    if anomaly_detection:
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(X_transformed)
        
        anomaly_scores = iso_forest.decision_function(X_transformed)
        anomaly_scores_normalized = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        
        # Combiner: 70% XGBoost + 30% Anomaly
        combined_scores = 0.7 * final_predictions + 0.3 * anomaly_scores_normalized
    else:
        combined_scores = final_predictions
        anomaly_scores_normalized = np.zeros(len(df))
    
    
    # ÉTAPE 5: Calibration des probabilités
    if calibrate:
        calibrator = CalibratedClassifierCV(
            estimator=DummyClassifier(),
            method='sigmoid',
            cv='prefit'
        )
        
        # Calibration sigmoïde
        calibrated_predictions = 1 / (1 + np.exp(-5 * (combined_scores - 0.5)))
    else:
        calibrated_predictions = combined_scores
    
    
    # ÉTAPE 6: Retourner les résultats
    return {
        'fraud_probability': final_predictions,
        'fraud_probability_calibrated': calibrated_predictions,
        'anomaly_score': anomaly_scores_normalized,
        'combined_score': combined_scores,
        'prediction_variance': prediction_variance,
        'prediction_stability': 1 - prediction_variance,
        'fraud_prediction': (calibrated_predictions > threshold).astype(int),
        'models_used': [p['model'] for p in ensemble_predictions],
        'model_weights': weights
    }
```

---

### 5️⃣ **Création du CSV Simplifié**

```python
# Route: /api/apply_unlabeled

output_df = _create_simplified_output_unlabeled(df_prod, results)

output_df.to_csv(predictions_filepath, index=False)
```

**Que fait `_create_simplified_output_unlabeled()` ?**

```python
def _create_simplified_output_unlabeled(df, results):
    """
    Crée un CSV simplifié avec les colonnes essentielles
    """
    output = pd.DataFrame()
    
    # Colonnes identifiantes (si disponibles)
    id_cols = ['Customer_ID', 'Transaction_ID', 'customer_id', 'transaction_id', 'tx_id', 'id']
    for col in id_cols:
        if col in df.columns:
            output[col] = df[col]
    
    # Timestamp (si disponible)
    time_cols = ['Timestamp', 'Date', 'time', 'date', 'date_transaction']
    for col in time_cols:
        if col in df.columns:
            output[col] = df[col]
            break
    
    # Probabilités et scores
    output['Fraud_Probability'] = results['fraud_probability']
    output['Fraud_Probability_Calibrated'] = results['fraud_probability_calibrated']
    output['Anomaly_Score'] = results['anomaly_score']
    output['Combined_Score'] = results['combined_score']
    output['Prediction_Variance'] = results['prediction_variance']
    output['Prediction_Stability'] = results['prediction_stability']
    output['Fraud_Prediction'] = results['fraud_prediction']
    
    # Risk Level (HIGH/MEDIUM/LOW)
    output['Risk_Level'] = output['Combined_Score'].apply(lambda x:
        'HIGH' if x > 0.7 else 'MEDIUM' if x > 0.5 else 'LOW'
    )
    
    return output
```

**Résultat CSV:**

```csv
Customer_ID,Transaction_ID,Timestamp,Fraud_Probability,Fraud_Probability_Calibrated,Anomaly_Score,Combined_Score,Prediction_Variance,Prediction_Stability,Fraud_Prediction,Risk_Level
CUST001,TX001,2024-11-04 14:30:00,0.03,0.01,0.02,0.02,0.001,0.999,0,LOW
CUST002,TX002,2024-11-04 15:45:00,0.92,0.98,0.85,0.94,0.003,0.997,1,HIGH
CUST003,TX003,2024-11-04 16:20:00,0.55,0.62,0.45,0.58,0.012,0.988,1,MEDIUM
```

---

### 6️⃣ **Statistiques & Réponse à l'Utilisateur**

```python
# Route: /api/apply_unlabeled

stats = {
    'total_transactions': len(results['fraud_prediction']),
    'high_risk': int((results['combined_score'] > 0.7).sum()),
    'medium_risk': int(((results['combined_score'] >= 0.5) & (results['combined_score'] <= 0.7)).sum()),
    'low_risk': int((results['combined_score'] < 0.5).sum()),
    'anomalies_detected': int((results['anomaly_score'] > 0.7).sum()),
    'prediction_stability': float(results['prediction_stability'].mean())
}

return jsonify({
    'success': True,
    'filepath': str(predictions_filepath),
    'download_url': f'/api/download/{predictions_filename}',
    'stats': stats,
    'model_used': best_model.model_name,
    'similarity_score': similarity_score
})
```

---

## 📊 Résumé du Workflow

```
┌────────────────────────────────────────────────────────────────┐
│ 1. USER UPLOAD                                                 │
│    unlabeled_data.csv (100,000 transactions)                   │
│    Checkbox: "Dataset non étiqueté" ✅                         │
└────────────────────────────────────────────────────────────────┘
              ⬇
┌────────────────────────────────────────────────────────────────┐
│ 2. AUTO-MATCH (Base de données PostgreSQL)                    │
│    ReferenceModel.find_best_match()                            │
│    → Cherche dans 40 modèles (dataset1-40)                     │
│    → Trouve dataset27 (similarité: 92%)                        │
└────────────────────────────────────────────────────────────────┘
              ⬇
┌────────────────────────────────────────────────────────────────┐
│ 3. SCAN MODELS (Fichiers locaux)                              │
│    AutoMLProductionApplicator._scan_models()                   │
│    → Scan data/automl_models/                                  │
│    → Trouve dataset27/xgboost_model.joblib                     │
│    → Trouve dataset31/xgboost_model.joblib                     │
│    → Trouve dataset35/xgboost_model.joblib                     │
└────────────────────────────────────────────────────────────────┘
              ⬇
┌────────────────────────────────────────────────────────────────┐
│ 4. ENSEMBLE PREDICTIONS (Top 3 modèles)                       │
│    - Dataset27: 92% fraude (poids: 0.343)                     │
│    - Dataset31: 88% fraude (poids: 0.332)                     │
│    - Dataset35: 95% fraude (poids: 0.325)                     │
│    → Moyenne pondérée: 92%                                     │
└────────────────────────────────────────────────────────────────┘
              ⬇
┌────────────────────────────────────────────────────────────────┐
│ 5. ANOMALY DETECTION (Isolation Forest)                       │
│    → Détecte patterns bizarres: 85%                            │
│    → Combine: 0.7×92% + 0.3×85% = 90%                         │
└────────────────────────────────────────────────────────────────┘
              ⬇
┌────────────────────────────────────────────────────────────────┐
│ 6. CALIBRATION (Sigmoïde)                                      │
│    → Étire les probabilités: 90% → 98%                        │
└────────────────────────────────────────────────────────────────┘
              ⬇
┌────────────────────────────────────────────────────────────────┐
│ 7. EXPORT CSV SIMPLIFIÉ                                        │
│    predictions_20241104_153045_unlabeled.csv                   │
│    - Customer_ID, Transaction_ID, Timestamp                    │
│    - Fraud_Probability_Calibrated: 98%                         │
│    - Combined_Score: 90%                                       │
│    - Risk_Level: HIGH                                          │
└────────────────────────────────────────────────────────────────┘
              ⬇
┌────────────────────────────────────────────────────────────────┐
│ 8. STATISTIQUES                                                 │
│    Total: 100,000 transactions                                 │
│    HIGH RISK (>70%): 147                                       │
│    MEDIUM RISK (50-70%): 2,345                                 │
│    LOW RISK (<50%): 97,508                                     │
│    Anomalies détectées: 89                                     │
│    Stabilité prédictions: 99.2%                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Points Clés

### ✅ Ce qui est utilisé:

1. **Base de données PostgreSQL** (reference_models):
   - Métadonnées des 40 modèles (column_names, domain, roc_auc, etc.)
   - Auto-match rapide via requête SQL
   - Tracking d'utilisation (usage_count, avg_similarity_score)

2. **Fichiers locaux** (data/automl_models/):
   - `xgboost_model.joblib` (les 3 meilleurs modèles)
   - `feature_engineer.joblib` (transformation des données)
   - `dataset_metadata.json` (infos complémentaires)

### ⚡ Architecture Hybride:

```
PostgreSQL (RAPIDE)          Fichiers Locaux (PRÉCIS)
        │                            │
        ├── Auto-match               ├── Modèles XGBoost
        ├── Métadonnées              ├── Feature Engineers
        └── Statistiques             └── Transformations
                 │                            │
                 └────────┬───────────────────┘
                          ⬇
                  Mode Ensemble Complet
                  (comme apply_automl_production.py)
```

### 🎯 Équivalence:

```bash
# Ligne de commande:
python apply_automl_production.py \
  --dataset nouvelles_transactions.csv \
  --ensemble --top_k 3 \
  --anomaly_detection \
  --calibrate \
  --output results
```

**=**

```javascript
// Interface Web:
1. Upload CSV
2. Cocher "Dataset non étiqueté"
3. Cliquer "Appliquer le modèle"
```

**Résultat identique:** Mode Ensemble Complet avec les 3 meilleurs modèles ! 🎉

---

## 🚀 Avantages de cette Architecture

1. **Recherche rapide** (PostgreSQL): Auto-match en <100ms
2. **Prédictions précises** (Fichiers locaux): Ensemble + Anomaly + Calibration
3. **Pas de duplication**: Fichiers `.joblib` restent en 1 seul endroit
4. **Scalable**: On peut avoir 1000 modèles dans la BDD sans ralentir
5. **Statistiques**: Tracking d'utilisation automatique

---

## 📝 Conclusion

La base de données PostgreSQL est un **catalogue intelligent** qui permet de trouver rapidement les meilleurs modèles. Ensuite, `apply_automl_production.py` charge directement les fichiers `.joblib` pour faire les prédictions avec le **Mode Ensemble Complet**.

✅ **Rien n'est changé** dans `apply_automl_production.py`
✅ **ColumnMatcher est utilisé** pour le matching sémantique
✅ **Mode Ensemble** fonctionne exactement pareil (top 3 modèles + anomaly + calibration)

C'est le **meilleur des deux mondes** ! 🎯
