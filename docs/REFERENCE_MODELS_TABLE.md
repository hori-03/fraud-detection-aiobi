## 📊 Table `reference_models` - Modèles de Référence (Backoffice)

### 🎯 Objectif

Cette table contient les **~40 modèles pré-entraînés** (Dataset1-40) stockés dans `data/automl_models/`. 

**Architecture à 2 niveaux**:
```
┌─────────────────────────────────────────────────────────────┐
│  NIVEAU 1: MODÈLES DE RÉFÉRENCE (Backoffice)               │
│  📊 Table: reference_models                                 │
│  📁 Dossier: data/automl_models/Dataset1-40/               │
│  👁️  Visibilité: INVISIBLE pour les utilisateurs           │
│  🎯 Utilisation: Auto-match, Ensemble, Transfert learning  │
└─────────────────────────────────────────────────────────────┘
                           ⬇
┌─────────────────────────────────────────────────────────────┐
│  NIVEAU 2: MODÈLES UTILISATEURS (Frontend)                 │
│  📊 Table: training_history                                 │
│  📁 Dossier: models/xgboost_models/user_models/            │
│  👁️  Visibilité: VISIBLE pour les utilisateurs             │
│  🎯 Utilisation: Prédictions, Téléchargement, Analytics    │
└─────────────────────────────────────────────────────────────┘
```

### 📋 Schéma de la Table

```sql
CREATE TABLE reference_models (
    -- Identification
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) UNIQUE NOT NULL,  -- Ex: "Dataset27"
    model_path VARCHAR(500) NOT NULL,          -- data/automl_models/Dataset27/
    
    -- Métadonnées dataset
    dataset_size INTEGER,                      -- Nombre de lignes
    num_features INTEGER,                      -- Features originales
    num_engineered_features INTEGER,           -- Features après engineering
    fraud_rate FLOAT,                          -- Taux de fraude (%)
    
    -- Colonnes (pour matching sémantique)
    column_names TEXT,                         -- JSON: ["tx_id", "amount", ...]
    column_types TEXT,                         -- JSON: {"amount": "float", ...}
    
    -- Métriques
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    roc_auc FLOAT,
    
    -- Configuration
    hyperparameters TEXT,                      -- JSON hyperparams
    feature_importance TEXT,                   -- JSON importance scores
    engineering_methods TEXT,                  -- JSON engineering flags
    
    -- Métadonnées matching
    domain VARCHAR(100),                       -- "banking", "e-commerce", etc.
    data_quality VARCHAR(50),                  -- "high", "medium", "low"
    imbalance_ratio FLOAT,                     -- Ratio fraud/normal
    
    -- Statut
    is_active BOOLEAN DEFAULT TRUE,
    version VARCHAR(50),                       -- "1.0", "2.1"
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_used_at TIMESTAMP,
    
    -- Statistiques
    usage_count INTEGER DEFAULT 0,
    avg_similarity_score FLOAT,
    
    -- Notes admin
    description TEXT,
    tags VARCHAR(500)                          -- "fraud,banking,africa"
);

CREATE INDEX idx_model_name ON reference_models(model_name);
CREATE INDEX idx_is_active ON reference_models(is_active);
CREATE INDEX idx_domain ON reference_models(domain);
```

### 🔄 Workflow Complet

#### 1. Dataset Non Étiqueté (Utilisateur ne voit PAS les modèles de référence)

```
Utilisateur upload transactions_janvier.csv
              ⬇
┌──────────────────────────────────────────────────┐
│  BACKOFFICE: Auto-match dans reference_models   │
│                                                   │
│  1. Analyse colonnes du CSV:                    │
│     ["tx_id", "amount", "merchant", "country"]  │
│                                                   │
│  2. Query SQL:                                   │
│     SELECT * FROM reference_models              │
│     WHERE is_active = TRUE                      │
│                                                   │
│  3. Calcul similarité sémantique:               │
│     - Dataset27: 92% match ✅                   │
│     - Dataset31: 89% match                      │
│     - Dataset35: 87% match                      │
│                                                   │
│  4. Ensemble de top-3 modèles                   │
│     + Anomaly Detection                         │
│     + Calibration                               │
└──────────────────────────────────────────────────┘
              ⬇
┌──────────────────────────────────────────────────┐
│  FRONTEND: CSV simplifié pour utilisateur       │
│                                                   │
│  📄 predictions_janvier.csv                      │
│  Customer_ID, Transaction_ID, Timestamp,        │
│  Fraud_Probability, Risk_Level                  │
│                                                   │
│  ✅ 147 HIGH RISK                               │
│  ⚠️  2,345 MEDIUM RISK                          │
│  ✅ 97,508 LOW RISK                             │
└──────────────────────────────────────────────────┘
```

**L'utilisateur voit**: Un CSV avec ses prédictions  
**L'utilisateur ne voit PAS**: Quels modèles (Dataset27, 31, 35) ont été utilisés

#### 2. Dataset Étiqueté (Utilisateur obtient son propre modèle)

```
Utilisateur upload training_data.csv + colonne fraude
              ⬇
┌──────────────────────────────────────────────────┐
│  TRAINING: full_automl.py                       │
│                                                   │
│  1. Feature Engineering                         │
│  2. Feature Selection                           │
│  3. Meta-Transformer Hyperparams                │
│  4. XGBoost Training                            │
└──────────────────────────────────────────────────┘
              ⬇
┌──────────────────────────────────────────────────┐
│  SAUVEGARDE:                                     │
│  📊 Table: training_history                     │
│  📁 Dossier: models/xgboost_models/user_123/    │
│       ├── fraud_model_20240104.joblib           │
│       ├── feature_engineer.joblib               │
│       └── metadata.json                         │
└──────────────────────────────────────────────────┘
              ⬇
┌──────────────────────────────────────────────────┐
│  FRONTEND: Interface utilisateur                │
│                                                   │
│  🎉 Modèle entraîné avec succès!                │
│  📊 Accuracy: 96.2%                             │
│  📊 F1 Score: 94.8%                             │
│                                                   │
│  [Voir le modèle]  [Faire des prédictions]     │
└──────────────────────────────────────────────────┘
```

**L'utilisateur voit**: Son propre modèle XGBoost, ses métriques, interface de prédiction  
**L'utilisateur ne voit PAS**: Les 40 modèles de référence utilisés en interne

### 🔧 Configuration & Setup

#### 1. Créer la Table

```bash
cd APP_autoML

# Générer migration
flask db migrate -m "Add reference_models table"

# Appliquer migration
flask db upgrade
```

#### 2. Peupler la Table (40 modèles)

```bash
# Peupler automatiquement depuis data/automl_models/
python populate_reference_models.py

# Résultat:
# ✨ Added: 40 models
# 📊 Total in DB: 40 models
```

#### 3. Vérifier le Contenu

```bash
python populate_reference_models.py --show

# Résultat:
# ✅ ACTIVE | Dataset1       | Domain: banking      | ROC-AUC: 0.9984 | Used: 0   times
# ✅ ACTIVE | Dataset2       | Domain: banking      | ROC-AUC: 0.9976 | Used: 0   times
# ...
# ✅ ACTIVE | Dataset40      | Domain: insurance    | ROC-AUC: 0.9892 | Used: 12  times
```

#### 4. Configuration des Domaines (Optionnel)

Modifier `populate_reference_models.py` pour ajuster les domaines:

```python
# Détecter le domaine basé sur le nom du dataset
dataset_num = int(model_dir.name.replace('Dataset', ''))
if dataset_num <= 10:
    metadata['domain'] = 'banking'
elif dataset_num <= 20:
    metadata['domain'] = 'e-commerce'
elif dataset_num <= 30:
    metadata['domain'] = 'telecom'
else:
    metadata['domain'] = 'insurance'
```

### 📊 API Python

#### Trouver le Meilleur Modèle

```python
from app.models.reference_model import ReferenceModel

# Auto-match basé sur colonnes
column_names = ["tx_id", "amount", "merchant", "country", "time"]
best_model, similarity = ReferenceModel.find_best_match(
    column_names=column_names,
    dataset_size=100000,
    fraud_rate=0.015
)

print(f"Best match: {best_model.model_name}")
print(f"Similarity: {similarity:.2%}")
print(f"ROC-AUC: {best_model.roc_auc:.4f}")
```

#### Récupérer les Modèles Actifs

```python
# Tous les modèles actifs
models = ReferenceModel.get_active_models()

# Filtrer par domaine
banking_models = ReferenceModel.get_active_models(domain='banking')

# Filtrer par performance
high_perf = ReferenceModel.get_active_models(min_roc_auc=0.99)
```

#### Incrémenter l'Utilisation

```python
# Après utilisation d'un modèle
model = ReferenceModel.query.filter_by(model_name='Dataset27').first()
model.increment_usage(similarity_score=0.92)

# Met à jour automatiquement:
# - usage_count += 1
# - last_used_at = now()
# - avg_similarity_score (moyenne glissante)
```

### 🔒 Sécurité & Isolation

#### Isolation Complète

```
┌─────────────────────────────────────────────────┐
│  UTILISATEUR A                                  │
│  👁️  Voit: Ses 5 modèles dans training_history │
│  ❌ Ne voit PAS: reference_models               │
│  ❌ Ne voit PAS: Modèles d'autres utilisateurs │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  UTILISATEUR B                                  │
│  👁️  Voit: Ses 3 modèles dans training_history │
│  ❌ Ne voit PAS: reference_models               │
│  ❌ Ne voit PAS: Modèles d'autres utilisateurs │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  SYSTÈME (Backoffice)                           │
│  🔧 Utilise: reference_models (40 modèles)     │
│  🔧 Utilise: training_history (tous)           │
│  🔒 Contrôle: Accès complet                    │
└─────────────────────────────────────────────────┘
```

#### Routes Protégées

- ✅ `/api/apply_unlabeled` : Utilise `reference_models` (transparent pour utilisateur)
- ✅ `/api/train` : Crée modèle dans `training_history` (visible pour utilisateur)
- ✅ `/models` : Liste uniquement les modèles de l'utilisateur
- ❌ `/admin/reference_models` : Admin uniquement (pas encore implémenté)

### 📈 Statistiques & Monitoring

#### Queries Utiles

```sql
-- Modèles les plus utilisés
SELECT model_name, domain, usage_count, avg_similarity_score
FROM reference_models
WHERE is_active = TRUE
ORDER BY usage_count DESC
LIMIT 10;

-- Performance par domaine
SELECT domain, AVG(roc_auc) as avg_roc_auc, COUNT(*) as count
FROM reference_models
WHERE is_active = TRUE
GROUP BY domain;

-- Modèles récemment utilisés
SELECT model_name, last_used_at, usage_count
FROM reference_models
WHERE last_used_at IS NOT NULL
ORDER BY last_used_at DESC
LIMIT 10;

-- Qualité du matching
SELECT model_name, usage_count, avg_similarity_score
FROM reference_models
WHERE avg_similarity_score > 0.80
ORDER BY avg_similarity_score DESC;
```

### 🎯 Avantages de cette Architecture

#### 1. **Séparation Claire**
```
reference_models (backoffice)
  ├── 40 modèles système
  ├── Invisible utilisateurs
  └── Auto-match & Ensemble

training_history (frontend)
  ├── Modèles utilisateurs
  ├── Visible & gérable
  └── Prédictions personnalisées
```

#### 2. **Performance**
- Index sur `model_name`, `is_active`, `domain`
- Cache des modèles chargés
- Query optimisées (similarité calculée en Python, pas SQL)

#### 3. **Flexibilité**
- Ajout/suppression modèles sans impacter utilisateurs
- Désactivation temporaire (`is_active=False`)
- Versioning des modèles

#### 4. **Monitoring**
- Compteur d'utilisation
- Score de similarité moyen
- Dernière utilisation
- Stats par domaine

### 🚀 Mise en Production

```bash
# 1. Backup de la BDD
pg_dump -U postgres -h railway.proxy.rlwy.net -p 45478 railway > backup.sql

# 2. Créer la table
cd APP_autoML
flask db upgrade

# 3. Peupler
python populate_reference_models.py

# 4. Vérifier
python populate_reference_models.py --show

# 5. Tester l'API
curl -X POST http://localhost:5000/api/apply_unlabeled \
  -H "Content-Type: application/json" \
  -d '{"filepath": "test.csv", "model_name": "test"}'

# 6. Déployer sur Railway
git add .
git commit -m "Add reference_models table"
git push railway main
```

### ✅ Checklist

- [x] Créer modèle `ReferenceModel`
- [x] Créer script `populate_reference_models.py`
- [x] Modifier route `/api/apply_unlabeled`
- [x] Ajouter méthodes `find_best_match()`, `increment_usage()`
- [x] Documentation complète
- [ ] Générer migration Flask-Migrate
- [ ] Appliquer migration sur Railway PostgreSQL
- [ ] Peupler table avec 40 modèles
- [ ] Tester auto-match sur dataset réel
- [ ] Monitoring des statistiques d'utilisation

---

**Date**: 2024-01-04  
**Version**: 1.0  
**Status**: ✅ Implémenté, ⏳ En attente migration BDD
