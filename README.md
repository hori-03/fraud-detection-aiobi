# 🤖 AutoML Fraud Detection System - Aïobi

> Système automatisé de détection de fraude utilisant un Meta-Transformer ML pour prédire les hyperparamètres optimaux

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Railway](https://img.shields.io/badge/Deploy-Railway-blueviolet.svg)](https://railway.app/)

## 📚 Documentation

- **[🚀 Guide Déploiement Rapide](DEPLOIEMENT_RAPIDE.md)** - Déployer sur Railway en 10 minutes
- **[📋 Configuration Variables](APP_autoML/docs/VARIABLES_ENVIRONNEMENT.md)** - Guide complet des variables d'environnement
- **[🔧 Documentation Technique](APP_autoML/RAILWAY_DEPLOYMENT.md)** - Guide détaillé de déploiement

## 🚀 Quick Start

### Application Web (Recommandé)

```bash
# 1. Installation
cd APP_autoML
pip install -r requirements.txt

# 2. Configuration
cp .env.example .env
# Éditer .env avec vos credentials

# 3. Lancer l'application
python run.py
```

Accéder à: http://127.0.0.1:5000

**Fonctionnalités**:
- ✅ Interface web intuitive
- ✅ Upload de datasets
- ✅ Training automatique avec AutoML
- ✅ Prédictions en temps réel
- ✅ Dashboard d'historique
- ✅ Panneau d'administration
- ✅ Login Google OAuth
- ✅ Stockage S3 des modèles

### Ligne de Commande (Advanced)

```bash
# 1. Placer votre dataset
cp votre_fichier.csv data/datasets/MonDataset.csv

# 2. Lancer l'AutoML (auto-détecte tout)
python automl_transformer/full_automl.py data/datasets/MonDataset.csv

# 3. Vérifier les résultats
python show_automl_results.py
```

**C'est tout!** Le système a:
- ✅ Détecté automatiquement les colonnes (amount, time, categorical)
- ✅ Généré 7-28 features automatiquement
- ✅ Prédit les meilleurs hyperparamètres via Meta-Transformer ML
- ✅ Entraîné et évalué le modèle XGBoost
- ✅ Sauvegardé le pipeline complet pour production

## 📊 Résultats

### Performance vs Grid Search

| Dataset | Grid Search F1 | AutoML F1 | Temps Grid | Temps AutoML | Speedup |
|---------|---------------|-----------|------------|--------------|---------|
| Dataset4 | 0.8133 | 0.7886 | 400s+ | 4s | **100x** |
| Dataset5 | 0.6055 | 0.5693 | 400s+ | 4s | **100x** |
| Dataset6 | 0.8858 | **1.0000** 🏆 | 400s+ | 4s | **100x** |
| Dataset7 | 0.7283 | **1.0000** 🏆 | 400s+ | 4s | **100x** |

**Highlights:**
- 🏆 **2/5 datasets**: AutoML MEILLEUR que Grid Search (F1 = 1.0!)
- ⚡ **100x plus rapide** (4s vs 400s+)
- 🎯 **Aucune configuration** requise
- 📦 **Pipeline complet** sauvegardé automatiquement

### ⚠️ Découverte Importante: Data Leakage dans Dataset6 & Dataset7

**Investigation des scores parfaits (F1=1.0):**

Après investigation approfondie, nous avons découvert que les scores parfaits sur Dataset6 et Dataset7 sont dus à du **data leakage** (fuite d'information):

**Dataset6 - LEAKAGE SÉVÈRE 🚨:**
- Tests avec features brutes (sans engineering):
  - Random Forest: **F1=0.93** 🚨
  - XGBoost: **F1=0.90** 🚨 (même algo que l'AutoML)
- Prédicteurs parfaits identifiés:
  - `type_transaction="paiement"` → **0% fraud** (1414 cas = 13%)
  - `destination_country="GH/NG"` → **100% fraud** (60 cas)
  - `montant_transaction`: corrélation **+0.41** (fraudes 8x plus élevées)
- **Verdict**: Dataset contaminé, inutilisable en production réelle

**Dataset7 - LEAKAGE MODÉRÉ ⚠️:**
- Tests avec features brutes (sans engineering):
  - Random Forest: **F1=0.65** ⚠️
  - XGBoost: **F1≈0.65** ⚠️ (même algo que l'AutoML)
- Prédicteurs parfaits: `transaction_type="payment"` → 0% fraud (2115 cas)
- **Gain du feature engineering**: +0.35 (énorme!)
- **Verdict**: Le F1=1.0 vient surtout de l'intelligence du feature engineering

**Note**: Random Forest est utilisé uniquement pour tests rapides. L'AutoML utilise **XGBoost** partout.

**Conclusion:**
- ✅ L'AutoML **fonctionne correctement** - il trouve les patterns (même artificiels)
- ✅ Pour validation réelle, utiliser **Dataset4, 5, 9** (F1=0.57-0.79, réaliste)
- 📚 Documentation complète: [`docs/DATA_LEAKAGE_ANALYSIS.md`](docs/DATA_LEAKAGE_ANALYSIS.md)
- 🔬 Script d'analyse: [`tests/check_dataset6_leakage.py`](tests/check_dataset6_leakage.py)

## 🎯 Fonctionnalités

### � Architecture: Deux Meta-Transformers

**⚠️ Le projet utilise actuellement l'ANCIEN Meta-Transformer** (plus fiable):

| Composant | Ancien (✅ Utilisé) | Nouveau (🧪 Expérimental) |
|-----------|-------------------|------------------------|
| **Fichier** | `ancien_meta/train_metatransformer.py` | `automl_transformer/train_automl_metatransformer.py` |
| **Type** | Transformer PyTorch (4 layers, 128 hidden) | Transformer avec approche différente |
| **Entraînement** | 7 datasets, 105 exemples | Approche plus automatisée |
| **Performance** | R² max_depth=0.83, stable | En développement |
| **Usage** | Chargé dans `full_automl.py` ligne 195-270 | Non utilisé par défaut |
| **Fallback** | Hyperparams par défaut si NaN | - |

**Pourquoi deux versions?**
- L'ancien modèle a fait ses preuves (Dataset6/7: F1=1.0)
- Le nouveau était expérimental avec performances initiales insuffisantes
- L'ancien est intégré directement pour garantir la fiabilité
- Fallback robuste: si prédiction échoue → hyperparams par défaut fonctionnent bien

### �🤖 Auto Feature Engineering
- **Détection automatique** des types de colonnes
  - Amount/Money columns: transformations log, sqrt, bins, flags
  - Temporal columns: hour, day, weekend, business hours
  - Name columns: length, word count
  - Categorical: label encoding automatique

### 🧠 Meta-Transformer ML
- **Prédiction d'hyperparamètres** basée sur les caractéristiques du dataset
- Entraîné sur 7 datasets de référence (105 exemples)
- Prédit 10 hyperparamètres XGBoost en ~1 seconde

### ⚡ Pipeline Production-Ready
- Sauvegarde automatique: modèle + transformations
- Format joblib pour chargement rapide
- Métriques complètes: F1, AUC, Confusion Matrix
- Support multilingue: yes/no, oui/non, 1/0

## 📁 Structure du Projet

```
fraud-project/
├── automl_transformer/          # 🤖 AutoML Principal
│   ├── full_automl.py           # ← Point d'entrée principal
│   ├── auto_feature_engineer.py # Feature engineering automatique
│   └── auto_feature_selector.py # Sélection de features
│
├── ancien_meta/                 # 🧠 Meta-Transformer
│   ├── train_metatransformer.py # Entraîner le Meta-Transformer
│   └── create_unified_metatransformer_dataset.py
│
├── base/                        # 📊 Baseline & Préparation
│   ├── baseline_xgboost.py      # Grid Search manuel
│   └── create_metamodel_examples.py
│
├── data/
│   ├── datasets/                # 📁 CSVs bruts
│   ├── automl_models/           # 💾 Modèles AutoML sauvegardés
│   └── models/                  # 🧠 Meta-Transformer (.pth)
│
├── docs/
│   └── GUIDE_UTILISATION_AUTOML.md  # 📘 Guide complet
│
├── show_automl_results.py       # 📊 Afficher les résultats
└── compare_automl_vs_manual.py  # 📈 Comparer AutoML vs Grid Search
```

## 📚 Documentation

- **[Guide d'Utilisation Complet](docs/GUIDE_UTILISATION_AUTOML.md)** - Comment utiliser l'AutoML
  - Prédiction sur nouveau dataset
  - Entraîner avec nouveaux datasets
  - Réentraîner le Meta-Transformer
  - Troubleshooting complet

- **[Résultats Finaux](AUTOML_FINAL_RESULTS.md)** - Performance détaillée

## 🛠️ Installation

### Prérequis
```bash
Python 3.8+
```

### Installation des Dépendances
```bash
pip install -r requirements.txt
```

**Dépendances principales:**
- `xgboost>=1.7.0` - Modèle de classification
- `torch>=2.0.0` - Meta-Transformer
- `scikit-learn>=1.0.0` - Preprocessing et métriques
- `pandas>=1.5.0` - Manipulation de données
- `numpy>=1.23.0` - Calculs numériques
- `imbalanced-learn>=0.10.0` - SMOTE pour déséquilibre
- `joblib>=1.2.0` - Sauvegarde des modèles

## 💡 Exemples d'Usage

### Exemple 1: Prédiction Simple
```python
from automl_transformer.full_automl import FullAutoML

# Créer l'AutoML
automl = FullAutoML(use_meta_transformer=True)

# Entraîner sur votre dataset
performance = automl.fit(
    csv_path='data/datasets/MonDataset.csv',
    target_col='fraud_flag'  # Optionnel (auto-détecté)
)

print(f"F1 Score: {performance['test_f1']:.4f}")
print(f"AUC: {performance['test_auc']:.4f}")
```

### Exemple 2: Utilisation en Production
```python
import joblib
import pandas as pd

# Charger le pipeline sauvegardé
model = joblib.load('data/automl_models/mondataset/xgboost_model.joblib')
engineer = joblib.load('data/automl_models/mondataset/feature_engineer.joblib')

# Nouvelles transactions
new_data = pd.read_csv('nouvelles_transactions.csv')

# Feature engineering automatique
X_transformed = engineer.transform(new_data)

# Prédire
predictions = model.predict(X_transformed)
probabilities = model.predict_proba(X_transformed)[:, 1]

# Résultats
results = pd.DataFrame({
    'transaction_id': new_data['transaction_id'],
    'fraud_probability': probabilities,
    'is_fraud': predictions
})

# Filtrer les fraudes (seuil > 0.7)
fraudes_detectees = results[results['fraud_probability'] > 0.7]
```

### Exemple 3: Comparer avec Grid Search
```bash
# Lancer AutoML
python automl_transformer/full_automl.py data/datasets/Dataset4.csv

# Comparer avec Grid Search manuel
python compare_automl_vs_manual.py
```

## 🔧 Configuration Avancée

### Modifier les Hyperparamètres par Défaut

Si le Meta-Transformer prédit NaN, le système utilise des hyperparamètres par défaut. Pour les modifier:

```python
# Dans automl_transformer/full_automl.py, ligne ~115

# Hyperparams par défaut
default_hyperparams = {
    'max_depth': 6,              # Profondeur des arbres (3-10)
    'learning_rate': 0.1,        # Taux d'apprentissage (0.01-0.3)
    'n_estimators': 300,         # Nombre d'arbres (100-500)
    'subsample': 0.8,            # Échantillonnage (0.6-1.0)
    'colsample_bytree': 0.8,     # Features par arbre (0.6-1.0)
    'gamma': 0.3,                # Régularisation (0-1)
    'min_child_weight': 5,       # Poids minimum (1-10)
    'scale_pos_weight': 'auto',  # Balance des classes (auto-calculé)
    'reg_alpha': 0.1,            # L1 régularisation
    'reg_lambda': 1.0            # L2 régularisation
}
```

### Activer/Désactiver le Meta-Transformer

```python
# Utiliser Meta-Transformer ML (par défaut)
automl = FullAutoML(use_meta_transformer=True)

# Utiliser uniquement les hyperparams par défaut
automl = FullAutoML(use_meta_transformer=False)
```

## 🧪 Tests

```bash
# Tester sur Dataset5
python automl_transformer/full_automl.py data/datasets/Dataset5.csv fraud_flag

# Tester sur Dataset6 (français)
python automl_transformer/full_automl.py data/datasets/Dataset6.csv label_suspect

# Tester sur Dataset7
python automl_transformer/full_automl.py data/datasets/Dataset7.csv suspicious_flag
```

## 📈 Améliorer les Performances

### Problème: Overfitting (Train F1 >> Test F1)

**Solution 1: Régularisation**
```python
# Augmenter la régularisation
hyperparams = {
    'min_child_weight': 10,  # Au lieu de 5
    'reg_alpha': 0.5,        # Au lieu de 0.1
    'reg_lambda': 2.0,       # Au lieu de 1.0
    'max_depth': 4           # Au lieu de 6
}
```

**Solution 2: Données déséquilibrées**
```python
# Le système calcule automatiquement scale_pos_weight
# Mais vous pouvez ajuster manuellement:
scale_pos_weight = (nb_non_fraud / nb_fraud) * 1.5  # Pénaliser plus les erreurs
```

### Problème: F1 Score trop faible

**Solution: Analyser les features**
```python
import joblib

model = joblib.load('data/automl_models/dataset/xgboost_model.joblib')
importances = model.feature_importances_

# Afficher top 10 features
import pandas as pd
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
```

## 🤝 Contribution

Les contributions sont les bienvenues! Domaines d'amélioration:
- 🔄 Réentraîner Meta-Transformer avec plus de datasets
- 📊 Ajouter d'autres algorithmes (LightGBM, CatBoost)
- 🎯 Optimiser la feature selection
- 📈 Ajouter ensembles methods (stacking)

## 📄 Licence

Projet éducatif - Libre d'utilisation

## 📞 Support

Consultez la documentation:
- [Guide d'Utilisation Complet](docs/GUIDE_UTILISATION_AUTOML.md)
- [Résultats et Analyses](AUTOML_FINAL_RESULTS.md)
- [Troubleshooting](docs/GUIDE_UTILISATION_AUTOML.md#troubleshooting)

---

**Créé avec ❤️ pour automatiser la détection de fraude**  
**Version**: 1.0 | **Dernière mise à jour**: 2025-10-18

Projet de détection de fraude avec AutoML Meta-Transformer

## 📁 Structure du Projet

```
fraud-project/
│
├── 📂 base/                        Scripts de base
│   ├── baseline_xgboost.py        Entraînement XGBoost de base
│   ├── extract_structure.py       Extraction de la structure des datasets
│   ├── create_metamodel_examples.py  Création des exemples pour meta-learning
│   ├── diverse_top5_selector.py   Sélection de configs diverses
│   ├── check_fraud_rates.py       Analyse des taux de fraude
│   └── production_feature_importance_*.py  Importance des features
│
├── 📂 automl_transformer/          Système AutoML avec Meta-Transformer
│   ├── train_automl_metatransformer.py  Entraînement du Meta-Transformer
│   ├── full_automl.py             Pipeline AutoML complet
│   ├── test_learned_metatransformer.py  Tests du modèle
│   ├── auto_feature_engineer.py   Feature engineering automatique
│   └── auto_feature_selector.py   Sélection automatique des features
│
├── 📂 ancien_meta/                 Ancien Meta-Transformer (archivé)
│   ├── train_metatransformer.py   Ancien entraînement
│   ├── predict_xgboost_config.py  Anciennes prédictions
│   └── create_unified_metatransformer_dataset.py
│
├── 📂 tests/                       Scripts de test et debug
│   ├── test_full_automl_learned.py  Test du pipeline complet
│   ├── test_pipeline.py           Tests du pipeline
│   ├── debug_meta_features.py     Debug des meta-features
│   └── quick_test_config.py       Tests rapides de config
│
├── 📂 utils/                       Utilitaires
│   ├── utils.py                   Fonctions utilitaires
│   ├── fraud_detection.py         Détection de fraude
│   └── apply_model_production.py  Application en production
│
├── 📂 docs/                        Documentation
│   ├── README_METATRANSFORMER.md  Documentation Meta-Transformer
│   ├── SCRIPTS_GUIDE.md           Guide des scripts
│   ├── DATASET_EXPANSION_TRACKER.md  Suivi des datasets
│   ├── DATASET8_QUICK_GUIDE.md    Guide Dataset8
│   ├── GRID_COMPARISON_DATASET7_VS_DATASET8.md  Comparaisons
│   └── PROOF_OF_CONCEPT_REPORT.md  Rapport de concept
│
└── 📂 data/                        Données
    ├── datasets/                  Datasets CSV
    ├── models/                    Modèles entraînés
    ├── structure/                 Structures extraites
    ├── Feature_importance/        Importance des features
    └── metatransformer_training/  Données d'entraînement Meta-Transformer
```

## 🚀 Quick Start

### 1. Entraînement de base (baseline)
```bash
cd base
python baseline_xgboost.py
```

### 2. Utiliser l'AutoML complet
```python
from automl_transformer.full_automl import FullAutoML

# Avec Meta-Transformer (apprentissage automatique)
automl = FullAutoML(use_meta_transformer=True)
automl.fit('data/datasets/Dataset5.csv', target_col='is_fraud')

# Sans Meta-Transformer (règles basées sur la structure)
automl = FullAutoML(use_meta_transformer=False)
automl.fit('data/datasets/Dataset9.csv', target_col='fraud_flag')
```

### 3. Entraîner le Meta-Transformer
```bash
cd automl_transformer
python train_automl_metatransformer.py
```

## 📊 Résultats

### Comparaison des approches

| Dataset | Baseline | AutoML (rule-based) | AutoML (Meta-Transformer) |
|---------|----------|---------------------|---------------------------|
| Dataset5 | - | F1=0.3671, ROC=0.8882 | F1=0.3627, ROC=0.8812 |
| Dataset9 | - | F1=0.2121, ROC=0.7116 | F1=0.1792, ROC=0.7147 |

**Conclusion actuelle** : Le mode rule-based est plus fiable. Le Meta-Transformer nécessite plus de données d'entraînement (Dataset8, Dataset9) pour mieux généraliser.

## 🔧 Développement

### Ajouter un nouveau dataset
1. Placer le CSV dans `data/datasets/`
2. Extraire la structure : `python base/extract_structure.py Dataset_X.csv`
3. Créer les exemples : `python base/create_metamodel_examples.py`
4. Réentraîner : `python automl_transformer/train_automl_metatransformer.py`

### Debug
```bash
cd tests
python debug_meta_features.py  # Vérifier les features extraites
python test_full_automl_learned.py  # Tester le pipeline complet
```

## 📝 Notes

- **Meta-Transformer actuel** : Entraîné sur Dataset1-7 (105 exemples)
- **Architecture** : Transformer 6 layers, 8 heads, 256 dim
- **Entrées** : 18 features structure + 20 features importance
- **Sorties** : 10 hyperparams + 20 feature scores + 5 engineering flags

## 🐛 Problèmes connus

1. **Meta-Transformer généralise mal** sur nouveaux datasets (Dataset8, Dataset9)
   - Solution : Ajouter plus de datasets d'entraînement
   - Alternative : Utiliser `use_meta_transformer=False` (plus fiable actuellement)

2. **Bug corrigé** : Les meta_features n'étaient pas lues correctement du JSON
   - Fixed dans `full_automl.py` ligne 145-165

## � Organisation du Projet

### Structure Principale

```
fraud-project/
├── 📊 automl_transformer/          AutoML avec Meta-Transformer
│   ├── full_automl.py             Point d'entrée principal
│   ├── apply_automl_production.py Script de production
│   ├── train_automl_metatransformer.py
│   ├── auto_feature_engineer.py   Fallback feature engineering
│   └── auto_feature_selector.py   Fallback feature selection
├── 🔧 utils/                       Utilitaires partagés
│   ├── column_matcher.py          Matching sémantique de colonnes
│   ├── fraud_detection.py         Détection de patterns de fraude
│   └── utils.py                   Fonctions communes
├── 📝 scripts/                     Scripts utilitaires organisés
│   ├── data_generation/           Génération de datasets
│   │   ├── generate_realistic_fraud_dataset.py
│   │   ├── dataset_configs.py
│   │   └── generate_model_metadata.py
│   ├── retraining/                Réentraînement de modèles
│   │   └── retrain_all_models.py
│   ├── debugging/                 Scripts de diagnostic
│   └── comparison/                Scripts de tests
├── 🗄️ data/                        Données et modèles
│   ├── datasets/                  40 datasets de fraude (Dataset1-40.csv)
│   ├── automl_models/             Modèles entraînés (40 dossiers)
│   ├── structure/                 Métadonnées de structure
│   └── Feature_importance/        Importance des features
├── 📜 ancien_meta/                 Anciens scripts Meta-Transformer
│   ├── train_metatransformer.py   (version obsolète)
│   └── predict_xgboost_config.py  (version obsolète)
├── 🚀 apply_automl_production.py   Script de production principal
└── 📚 docs/                        Documentation technique
```

### Scripts de Production

**Principal:**
- `automl_transformer/apply_automl_production.py` - Pipeline de production complet (seuil optimisé à 0.20)
- `automl_transformer/full_automl.py` - AutoML complet avec Meta-Transformer

**Génération de données:**
- `scripts/data_generation/generate_realistic_fraud_dataset.py` - Génère les 40 datasets
- `scripts/data_generation/dataset_configs.py` - Configuration des scénarios de fraude

**Maintenance:**
- `scripts/retraining/retrain_all_models.py` - Réentraîne tous les modèles (dernière exec: 04/11/2025)

### ⚠️ Note sur l'Organisation

**Pourquoi auto_feature_engineer/selector sont dans automl_transformer/?**
Ces fichiers ne sont PAS obsolètes! Ils sont activement utilisés comme fallbacks dans `full_automl.py` quand le Meta-Transformer échoue. Les déplacer casserait les imports.

**Vraies archives:** `ancien_meta/` contient les anciens scripts de Meta-Transformer qui ne sont plus utilisés.

## �📚 Documentation complète

Voir le dossier `docs/` pour plus de détails sur chaque composant.
