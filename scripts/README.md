# Scripts Organization

Ce dossier contient tous les scripts utilitaires du projet, organisés par catégorie.

## Structure

### 📊 data_generation/
Scripts de génération et manipulation des datasets de fraude.

- **generate_realistic_fraud_dataset.py**: Génère les 40 datasets de fraude réalistes avec différents scénarios
- **regenerate_structures_with_new_features.py**: Régénère les structures avec de nouvelles features
- **dataset_configs.py**: Configuration des différents types de fraudes et scénarios
- **generate_model_metadata.py**: Génère les métadonnées des modèles entraînés

### 🔄 retraining/
Scripts de réentraînement des modèles AutoML.

- **retrain_all_models.py**: Réentraîne séquentiellement tous les 40 modèles AutoML
  - Utilisé pour mettre à jour tous les modèles après modifications du code
  - Dernière exécution: 04/11/2025 03:13-03:23

### 🐛 debugging/
Scripts de diagnostic et correction des problèmes de données.

- **(Anciens scripts de debug supprimés après résolution des problèmes)**
- Ces scripts étaient utilisés pour:
  - Diagnostiquer Dataset23 (valeurs extrêmes 1.19e+14 FCFA)
  - Vérifier tous les datasets pour valeurs aberrantes
  - Appliquer transformations log pour corriger les données

### 📈 comparison/
Scripts de comparaison et tests de performance.

- **(Anciens scripts de comparaison de seuils supprimés)**
- Ces scripts étaient utilisés pour:
  - Tester différents seuils (15%, 25%, 70%)
  - Comparer les performances sur Dataset27 et Dataset36
  - Optimiser le seuil par défaut (maintenant fixé à 0.20)

## Note importante

Les modules **auto_feature_engineer.py** et **auto_feature_selector.py** restent dans `automl_transformer/` car ils sont activement utilisés comme fallbacks dans `full_automl.py`.

Les vrais scripts obsolètes (ancien meta-transformer) sont dans `ancien_meta/`:
- create_unified_metatransformer_dataset.py
- predict_xgboost_config.py
- train_metatransformer.py
