"""
Script unifié pour créer le dataset Meta-Transformer optimisé
Combine la création comprehensive, l'optimisation et la correction des fraud rates
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def load_csv_fraud_rate(dataset_name):
    """Calculer le vrai fraud rate à partir du CSV source"""
    csv_path = f'data/datasets/{dataset_name}.csv'
    if not os.path.exists(csv_path):
        print(f"⚠️  CSV non trouvé pour {dataset_name}")
        return 0.05  # Défaut
    
    try:
        df = pd.read_csv(csv_path)
        # Chercher la colonne target
        target_cols = [col for col in df.columns if 'target' in col.lower() or 'fraud' in col.lower() or 'label' in col.lower() or 'suspect' in col.lower()]
        
        if target_cols:
            target_col = target_cols[0]
            
            # Convertir les valeurs textuelles en binaire si nécessaire
            if df[target_col].dtype == 'object' or df[target_col].dtype == 'str':
                positive_values = ['yes', 'oui', 'true', 'fraud', '1', 1]
                df[target_col] = df[target_col].apply(
                    lambda x: 1 if str(x).strip().lower() in [str(v).lower() for v in positive_values] else 0
                )
            
            fraud_rate = df[target_col].astype(float).mean()
            print(f"   {dataset_name}: fraud_rate = {fraud_rate:.6f} (colonne: {target_col})")
            return fraud_rate
        else:
            # Essayer de détecter automatiquement (colonne binaire avec peu de 1)
            binary_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and set(df[col].unique()).issubset({0, 1, np.nan})]
            for col in binary_cols:
                rate = df[col].mean()
                if 0.01 <= rate <= 0.15:  # Taux de fraude réaliste
                    print(f"   {dataset_name}: fraud_rate = {rate:.6f} (détecté: {col})")
                    return rate
    except Exception as e:
        print(f"⚠️  Erreur lecture {dataset_name}: {e}")
    
    return 0.05  # Défaut

def extract_comprehensive_features(dataset_name, example_id):
    """Extraire toutes les features d'un exemple dataset avec gestion robuste et optimisée"""
    
    print(f"📊 Extraction {dataset_name} - exemple {example_id}")
    
    # 1. DONNÉES DE BASE
    data = {
        'dataset_name': dataset_name,
        'example_id': f"{dataset_name}_{example_id}",
        'created_timestamp': datetime.now().isoformat()
    }
    
    # 2. CHARGER LES FICHIERS SOURCE
    files = {
        'structure': f'data/structure/{dataset_name}_dataset_structure.json',  # PRIORITÉ: plus rapide et complet
        'csv': f'data/datasets/{dataset_name}.csv',
        'summary': f'data/metamodel_data/{dataset_name}_metamodel_summary.json',
        'examples': f'data/metamodel_data/{dataset_name}_metamodel_training_examples.json',
        'feature_importance': f'data/Feature_importance/{dataset_name}_production_feature_importance.json'
    }
    
    # 3. VALEURS PAR DÉFAUT POUR TOUTES LES FEATURES (ENRICHIES ET OPTIMISÉES)
    default_data = {
        # Dataset metadata (ENRICHI)
        'dataset_rows': 0, 'dataset_columns': 0, 'fraud_rate': 0.05, 'target_column': 'unknown',
        'imbalance_ratio': 20.0, 'dataset_size_category': 'medium',
        
        # Statistical features (ENRICHI)
        'actual_numeric_count': 0, 'actual_categorical_count': 0, 'actual_binary_count': 0,
        'actual_missing_rate': 0.0, 'actual_duplicate_rate': 0.0,
        'numeric_mean_std': 0.0, 'numeric_mean_skewness': 0.0, 'numeric_correlation_strength': 0.0,
        'numeric_mean_kurtosis': 0.0, 'numeric_variance_max': 0.0, 'numeric_variance_mean': 0.0,
        'correlation_with_target_max': 0.0, 'correlation_with_target_mean': 0.0,
        
        # Feature engineering (ENRICHI)
        'total_features_engineered': 0, 'top_feature_importance': 0.0, 'avg_feature_importance': 0.0,
        'feature_importance_entropy': 0.0, 'top5_feature_concentration': 0.0,
        'top10_feature_concentration': 0.0, 'feature_importance_std': 0.0,
        'feature_importance_gini': 0.0, 'num_dominant_features': 0,
        
        # Hyperparameters (CRITIQUES - INCHANGÉ)
        'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100, 'subsample': 1.0,
        'colsample_bytree': 1.0, 'gamma': 0.0, 'min_child_weight': 1, 'reg_alpha': 0.0, 'reg_lambda': 1.0, 'scale_pos_weight': 1.0,
        
        # Performance metrics (CRITIQUES - ENRICHI)
        'target_cv_score': 0.5, 'target_rank': example_id, 'target_score_category': 'unknown',
        'target_test_score': 0.5, 'target_precision': 0.5, 'target_recall': 0.5, 'target_roc_auc': 0.5,
        'overfitting_score': 0.0, 'score_stability': 0.0,
        
        # Derived features (ENRICHI)
        'complexity_score': 0.0, 'fraud_detection_suitability': 1.0, 'training_efficiency': 1.0,
        'feature_engineering_level': 'basic', 'recommended_use_case': 'development', 
        'structure_target_distribution': 0.05, 'dataset_quality_score': 0.5,
        'hyperparameter_diversity': 0.0, 'model_complexity': 0.0
    }
    data.update(default_data)
    
    # 4. EXTRACTION CSV (métadonnées dataset)
    if os.path.exists(files['csv']):
        try:
            df_csv = pd.read_csv(files['csv'])
            
            # Analyse statistique robuste (ENRICHIE)
            numeric_cols = df_csv.select_dtypes(include=[np.number])
            data.update({
                'dataset_rows': len(df_csv),
                'dataset_columns': len(df_csv.columns),
                'fraud_rate': load_csv_fraud_rate(dataset_name),  # VRAIE VALEUR
                'target_column': 'unknown',
                
                # Statistiques de base
                'actual_numeric_count': len(numeric_cols.columns),
                'actual_categorical_count': len(df_csv.select_dtypes(include=['object']).columns),
                'actual_missing_rate': df_csv.isnull().sum().sum() / (len(df_csv) * len(df_csv.columns)),
                
                # Features statistiques avancées avec protection contre erreurs
                'numeric_mean_std': numeric_cols.std().mean() if len(numeric_cols.columns) > 0 else 0,
                'numeric_mean_skewness': numeric_cols.skew().mean() if len(numeric_cols.columns) > 0 else 0,
                'numeric_mean_kurtosis': numeric_cols.kurtosis().mean() if len(numeric_cols.columns) > 0 else 0,
                'numeric_variance_max': numeric_cols.var().max() if len(numeric_cols.columns) > 0 else 0,
                'numeric_variance_mean': numeric_cols.var().mean() if len(numeric_cols.columns) > 0 else 0,
                'numeric_correlation_strength': abs(numeric_cols.corr()).mean().mean() if len(numeric_cols.columns) > 1 else 0,
                
                # Métadonnées enrichies
                'imbalance_ratio': (1 - data['fraud_rate']) / data['fraud_rate'] if data['fraud_rate'] > 0 else 20.0,
                'dataset_size_category': 'small' if len(df_csv) < 20000 else ('medium' if len(df_csv) < 50000 else 'large')
            })
        except Exception as e:
            print(f"⚠️  Erreur CSV {dataset_name}: {e}")
    
    # 5. EXTRACTION FEATURE IMPORTANCE (ROBUSTE ET ENRICHIE)
    if os.path.exists(files['feature_importance']):
        try:
            with open(files['feature_importance'], 'r') as f:
                fi_data = json.load(f)
            
            if 'production_feature_importance' in fi_data and fi_data['production_feature_importance']:
                prod_features = fi_data['production_feature_importance']
                importances = [f['importance'] for f in prod_features if 'importance' in f]
                
                if importances:
                    sorted_importances = sorted(importances, reverse=True)
                    total_importance = sum(importances)
                    
                    data.update({
                        'total_features_engineered': len(prod_features),
                        'top_feature_importance': max(importances),
                        'avg_feature_importance': np.mean(importances),
                        'feature_importance_std': np.std(importances),
                        'feature_importance_entropy': -sum(p * np.log(p + 1e-10) for p in importances if p > 0),
                        'top5_feature_concentration': sum(sorted_importances[:5]) / total_importance if total_importance > 0 else 0,
                        'top10_feature_concentration': sum(sorted_importances[:10]) / total_importance if total_importance > 0 else 0
                    })
                    
                    # NOUVEAU: Gini coefficient (inégalité distribution features)
                    n = len(sorted_importances)
                    gini = (2 * sum((i+1) * imp for i, imp in enumerate(sorted_importances))) / (n * total_importance) - (n + 1) / n if total_importance > 0 else 0
                    data['feature_importance_gini'] = max(0, gini)
                    
                    # NOUVEAU: Nombre de features dominantes (>10% importance totale)
                    data['num_dominant_features'] = sum(1 for imp in importances if imp > 0.1 * total_importance)
                    
        except Exception as e:
            print(f"⚠️  Erreur Feature Importance {dataset_name}: {e}")
    
    # 6. EXTRACTION EXAMPLES (HYPERPARAMÈTRES ET PERFORMANCE) - CORRIGÉE !
    if os.path.exists(files['examples']):
        try:
            with open(files['examples'], 'r') as f:
                examples_data = json.load(f)
            
            print(f"   🔍 Fichier examples chargé: {len(examples_data) if isinstance(examples_data, list) else 'Pas une liste'}")
            
            # CORRECTION CRITIQUE : Chercher l'exemple avec le bon rank, pas par index séquentiel
            if isinstance(examples_data, list):
                # Chercher l'exemple qui a le rank correspondant à example_id
                example = None
                for ex in examples_data:
                    if ('optimal_xgb_config' in ex and 
                        'performance' in ex['optimal_xgb_config'] and
                        ex['optimal_xgb_config']['performance'].get('rank') == example_id):
                        example = ex
                        break
                
                if example:
                    print(f"   📋 Exemple trouvé: ID {example.get('id', 'N/A')} avec rank {example_id}")
                    
                    # Extraction hyperparamètres (CRITIQUE !)
                    if 'optimal_xgb_config' in example and 'hyperparameters' in example['optimal_xgb_config']:
                        hyperparams = example['optimal_xgb_config']['hyperparameters']
                        data.update({
                            'max_depth': hyperparams.get('max_depth', 3),
                            'learning_rate': hyperparams.get('learning_rate', 0.1),
                            'n_estimators': hyperparams.get('n_estimators', 100),
                            'subsample': hyperparams.get('subsample', 1.0),
                            'colsample_bytree': hyperparams.get('colsample_bytree', 1.0),
                            'gamma': hyperparams.get('gamma', 0.0),
                            'min_child_weight': hyperparams.get('min_child_weight', 1),
                            'reg_alpha': hyperparams.get('reg_alpha', 0.0),
                            'reg_lambda': hyperparams.get('reg_lambda', 1.0),
                            'scale_pos_weight': hyperparams.get('scale_pos_weight', 1.0)
                        })
                        print(f"   ✅ Hyperparams extraits: max_depth={data['max_depth']}, lr={data['learning_rate']}, n_est={data['n_estimators']}")
                    else:
                        print(f"   ❌ Pas d'hyperparamètres dans l'exemple")
                    
                    # Extraction performance (CRITIQUE ET ENRICHIE!)
                    if 'optimal_xgb_config' in example and 'performance' in example['optimal_xgb_config']:
                        perf = example['optimal_xgb_config']['performance']
                        cv_score = perf.get('cv_score', 0.5)
                        test_score = perf.get('test_score', cv_score)
                        
                        data.update({
                            'target_cv_score': cv_score,
                            'target_rank': perf.get('rank', example_id),
                            'target_score_category': perf.get('score_category', 'unknown'),
                            'target_test_score': test_score,
                            'target_precision': perf.get('precision', 0.5),
                            'target_recall': perf.get('recall', 0.5),
                            'target_roc_auc': perf.get('roc_auc', 0.5)
                        })
                        
                        # NOUVEAU: Overfitting score (différence train-test)
                        train_score = perf.get('train_score', 1.0)
                        data['overfitting_score'] = max(0, train_score - test_score)
                        
                        # NOUVEAU: Score stability (similarité CV/test)
                        data['score_stability'] = 1.0 - abs(cv_score - test_score)
                        
                        print(f"   ✅ Performance: cv={cv_score:.4f}, test={test_score:.4f}, precision={data['target_precision']:.4f}, recall={data['target_recall']:.4f}, overfitting={data['overfitting_score']:.4f}")
                    else:
                        print(f"   ❌ Pas de performance dans l'exemple")
                    
                    # Extraction target column (NOUVEAU !) avec FALLBACK CSV
                    target_feature = None
                    if 'advanced_analytics' in example and 'feature_correlations' in example['advanced_analytics']:
                        target_feature = example['advanced_analytics']['feature_correlations'].get('target_feature')
                    
                    # FALLBACK: Si target_feature est null ou absent, chercher dans le CSV
                    if not target_feature or target_feature == 'unknown':
                        csv_file = f'data/datasets/{dataset_name}.csv'
                        if os.path.exists(csv_file):
                            try:
                                df_temp = pd.read_csv(csv_file, nrows=1)
                                # Chercher colonnes avec pattern fraud/target (pattern étendu)
                                fraud_cols = [col for col in df_temp.columns if any(x in col.lower() for x in 
                                    ['fraud', 'flag', 'label', 'suspect', 'suspicious', 'alert', 'aml', 'default', 
                                     'irregularity', 'manipulation', 'chargeback', 'depletes', 'skimming'])]
                                if fraud_cols:
                                    target_feature = fraud_cols[0]
                                    print(f"   🔍 Target column trouvée dans CSV: {target_feature}")
                            except:
                                pass
                    
                    data['target_column'] = target_feature if target_feature else 'unknown'
                    if target_feature:
                        print(f"   ✅ Target column extraite: {target_feature}")
                    else:
                        print(f"   ⚠️  Target column non trouvée (défaut: unknown)")
                else:
                    print(f"   ❌ Aucun exemple trouvé avec rank {example_id}")
            else:
                print(f"   ❌ Format examples_data incorrect")
                    
        except Exception as e:
            print(f"⚠️  Erreur Examples {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 7. CALCULS DÉRIVÉS (ROBUSTES ET ENRICHIS)
    try:
        # Complexité du dataset
        complexity_score = (data.get('dataset_rows', 1000) * data.get('total_features_engineered', 10)) / 1000000
        
        # Score de qualité basé sur plusieurs facteurs
        quality_factors = [
            1.0 - data.get('actual_missing_rate', 0),  # Moins de données manquantes = meilleur
            min(data.get('total_features_engineered', 0) / 50, 1.0),  # Plus de features engineering = meilleur
            min(data.get('fraud_rate', 0.05) * 10, 1.0),  # Fraud rate raisonnable = meilleur
            1.0 - min(data.get('numeric_correlation_strength', 0.5), 1.0)  # Moins de corrélation = meilleur
        ]
        dataset_quality = np.mean(quality_factors)
        
        # NOUVEAU: Hyperparameter diversity score (variabilité des hyperparams par rapport aux defaults)
        default_depth = 3
        default_lr = 0.1
        default_n_est = 100
        
        depth_diversity = abs(data.get('max_depth', default_depth) - default_depth) / default_depth
        lr_diversity = abs(data.get('learning_rate', default_lr) - default_lr) / default_lr
        n_est_diversity = abs(data.get('n_estimators', default_n_est) - default_n_est) / default_n_est
        
        hyperparameter_diversity = np.mean([depth_diversity, lr_diversity, n_est_diversity])
        
        # NOUVEAU: Model complexity score (combinaison depth, n_estimators, features)
        model_complexity = (
            data.get('max_depth', 3) / 10.0 +  # depth contribution
            data.get('n_estimators', 100) / 500.0 +  # n_estimators contribution
            data.get('total_features_engineered', 10) / 100.0  # features contribution
        ) / 3.0
        
        data.update({
            'complexity_score': complexity_score,
            'fraud_detection_suitability': min(data.get('fraud_rate', 0.05) * 20, 1.0),
            'training_efficiency': 1.0,
            'feature_engineering_level': 'advanced' if data.get('total_features_engineered', 0) > 20 else 'basic',
            'recommended_use_case': 'production' if dataset_quality > 0.7 else 'development',
            'structure_target_distribution': data.get('fraud_rate', 0.05),
            'dataset_quality_score': dataset_quality,
            'hyperparameter_diversity': hyperparameter_diversity,
            'model_complexity': model_complexity
        })
    except Exception as e:
        print(f"⚠️  Erreur calculs dérivés {dataset_name}: {e}")
    
    return data

def create_unified_metatransformer_dataset():
    """Créer le dataset Meta-Transformer unifié et optimisé"""
    
    print("🚀 CRÉATION DATASET META-TRANSFORMER UNIFIÉ")
    print("=" * 60)
    
    # 1. CRÉER DATASET COMPREHENSIVE
    print("\n1️⃣ CRÉATION DATASET COMPREHENSIVE...")
    
    all_examples = []
    # INCLURE TOUS LES DATASETS 1-30 pour le MVP
    datasets = [f'Dataset{i}' for i in range(1, 31)]
    
    print(f"📊 Traitement de {len(datasets)} datasets (Dataset1 à Dataset30)")
    
    for dataset in datasets:
        print(f"\n📁 Processing {dataset}...")
        
        # Déterminer le nombre d'exemples disponibles
        examples_file = f'data/metamodel_data/{dataset}_metamodel_training_examples.json'
        if os.path.exists(examples_file):
            try:
                with open(examples_file, 'r') as f:
                    examples_data = json.load(f)
                example_keys = [k for k in examples_data.keys() if k.startswith('example_')]
                num_examples = len(example_keys)
                print(f"   Trouvé {num_examples} exemples")
            except:
                num_examples = 15  # Défaut
        else:
            num_examples = 15  # Défaut
        
        # Extraire tous les exemples
        for i in range(1, min(num_examples + 1, 16)):  # Max 15 exemples par dataset
            try:
                example_data = extract_comprehensive_features(dataset, i)
                all_examples.append(example_data)
            except Exception as e:
                print(f"⚠️  Erreur exemple {i}: {e}")
    
    # Créer DataFrame
    df_comprehensive = pd.DataFrame(all_examples)
    print(f"\n✅ Dataset comprehensive créé: {df_comprehensive.shape}")
    
    # 2. OPTIMISATION - SUPPRIMER COLONNES VIDES
    print("\n2️⃣ OPTIMISATION - SUPPRESSION COLONNES VIDES...")
    
    # Identifier colonnes vides ou presque vides
    empty_columns = []
    for col in df_comprehensive.columns:
        if col in ['dataset_name', 'example_id', 'created_timestamp']:
            continue  # Garder les métadonnées
            
        if df_comprehensive[col].dtype == 'object':
            # Pour colonnes texte: vérifier valeurs vides
            if df_comprehensive[col].isnull().all() or (df_comprehensive[col] == '').all():
                empty_columns.append(col)
        else:
            # Pour colonnes numériques: vérifier zéros et NaN
            non_zero_count = (df_comprehensive[col] != 0).sum()
            non_null_count = df_comprehensive[col].notnull().sum()
            if non_zero_count == 0 or non_null_count == 0:
                empty_columns.append(col)
    
    print(f"   🗑️  Suppression de {len(empty_columns)} colonnes vides")
    for col in empty_columns[:5]:  # Afficher les 5 premières
        print(f"      • {col}")
    if len(empty_columns) > 5:
        print(f"      ... et {len(empty_columns)-5} autres")
    
    # Supprimer colonnes vides
    df_optimized = df_comprehensive.drop(columns=empty_columns)
    print(f"   ✅ Dataset optimisé: {df_optimized.shape}")
    
    # 2.5. SUPPRESSION COLONNES INUTILES
    print("\n2️⃣.5 SUPPRESSION COLONNES INUTILES...")
    
    # Colonnes inutiles identifiées (garder reg_lambda comme demandé)
    useless_cols = [
        'dataset_name',                 # Identifiant arbitraire sans valeur sémantique (NOUVEAU)
        'fraud_detection_suitability',  # Constante = 1.0
        'training_efficiency',          # Constante = 1.0  
        'feature_engineering_level',    # Constante = "advanced"
        'recommended_use_case',         # Constante = "development"
        'created_timestamp',            # Métadonnée sans valeur ML
        'example_id'                    # Métadonnée sans valeur ML
    ]
    
    # Supprimer les colonnes inutiles présentes
    cols_to_remove = [col for col in useless_cols if col in df_optimized.columns]
    if cols_to_remove:
        df_cleaned = df_optimized.drop(columns=cols_to_remove)
        print(f"   🗑️  Suppression de {len(cols_to_remove)} colonnes inutiles:")
        for col in cols_to_remove:
            if col == 'dataset_name':
                print(f"      • {col} ⭐ (améliore généralisation)")
            else:
                print(f"      • {col}")
    else:
        df_cleaned = df_optimized
        print("   ✅ Aucune colonne inutile trouvée")
        
    print(f"   ✅ Dataset nettoyé: {df_cleaned.shape}")

    # 2.6. CORRECTION DES RANKS (CRITIQUE!)
    print("\n2️⃣.6 CORRECTION RANKS BASÉS SUR SCORES...")
    
    # Note: dataset_name a été supprimé, donc on recalcule les ranks GLOBALEMENT
    # Cela force le modèle à apprendre des patterns universels plutôt que dataset-spécifiques
    df_cleaned = df_cleaned.sort_values('target_cv_score', ascending=False).copy()
    df_cleaned['target_rank'] = range(1, len(df_cleaned) + 1)
    
    # Recalculer score_category basé sur les nouveaux ranks globaux
    df_cleaned['target_score_category'] = df_cleaned['target_rank'].apply(
        lambda rank: 'excellent' if rank <= 25  # Top 25 (environ 25% des 105 exemples)
                    else 'good' if rank <= 50       # Top 50
                    else 'fair' if rank <= 80       # Top 80
                    else 'poor'                      # Reste
    )
    
    print(f"   ✅ Ranks globaux recalculés basés sur CV scores: {df_cleaned.shape}")
    print(f"   📊 Distribution: {df_cleaned['target_score_category'].value_counts().to_dict()}")
    
    # 2.7. NORMALISATION DES FEATURES (AMÉLIORE CONVERGENCE META-MODEL!)
    print("\n2️⃣.7 NORMALISATION FEATURES NUMÉRIQUES...")
    
    # Exclure targets et catégories
    exclude_cols = ['target_cv_score', 'target_test_score', 'target_precision', 'target_recall', 
                    'target_roc_auc', 'target_rank', 'target_score_category', 'target_column']
    
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
    
    # Créer versions normalisées (0-1 MinMax scaling)
    df_final = df_cleaned.copy()
    normalization_stats = {}
    
    for col in cols_to_normalize:
        col_min = df_cleaned[col].min()
        col_max = df_cleaned[col].max()
        col_range = col_max - col_min
        
        if col_range > 0:
            df_final[f'{col}_norm'] = (df_cleaned[col] - col_min) / col_range
            normalization_stats[col] = {'min': float(col_min), 'max': float(col_max), 'range': float(col_range)}
        else:
            df_final[f'{col}_norm'] = 0.5  # Neutre si constante
            normalization_stats[col] = {'min': float(col_min), 'max': float(col_max), 'range': 0.0, 'constant': True}
    
    print(f"   ✅ {len(cols_to_normalize)} features normalisées (_norm suffix)")
    print(f"   💾 Stats normalisation sauvegardées (pour production)")

    # 3. VÉRIFICATION FRAUD RATES
    print("\n3️⃣ VÉRIFICATION FRAUD RATES...")
    print(f"   Note: dataset_name supprimé, vérification globale uniquement")
    print(f"   Fraud rate moyen: {df_final['fraud_rate'].mean():.4f}")
    print(f"   Fraud rate min: {df_final['fraud_rate'].min():.4f}")
    print(f"   Fraud rate max: {df_final['fraud_rate'].max():.4f}")
    
    # Garder l'ancienne logique pour les métadonnées (utiliser la source originale pour TOUS les datasets 1-30)
    fraud_rates = {}
    datasets_list = [f'Dataset{i}' for i in range(1, 31)]
    for dataset in datasets_list:
        rate = load_csv_fraud_rate(dataset)
        fraud_rates[dataset] = rate
    
    print("   Fraud rates par dataset (source) - Échantillon:")
    for i, (dataset, rate) in enumerate(list(fraud_rates.items())[:10]):  # Afficher les 10 premiers
        print(f"      • {dataset}: {rate:.6f} ({rate*100:.2f}%)")
    if len(fraud_rates) > 10:
        print(f"      ... et {len(fraud_rates)-10} autres datasets")
    
    # 4. SAUVEGARDE
    print("\n4️⃣ SAUVEGARDE...")
    
    # Créer dossier si nécessaire
    os.makedirs('data/metatransformer_training', exist_ok=True)
    
    # Sauvegarder dataset final AVEC features normalisées
    output_file = 'data/metatransformer_training/unified_metatransformer_dataset.csv'
    df_final.to_csv(output_file, index=False)
    
    # Sauvegarder stats de normalisation (CRITIQUE pour production!)
    norm_stats_file = 'data/metatransformer_training/normalization_stats.json'
    with open(norm_stats_file, 'w') as f:
        json.dump(normalization_stats, f, indent=2)
    
    print(f"   ✅ Dataset sauvegardé: {output_file}")
    print(f"   ✅ Stats normalisation: {norm_stats_file}")
    
    # Sauvegarder métadonnées ENRICHIES
    metadata = {
        'creation_date': datetime.now().isoformat(),
        'total_examples': len(df_final),
        'total_features': len(df_final.columns),
        'datasets_included': datasets_list,  # Tous les datasets 1-30
        'removed_empty_columns': empty_columns,
        'removed_useless_columns': cols_to_remove,
        'fraud_rates': fraud_rates,  # Dict avec tous les fraud rates
        'normalization_applied': True,  # NOUVEAU
        'normalized_features_count': len(cols_to_normalize),  # NOUVEAU
        'statistics': {
            'fraud_rate_mean': float(df_final['fraud_rate'].mean()),
            'fraud_rate_std': float(df_final['fraud_rate'].std()),
            'fraud_rate_min': float(df_final['fraud_rate'].min()),
            'fraud_rate_max': float(df_final['fraud_rate'].max()),
            'cv_score_mean': float(df_final['target_cv_score'].mean()),
            'cv_score_std': float(df_final['target_cv_score'].std()),
            'score_category_distribution': df_final['target_score_category'].value_counts().to_dict()
        },
        'feature_categories': {
            'dataset_metadata': [col for col in df_final.columns if any(x in col for x in ['rows', 'columns', 'fraud_rate', 'imbalance', 'size'])],
            'statistical_features': [col for col in df_final.columns if any(x in col for x in ['numeric', 'categorical', 'missing', 'correlation', 'variance', 'skewness', 'kurtosis'])],
            'feature_engineering': [col for col in df_final.columns if any(x in col for x in ['features_engineered', 'importance', 'entropy', 'concentration', 'gini', 'dominant'])],
            'model_hyperparameters': [col for col in df_final.columns if any(x in col for x in ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'gamma', 'reg_'])],
            'performance_metrics': [col for col in df_final.columns if any(x in col for x in ['cv_score', 'test_score', 'precision', 'recall', 'roc_auc', 'overfitting', 'stability'])],
            'derived_features': [col for col in df_final.columns if any(x in col for x in ['complexity', 'quality', 'diversity', 'suitability'])],
            'normalized_features': [col for col in df_final.columns if col.endswith('_norm')]
        }
    }
    
    metadata_file = 'data/metatransformer_training/unified_dataset_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 5. RAPPORT FINAL
    print("\n5️⃣ RAPPORT FINAL")
    print("=" * 40)
    print(f"✅ Dataset créé: {output_file}")
    print(f"📊 Dimensions: {df_cleaned.shape[0]} exemples × {df_cleaned.shape[1]} features")
    print(f"🗑️  Colonnes vides supprimées: {len(empty_columns)}")
    print(f"🗑️  Colonnes inutiles supprimées: {len(cols_to_remove)}")
    # fraud_rates est maintenant un dict {dataset: rate}
    rates_values = list(fraud_rates.values())
    print(f"📈 Fraud rates réalistes: {min(rates_values):.1%} - {max(rates_values):.1%}")
    print(f"💾 Métadonnées: {metadata_file}")
    
    print(f"\n🎯 DATASET PRÊT pour entraînement Meta-Transformer!")
    
    return df_cleaned, metadata

if __name__ == "__main__":
    df_final, metadata = create_unified_metatransformer_dataset()