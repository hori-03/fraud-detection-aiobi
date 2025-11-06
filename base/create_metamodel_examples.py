# -*- coding: utf-8 -*-
"""
Créateur d'exemples pour méta-modèle
SCRIPT ESSENTIEL qui combine structure du dataset + configurations XGBoost
pour créer les données d'entraînement du méta-modèle Transformers
"""

import sys
import json
import pandas as pd
import numpy as np
import joblib
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_dataset_structure(dataset_name="Dataset8"):
    """Charger ou créer la structure du dataset"""
    structure_file = f'data/structure/{dataset_name}_dataset_structure.json'
    try:
        with open(structure_file, 'r', encoding='utf-8') as f:
            structure = json.load(f)
        print(f"✅ Structure du dataset {dataset_name} chargée")
        return structure
    except FileNotFoundError:
        print(f"📊 Structure non trouvée pour {dataset_name}, création en cours...")
        # Lancer extract_structure.py avec le bon dataset
        import subprocess
        subprocess.run(['python', 'extract_structure.py', f'data/datasets/{dataset_name}.csv'])
        
        # Recharger après création
        with open(structure_file, 'r', encoding='utf-8') as f:
            structure = json.load(f)
        return structure

def load_diverse_configs(dataset_name="Dataset8"):
    """Charger les configurations diversifiées"""
    config_file = f'data/top5/{dataset_name}_diverse_top15_selection.json'
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        print(f"🎯 {len(configs)} configurations diversifiées chargées pour {dataset_name}")
        return configs
    except FileNotFoundError:
        # Essayer l'ancien format top5 en fallback
        old_config_file = f'data/top5/{dataset_name}_diverse_top5_selection.json'
        try:
            with open(old_config_file, 'r', encoding='utf-8') as f:
                configs = json.load(f)
            print(f"🎯 {len(configs)} configurations (ancien format top5) chargées pour {dataset_name}")
            return configs
        except FileNotFoundError:
            print(f"❌ Configurations diversifiées non trouvées pour {dataset_name}")
            print(f"   Lancez d'abord: python baseline_xgboost.py puis python diverse_top15_selector.py {dataset_name}")
            return None

def extract_advanced_performance_metrics(dataset_name):
    """Extraire métriques de performance avancées depuis GridSearch results"""
    results_file = f'data/results/{dataset_name}_grid_search_results.json'
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Analyser la variance et stabilité des performances
        scores = [r.get('mean_test_score', r.get('mean_test_f1', 0)) for r in results]
        std_scores = [r.get('std_test_score', r.get('std_test_f1', 0)) for r in results]
        
        performance_analysis = {
            'total_configurations_tested': len(results),
            'best_score': max(scores),
            'worst_score': min(scores),
            'score_range': max(scores) - min(scores),
            'average_score': np.mean(scores),
            'score_std': np.std(scores),
            'stable_configs_count': len([s for s in std_scores if s < 0.05]),
            'unstable_configs_count': len([s for s in std_scores if s > 0.1]),
            'performance_variance_analysis': {
                'low_variance_configs': len([s for s in std_scores if s < 0.03]),
                'medium_variance_configs': len([s for s in std_scores if 0.03 <= s <= 0.07]),
                'high_variance_configs': len([s for s in std_scores if s > 0.07])
            }
        }
        
        return performance_analysis
    except Exception as e:
        print(f"Erreur extraction métriques {dataset_name}: {e}")
        return {}

def extract_feature_correlations(dataset_name):
    """Analyser les corrélations entre features"""
    try:
        df = pd.read_csv(f'data/datasets/{dataset_name}.csv')
        
        # Identifier la variable cible - Support pour tous les datasets
        target_candidates = ['is_fraud', 'fraud_flag', 'suspicious_flag', 'label_suspect', 
                           'is_fraudulent', 'fraud', 'label', 'target',
                           'flagged_suspicious', 'flagged_anomaly', 'skimming_detected', 'aml_flagged']
        target_col = None
        for candidate in target_candidates:
            if candidate in df.columns:
                target_col = candidate
                break
        
        # Si pas trouvé dans la liste, chercher par pattern
        if not target_col:
            for col in df.columns:
                col_lower = col.lower()
                if ('fraud' in col_lower or 'suspect' in col_lower or 
                    'flagged' in col_lower or 'skimming' in col_lower or 
                    'aml' in col_lower or 'anomaly' in col_lower):
                    target_col = col
                    break
        
        # Preprocessing pour analyse corrélation
        df_analysis = df.copy()
        
        # Encoder les variables catégorielles pour corrélation
        categorical_cols = df_analysis.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != target_col:
                try:
                    le = LabelEncoder()
                    df_analysis[col] = le.fit_transform(df_analysis[col].astype(str))
                except Exception as e:
                    print(f"Erreur encodage {col}: {e}")
                    # Supprimer la colonne si elle ne peut pas être encodée
                    df_analysis = df_analysis.drop(columns=[col])
        
        # Encoder également la colonne cible si elle est catégorielle
        if target_col and df_analysis[target_col].dtype == 'object':
            try:
                le_target = LabelEncoder()
                df_analysis[target_col] = le_target.fit_transform(df_analysis[target_col].astype(str))
            except Exception as e:
                print(f"Erreur encodage target {target_col}: {e}")
                return {}
        
        # FIX: Supprimer les colonnes avec variance 0 pour éviter NaN dans les corrélations
        numeric_cols = df_analysis.select_dtypes(include=['number']).columns
        zero_var_cols = [col for col in numeric_cols if df_analysis[col].std() == 0]
        if zero_var_cols:
            df_analysis = df_analysis.drop(columns=zero_var_cols)
        
        # Calculer matrice de corrélation
        correlation_matrix = df_analysis.corr()
        
        # Features les plus corrélées avec la cible
        if target_col and target_col in correlation_matrix.columns:
            target_correlations = correlation_matrix[target_col].abs().sort_values(ascending=False)
            top_correlated_features = target_correlations.head(10).to_dict()
        else:
            top_correlated_features = {}
        
        # Identifier features redondantes (haute corrélation entre elles)
        redundant_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                if abs(correlation_matrix.iloc[i, j]) > 0.8 and col1 != target_col and col2 != target_col:
                    redundant_pairs.append({
                        'feature1': col1,
                        'feature2': col2,
                        'correlation': float(correlation_matrix.iloc[i, j])
                    })
        
        correlation_analysis = {
            'total_features_analyzed': len(correlation_matrix.columns),
            'target_feature': target_col,
            'top_correlated_with_target': {k: float(v) for k, v in top_correlated_features.items()},
            'redundant_feature_pairs': redundant_pairs[:10],  # Top 10 most redundant
            'high_correlation_threshold': 0.8,
            'correlation_matrix_stats': {
                'mean_correlation': float(np.nanmean(correlation_matrix.values[correlation_matrix.values != 1])),
                'max_correlation': float(np.nanmax(correlation_matrix.values[correlation_matrix.values != 1])),
                'min_correlation': float(np.nanmin(correlation_matrix.values[correlation_matrix.values != 1]))
            }
        }
        
        return correlation_analysis
        
    except Exception as e:
        print(f"Erreur analyse corrélations {dataset_name}: {e}")
        return {}

def extract_business_patterns(dataset_name):
    """Extraire patterns business spécifiques à la détection de fraude"""
    try:
        df = pd.read_csv(f'data/datasets/{dataset_name}.csv')
        
        # Identifier la variable cible
        target_candidates = ['is_fraud', 'fraud_flag', 'suspicious_flag', 'label_suspect']
        target_col = None
        for candidate in target_candidates:
            if candidate in df.columns:
                target_col = candidate
                break
        
        if not target_col:
            return {}
        
        # Analyser la distribution des fraudes
        fraud_rate = df[target_col].value_counts(normalize=True)
    
        # Gérer différents formats de valeurs cibles (numériques ou textuelles)
        fraud_proportion = 0
        normal_proportion = 0
        
        # Essayer d'identifier le taux de fraude selon différents formats
        fraud_values = [1, '1', 'yes', 'oui', 'fraud', 'true', 'True', 'YES', 'OUI']
        normal_values = [0, '0', 'no', 'non', 'normal', 'false', 'False', 'NO', 'NON']
        
        for val in fraud_values:
            if val in fraud_rate.index:
                fraud_proportion = float(fraud_rate[val])
                break
        
        for val in normal_values:
            if val in fraud_rate.index:
                normal_proportion = float(fraud_rate[val])
                break
        
        business_patterns = {
            'fraud_distribution': {
                'total_transactions': len(df),
                'fraud_rate': fraud_proportion,
                'normal_rate': normal_proportion,
                'unique_target_values': list(fraud_rate.index)
            }
        }
        
        # Patterns temporels si disponibles
        time_columns = [col for col in df.columns if any(word in col.lower() for word in ['hour', 'time', 'date'])]
        if time_columns:
            time_patterns = {}
            for time_col in time_columns[:2]:  # Analyser max 2 colonnes temporelles
                try:
                    if df[time_col].dtype in ['int64', 'float64']:
                        # Créer une variable binaire pour les fraudes (0/1)
                        df_temp = df.copy()
                        df_temp['fraud_binary'] = 0
                        for val in fraud_values:
                            if val in df_temp[target_col].values:
                                df_temp.loc[df_temp[target_col] == val, 'fraud_binary'] = 1
                        
                        fraud_by_time = df_temp.groupby(time_col)['fraud_binary'].agg(['count', 'mean'])
                        time_patterns[time_col] = {
                            'fraud_rate_by_value': fraud_by_time['mean'].to_dict(),
                            'transaction_count_by_value': fraud_by_time['count'].to_dict()
                        }
                except Exception as e:
                    print(f"Erreur patterns temporels pour {time_col}: {e}")
            business_patterns['temporal_patterns'] = time_patterns
        
        # Patterns montants si disponibles  
        amount_columns = [col for col in df.columns if any(word in col.lower() for word in ['amount', 'montant'])]
        if amount_columns:
            amount_col = amount_columns[0]
            try:
                if df[amount_col].dtype in ['int64', 'float64']:
                    # Créer une variable binaire pour les fraudes (0/1)
                    df_temp = df.copy()
                    df_temp['fraud_binary'] = 0
                    for val in fraud_values:
                        if val in df_temp[target_col].values:
                            df_temp.loc[df_temp[target_col] == val, 'fraud_binary'] = 1
                    
                    # Créer des bins de montants
                    df_temp['amount_bin'] = pd.cut(df_temp[amount_col], bins=10, labels=False)
                    fraud_by_amount = df_temp.groupby('amount_bin')['fraud_binary'].agg(['count', 'mean'])
                    
                    business_patterns['amount_patterns'] = {
                        'fraud_rate_by_amount_bin': fraud_by_amount['mean'].to_dict(),
                        'transaction_count_by_amount_bin': fraud_by_amount['count'].to_dict(),
                        'amount_column_analyzed': amount_col
                    }
            except Exception as e:
                print(f"Erreur patterns montants pour {amount_col}: {e}")
        
        return business_patterns
        
    except Exception as e:
        print(f"Erreur analyse patterns business {dataset_name}: {e}")
        return {}

def extract_model_robustness(dataset_name):
    """Analyser la robustesse et stabilité du modèle"""
    try:
        # Charger le modèle entraîné
        model_path = f'data/models/{dataset_name}_xgb_model.joblib'
        model_data = joblib.load(model_path)
        
        if hasattr(model_data, 'best_estimator_'):
            best_model = model_data.best_estimator_
            best_params = model_data.best_params_
            cv_results = model_data.cv_results_
        else:
            best_model = model_data
            best_params = {}
            cv_results = {}
        
        # Note: Feature importances détaillées sont dans data/Feature_importance/
        # On garde seulement les métadonnées essentielles ici
        
        # Analyser la convergence si CV results disponibles
        convergence_analysis = {}
        if cv_results:
            test_scores = cv_results.get('mean_test_score', [])
            if test_scores:
                convergence_analysis = {
                    'score_progression_stability': float(np.std(test_scores)),
                    'best_score_frequency': len([s for s in test_scores if s >= max(test_scores) * 0.95]),
                    'poor_score_frequency': len([s for s in test_scores if s <= max(test_scores) * 0.8])
                }
        
        robustness_analysis = {
            'model_type': str(type(best_model).__name__),
            'best_hyperparameters': best_params,
            'convergence_analysis': convergence_analysis,
            'parameter_sensitivity': {
                'critical_params': ['max_depth', 'learning_rate', 'subsample'],
                'stable_params': ['colsample_bytree', 'reg_alpha', 'reg_lambda']
            }
        }
        
        return robustness_analysis
        
    except Exception as e:
        print(f"Erreur analyse robustesse {dataset_name}: {e}")
        return {}

def create_metamodel_examples(dataset_name="Dataset7"):
    """Créer les exemples structure + config pour le méta-modèle"""
    
    print("🏗️ CRÉATION EXEMPLES MÉTA-MODÈLE ENRICHIS")
    print("=" * 50)
    
    # Charger structure et configurations
    structure = load_dataset_structure(dataset_name)
    configs = load_diverse_configs(dataset_name)
    
    if configs is None:
        return None
    
    print("📊 Extraction des analyses avancées...")
    
    # Extraire toutes les analyses avancées
    performance_metrics = extract_advanced_performance_metrics(dataset_name)
    feature_correlations = extract_feature_correlations(dataset_name)
    business_patterns = extract_business_patterns(dataset_name)
    model_robustness = extract_model_robustness(dataset_name)
    
    # Charger feature importance production si disponible
    production_features = {}
    importance_files = [
        f'data/Feature_importance/{dataset_name}_production_feature_importance.json'
    ]
    for imp_file in importance_files:
        try:
            with open(imp_file, 'r') as f:
                production_features = json.load(f)
            break
        except:
            pass
    
    print(f"✅ Analyses extraites pour {dataset_name}")
    
    # Créer les exemples combinés
    examples = []
    
    for i, config in enumerate(configs):
        params = config['params']
        
        # Détecter la colonne de score (compatible avec multiple scoring)
        score_column = None
        possible_score_columns = ['mean_test_score', 'mean_test_f1', 'mean_test_roc_auc', 'mean_test_precision']
        
        for col in possible_score_columns:
            if col in config:
                score_column = col
                break
                
        if score_column is None:
            available_keys = [k for k in config.keys() if 'mean_test' in k]
            raise KeyError(f"Aucune colonne de score trouvée dans config {i}. Colonnes mean_test disponibles: {available_keys}")
        
        score = config[score_column]
        
        # Créer l'exemple enrichi complet
        example = {
            "id": i + 1,
            "dataset_name": dataset_name,
            
            # Structure dataset enrichie - UTILISE LES 18 FEATURES STANDARDISÉES
            "dataset_structure": {
                "basic_info": {
                    "n_rows": structure['n_rows'],
                    "n_cols": structure['n_cols'],
                    "dataset_size_category": "large" if structure['n_rows'] > 20000 else ("medium" if structure['n_rows'] > 10000 else "small")
                },
                "meta_transformer_features": {
                    # LES 18 FEATURES POUR LE META-TRANSFORMER (version v2.0)
                    "rows": structure['n_rows'],
                    "columns": structure['n_cols'],
                    "fraud_rate": structure.get('class_balance', {}).get('fraud_ratio', 0.0) if structure.get('class_balance') else 0.0,
                    "missing_rate": structure.get('missing_rate', 0.0),
                    "duplicate_rate": structure.get('duplicate_rate', 0.0),
                    "numeric_columns": structure.get('feature_summary', {}).get('numerical_features', 0),
                    "categorical_columns": structure.get('feature_summary', {}).get('categorical_features', 0),
                    "correlation_max": structure.get('correlation_max', 0.0),
                    "correlation_min": structure.get('correlation_min', 0.0),
                    "correlation_mean": structure.get('correlation_mean', 0.0),
                    "variance_max": structure.get('variance_max', 0.0),
                    "variance_min": structure.get('variance_min', 0.0),
                    "variance_mean": structure.get('variance_mean', 0.0),
                    "skewness_max": structure.get('skewness_max', 0.0),
                    "skewness_min": structure.get('skewness_min', 0.0),
                    "skewness_mean": structure.get('skewness_mean', 0.0),
                    "kurtosis_max": structure.get('kurtosis_max', 0.0),
                    "kurtosis_mean": structure.get('kurtosis_mean', 0.0)
                },
                "feature_summary": {
                    "numerical_features": structure.get('feature_summary', {}).get('numerical_features', 0),
                    "categorical_features": structure.get('feature_summary', {}).get('categorical_features', 0),
                    "total_missing": sum(col.get('missing', 0) for col in structure['columns']),
                    "avg_unique_values": structure.get('feature_summary', {}).get('avg_unique_values', 0),
                    "high_cardinality_features": structure.get('feature_summary', {}).get('high_cardinality_features', 0)
                },
                "data_quality": {
                    "missing_data_ratio": structure.get('missing_rate', 0.0),
                    "feature_density": structure['n_cols'] / structure['n_rows'] if structure['n_rows'] > 0 else 0
                }
            },
            
            # Configuration XGBoost optimale
            "optimal_xgb_config": {
                "hyperparameters": {
                    "max_depth": params['max_depth'],
                    "learning_rate": params['learning_rate'],
                    "subsample": params['subsample'],
                    "colsample_bytree": params['colsample_bytree'],
                    "gamma": params['gamma'],
                    "min_child_weight": params['min_child_weight'],
                    "n_estimators": params['n_estimators'],
                    "scale_pos_weight": params['scale_pos_weight'],
                    "reg_alpha": params.get('reg_alpha', 0),  # Régularisation L1
                    "reg_lambda": params.get('reg_lambda', 1)  # Régularisation L2
                },
                "performance": {
                    "cv_score": score,
                    "rank": i + 1,
                    "score_category": "excellent" if score >= 0.9 else ("good" if score >= 0.8 else ("fair" if score >= 0.7 else "poor"))
                }
            },
            
            # Analyses avancées
            "advanced_analytics": {
                "performance_analysis": performance_metrics,
                "feature_correlations": feature_correlations,
                "business_patterns": business_patterns,
                "model_robustness": {
                    "model_type": model_robustness.get('model_type', ''),
                    "convergence_analysis": model_robustness.get('convergence_analysis', {}),
                    "parameter_sensitivity": model_robustness.get('parameter_sensitivity', {})
                },
                "production_readiness": production_features.get('recommendations', {}),
                "feature_importance_reference": {
                    "source_file": f"data/Feature_importance/{dataset_name}_production_feature_importance.json",
                    "total_production_features": len(production_features.get('production_feature_importance', [])),
                    "total_leaky_features": len(production_features.get('excluded_feature_importance', [])),
                    "production_viability": production_features.get('recommendations', {}).get('model_viability', 'unknown'),
                    "top_5_features": production_features.get('recommendations', {}).get('top_5_production_features', [])
                }
            },
            
            # Méta-informations pour le Transformer
            "meta_context": {
                "complexity_score": (structure['n_rows'] * structure['n_cols']) / 1000000,  # Normalized complexity
                "fraud_detection_suitability": business_patterns.get('fraud_distribution', {}).get('fraud_rate', 0) * 100,
                "training_efficiency": performance_metrics.get('stable_configs_count', 0) / max(performance_metrics.get('total_configurations_tested', 1), 1),
                "feature_engineering_level": "advanced" if structure['n_cols'] > 25 else ("intermediate" if structure['n_cols'] > 15 else "basic"),
                "recommended_use_case": "production" if production_features.get('recommendations', {}).get('model_viability') in ['EXCELLENT', 'GOOD'] else "development"
            }
        }
        
        examples.append(example)
        
        print(f"✅ Exemple {i+1}: depth={params['max_depth']}, lr={params['learning_rate']:.3f}, score={score:.4f}")
    
    return examples

def save_examples(examples, dataset_name="Dataset7"):
    """Sauvegarder les exemples dans différents formats"""
    
    if examples is None:
        return
    
    # Format complet pour méta-modèle
    training_file = f'data/metamodel_data/{dataset_name}_metamodel_training_examples.json'
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    # Créer un résumé global enrichi pour le méta-modèle (sécurisé)
    if not examples:
        print("⚠️ Aucun exemple généré, impossible de créer le résumé global")
        return examples
    
    # Accès sécurisé au premier exemple
    first_example = examples[0]
    scores = [ex['optimal_xgb_config']['performance']['cv_score'] for ex in examples]
    
    # Extraire les analyses depuis le premier exemple de façon sécurisée
    performance_metrics = first_example.get('advanced_analytics', {}).get('performance_analysis', {})
    feature_correlations = first_example.get('advanced_analytics', {}).get('feature_correlations', {})
    business_patterns = first_example.get('advanced_analytics', {}).get('business_patterns', {})
    production_features = first_example.get('advanced_analytics', {}).get('production_readiness', {})
    
    # Variables pour améliorer la lisibilité et éviter les warnings
    score_std = performance_metrics.get('score_std', 0)
    redundant_pairs_count = len(feature_correlations.get('redundant_feature_pairs', []))
    model_viability = production_features.get('model_viability', 'unknown')
    # FIX: Récupérer fraud_rate depuis dataset_structure au lieu de business_patterns
    fraud_rate = first_example['dataset_structure']['meta_transformer_features'].get('fraud_rate', 0)
    
    global_summary = {
        "dataset_overview": {
            "name": dataset_name,
            "total_configurations": len(examples),
            "best_score": max(scores) if scores else 0,
            "dataset_characteristics": first_example['dataset_structure']['basic_info'],
            "complexity_assessment": first_example['meta_context']['complexity_score'],
            "production_readiness": first_example['meta_context']['recommended_use_case']
        },
        "performance_insights": performance_metrics,
        "business_context": {
            "fraud_rate": fraud_rate,
            "temporal_patterns_available": len(business_patterns.get('temporal_patterns', {})) > 0,
            "amount_patterns_available": 'amount_patterns' in business_patterns
        },
        "feature_analysis_summary": {
            "correlation_complexity": redundant_pairs_count,
            "production_feature_ratio": production_features.get('feature_analysis', {}).get('production_importance_ratio', 0),
            "top_predictive_features": feature_correlations.get('top_correlated_with_target', {})
        },
        "model_stability": {
            "model_type": first_example.get('advanced_analytics', {}).get('model_robustness', {}).get('model_type', ''),
            "parameter_sensitivity": first_example.get('advanced_analytics', {}).get('model_robustness', {}).get('parameter_sensitivity', {}),
            "convergence_stability": len(first_example.get('advanced_analytics', {}).get('model_robustness', {}).get('convergence_analysis', {})) > 0
        },
        "transformer_training_recommendations": {
            "focus_areas": [
                "hyperparameter_sensitivity" if score_std > 0.05 else "stable_performance",
                "feature_selection" if redundant_pairs_count > 5 else "feature_engineering",
                "production_optimization" if model_viability == 'POOR' else "deployment_ready"
            ],
            "complexity_level": first_example['meta_context']['feature_engineering_level'],
            "recommended_attention": "high" if fraud_rate < 0.1 else "standard"
        }
    }
    
    # Format simplifié pour analyse rapide
    simplified = []
    for ex in examples:
        simple = {
            "config_id": ex['id'],
            "dataset_summary": f"Rows: {ex['dataset_structure']['basic_info']['n_rows']}, Features: {ex['dataset_structure']['basic_info']['n_cols']}, Complexity: {ex['meta_context']['complexity_score']:.2f}",
            "best_config": f"depth={ex['optimal_xgb_config']['hyperparameters']['max_depth']}, lr={ex['optimal_xgb_config']['hyperparameters']['learning_rate']}, score={ex['optimal_xgb_config']['performance']['cv_score']:.4f}",
            "production_ready": ex['meta_context']['recommended_use_case'] == 'production',
            "performance_category": ex['optimal_xgb_config']['performance']['score_category']
        }
        simplified.append(simple)
    
    summary_file = f'data/metamodel_data/{dataset_name}_metamodel_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "global_analysis": global_summary,
            "configuration_summaries": simplified
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Fichiers créés pour {dataset_name}:")
    print(f"   • {dataset_name}_metamodel_training_examples.json - Format complet pour entraînement")
    print(f"   • {dataset_name}_metamodel_summary.json - Format résumé pour analyse")

def create_training_prompt(dataset_name="Dataset7"):
    """Créer un prompt d'exemple pour Transformers"""
    
    try:
        training_file = f'data/metamodel_data/{dataset_name}_metamodel_training_examples.json'
        with open(training_file, 'r') as f:
            examples = json.load(f)
        
        if examples:
            example = examples[0]  # Premier exemple
            
            # Format texte pour Transformers
            dataset_info = example['dataset_structure']
            config_info = example['optimal_xgb_config']
            
            prompt = f"""Dataset Analysis:
- Samples: {dataset_info['basic_info']['n_rows']:,}
- Features: {dataset_info['basic_info']['n_cols']}
- Numerical features: {dataset_info['feature_summary']['numerical_features']}
- Categorical features: {dataset_info['feature_summary']['categorical_features']}
- Missing values: {dataset_info['feature_summary']['total_missing']}
- Average unique values per feature: {dataset_info['feature_summary']['avg_unique_values']:.1f}

Optimal XGBoost Configuration:
max_depth: {config_info['hyperparameters']['max_depth']}
learning_rate: {config_info['hyperparameters']['learning_rate']}
subsample: {config_info['hyperparameters']['subsample']}
colsample_bytree: {config_info['hyperparameters']['colsample_bytree']}
n_estimators: {config_info['hyperparameters']['n_estimators']}

Performance: {config_info['performance']['cv_score']:.4f}"""
            
            prompt_file = f'data/metamodel_data/{dataset_name}_training_prompt_example.txt'
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            print(f"✅ Exemple de prompt créé: {dataset_name}_training_prompt_example.txt")
            
    except Exception as e:
        print(f"❌ Erreur création prompt: {e}")

if __name__ == "__main__":
    import sys
    
    print("🏗️ CRÉATEUR D'EXEMPLES MÉTA-MODÈLE")
    print("=" * 45)
    print("Ce script combine structure dataset + configs XGBoost")
    print("pour créer les données d'entraînement du méta-modèle")
    print("Usage: python create_metamodel_examples.py [DATASET_NAME]")
    
    # Déterminer le nom du dataset
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "Dataset12"
    print(f"📊 Dataset: {dataset_name}")
    
    # Créer les exemples
    examples = create_metamodel_examples(dataset_name)
    
    # Sauvegarder
    save_examples(examples, dataset_name)
    
    # Créer prompt d'exemple
    create_training_prompt(dataset_name)
    
    print(f"\n🎯 RÉSUMÉ pour {dataset_name}:")
    print(f"✅ Exemples structure + config créés")
    print(f"✅ Formats multiples sauvegardés")
    print(f"✅ Prêt pour entraînement méta-modèle")
    print(f"\n🚀 Prochaine étape: python metamodel.py")