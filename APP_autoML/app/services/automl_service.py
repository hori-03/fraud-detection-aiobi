"""
Service AutoML - Intégration avec le système AutoML existant

Ce service fait le pont entre l'application Flask et le système AutoML
existant (automl_transformer/full_automl.py).
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

# Ajouter le répertoire racine du projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import du système AutoML existant
from automl_transformer.full_automl import FullAutoML
from automl_transformer.apply_automl_production import AutoMLProductionApplicator
from utils.column_matcher import ColumnMatcher


class AutoMLService:
    """
    Service pour exécuter le système AutoML sur les datasets uploadés
    
    Workflow simple:
    1. Utilisateur upload un dataset
    2. full_automl.fit() entraîne automatiquement un modèle XGBoost sur mesure
    3. Résultats affichés + modèle sauvegardé dans l'historique
    """
    
    def __init__(self, app_config):
        """
        Initialiser le service AutoML
        
        Args:
            app_config: Configuration Flask (pour les chemins)
        """
        self.app_config = app_config
        self.uploads_dir = app_config['UPLOAD_FOLDER']
        self.models_dir = app_config['MODELS_FOLDER']
        self.column_matcher = ColumnMatcher(fuzzy_threshold=0.7)  # Matching sémantique
    
    def analyze_dataset(self, dataset_path):
        """
        Analyser un dataset uploadé avec détection ROBUSTE du target
        
        Utilise la même méthode que full_automl.py:
        - Approche hybride: 70% statistiques + 30% noms de colonnes
        - Privilégie les colonnes binaires avec class imbalance
        - Détecte les patterns de fraude typiques (0.1%-5% minorité)
        
        Args:
            dataset_path: Chemin vers le fichier CSV/JSON
        
        Returns:
            dict: Informations sur le dataset
        """
        try:
            # Charger le dataset
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                df = pd.read_json(dataset_path)
            else:
                raise ValueError("Format non supporté. Utilisez CSV ou JSON.")
            
            # Détection automatique ULTRA-ROBUSTE du target (méthode full_automl)
            target_col = None
            possible_targets = []
            
            for col in df.columns:
                col_lower = col.lower()
                n_unique = df[col].nunique()
                
                # SCORE STATISTIQUE (jusqu'à 1000 points) - CRITÈRE PRINCIPAL
                stat_score = 0
                
                # 1. Nombre de valeurs uniques
                if n_unique == 2:
                    stat_score += 300  # Binaire = excellent
                elif 3 <= n_unique <= 5:
                    stat_score += 200
                elif 6 <= n_unique <= 10:
                    stat_score += 100
                elif 11 <= n_unique <= 20:
                    stat_score += 50
                else:
                    continue  # Trop de valeurs
                
                # 2. Analyse du déséquilibre (class imbalance)
                if n_unique <= 20:
                    try:
                        class_distribution = df[col].value_counts(normalize=True)
                        min_class_ratio = class_distribution.min()
                        
                        # Fraud typique: 0.1%-5% minorité
                        if 0.001 <= min_class_ratio <= 0.05:
                            stat_score += 400  # JACKPOT!
                        elif 0.05 < min_class_ratio <= 0.15:
                            stat_score += 250
                        elif 0.15 < min_class_ratio <= 0.30:
                            stat_score += 100
                        elif 0.30 < min_class_ratio <= 0.45:
                            stat_score += 20
                    except:
                        pass
                
                # 3. Position de la colonne (dernière = probable target)
                col_position = list(df.columns).index(col)
                total_cols = len(df.columns)
                
                if col_position == total_cols - 1:
                    stat_score += 150
                elif col_position >= total_cols - 3:
                    stat_score += 75
                elif col_position >= total_cols - 5:
                    stat_score += 30
                
                # 4. Type de données
                if df[col].dtype in ['int64', 'int32', 'bool']:
                    stat_score += 50
                elif df[col].dtype == 'object':
                    unique_vals = df[col].unique()
                    str_vals = [str(v).lower() for v in unique_vals]
                    if any(val in str_vals for val in ['0', '1', 'yes', 'no', 'true', 'false']):
                        stat_score += 100
                
                # SCORE NOMINAL (jusqu'à 300 points) - CRITÈRE SECONDAIRE
                name_score = 0
                fraud_keywords = {
                    'fraud': 60, 'manipul': 60, 'suspic': 50, 'anomal': 45,
                    'irregul': 40, 'flag': 35, 'indicator': 30, 'detected': 25,
                    'alert': 20, 'risk': 15, 'label': 10, 'target': 8, 'class': 10
                }
                
                for kw, kw_score in fraud_keywords.items():
                    if kw in col_lower:
                        if col_lower.startswith(kw) or col_lower.startswith(f'is_{kw}'):
                            name_score += kw_score * 2
                        elif f'_{kw}' in col_lower or col_lower.endswith(f'_{kw}'):
                            name_score += kw_score * 1.5
                        else:
                            name_score += kw_score
                
                # BONUS SÉMANTIQUE (ColumnMatcher)
                # Vérifier la similarité sémantique avec des termes target connus
                semantic_score = 0
                target_reference_terms = ['fraud', 'is_fraud', 'fraud_flag', 'label', 'target', 
                                         'suspicious', 'anomaly', 'class', 'y']
                
                for ref_term in target_reference_terms:
                    similarity = self.column_matcher.fuzzy_similarity(col, ref_term)
                    if similarity >= 0.7:  # Haute similarité
                        semantic_score += 50 * similarity  # Max +50 par terme
                
                # Ajouter le bonus sémantique au score nominal
                name_score += int(semantic_score)
                
                # Pénalité pour mots suspects
                penalty_words = ['country', 'region', 'zone', 'location', 'city', 'type', 'category']
                for penalty_word in penalty_words:
                    if penalty_word in col_lower:
                        name_score = int(name_score * 0.5)
                        break
                
                # SCORE FINAL = 70% stats + 30% noms
                final_score = int(stat_score * 0.7 + name_score * 0.3)
                
                if final_score > 0:
                    possible_targets.append((col, final_score, n_unique, stat_score, name_score))
            
            # Trier par score
            possible_targets.sort(key=lambda x: (-x[1], x[2]))
            
            # Déterminer si étiqueté
            is_labeled = len(possible_targets) > 0 and possible_targets[0][1] >= 100  # Seuil minimum
            
            if is_labeled:
                target_col = possible_targets[0][0]
                
                # Calculer taux de fraude
                class_dist = df[target_col].value_counts(normalize=True)
                minority_class = class_dist.min()
                fraud_rate = minority_class * 100  # En %
            else:
                fraud_rate = None
            
            return {
                'success': True,
                'is_labeled': bool(is_labeled),
                'dataset_size': int(len(df)),
                'num_features': int(len(df.columns) - (1 if target_col else 0)),
                'columns': df.columns.tolist(),
                'target_column': str(target_col) if target_col else None,
                'fraud_rate': float(round(fraud_rate, 2)) if fraud_rate else None,
                'has_missing': bool(df.isnull().any().any()),
                'missing_count': int(df.isnull().sum().sum()),
                'target_candidates': [
                    {
                        'column': str(col),
                        'score': int(score),
                        'unique_values': int(n_unique)
                    }
                    for col, score, n_unique, _, _ in possible_targets[:3]
                ]
            }
        
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def train_model(self, dataset_path, target_col=None, model_name=None, use_meta_transformer=True, is_labeled=True):
        """
        Entraîner un modèle AutoML sur le dataset uploadé (étiqueté)
        
        Le système full_automl.py:
        - Détecte automatiquement la structure du dataset
        - Utilise le Meta-Transformer pour prédire les meilleurs hyperparamètres
        - Génère automatiquement les features pertinentes
        - Entraîne un modèle XGBoost optimisé sur mesure
        
        Args:
            dataset_path: Chemin vers le dataset uploadé
            target_col: Colonne target (auto-détection si None)
            model_name: Nom du modèle fourni par l'utilisateur (optionnel)
            use_meta_transformer: Utiliser le Meta-Transformer (True) ou règles (False)
            is_labeled: Dataset étiqueté (True) ou non (False)
        
        Returns:
            dict: Résultats de l'entraînement
        """
        if not is_labeled:
            return {
                'success': False,
                'error': 'Dataset non étiqueté. Utilisez predict_unlabeled() à la place.'
            }
        
        start_time = time.time()
        
        try:
            print(f"\n🚀 Démarrage de l'AutoML sur {Path(dataset_path).name}")
            print(f"📊 Meta-Transformer: {'Activé' if use_meta_transformer else 'Désactivé'}")
            
            # Créer l'instance AutoML (avec le Meta-Transformer pré-entraîné)
            automl = FullAutoML(use_meta_transformer=use_meta_transformer)
            
            # Entraîner le modèle (automatique: détection colonnes, feature engineering, hyperparams)
            # fit() retourne directement self.performance
            performance = automl.fit(dataset_path, target_col=target_col)
            
            # Temps d'exécution
            training_time = automl.training_time
            
            # Lire le dataset pour obtenir les infos
            df = pd.read_csv(dataset_path)
            target = automl.target_col
            fraud_rate = None
            if target and target in df.columns:
                fraud_rate = float(df[target].mean() * 100) if df[target].dtype in ['int64', 'bool'] else None
            
            # Calculer l'accuracy à partir de la confusion matrix
            accuracy = None
            confusion_matrix = performance.get('confusion_matrix')
            if confusion_matrix:
                # confusion_matrix = [[TN, FP], [FN, TP]]
                tn, fp = confusion_matrix[0]
                fn, tp = confusion_matrix[1]
                total = tn + fp + fn + tp
                accuracy = float((tn + tp) / total) if total > 0 else None
            
            # Récupérer les features créées par le feature engineer
            features_engineered = []
            if automl.feature_engineer and hasattr(automl.feature_engineer, 'feature_names_generated'):
                features_engineered = automl.feature_engineer.feature_names_generated
            
            # Récupérer les engineering flags et compteurs
            engineering_flags = performance.get('engineering_flags')
            features_count = performance.get('features_engineered_count', {})
            
            # Formater les engineering flags en format lisible
            engineering_info = {}
            if engineering_flags:
                flag_names = ['polynomial', 'interaction', 'binning', 'log_transform', 'aggregation']
                thresholds = {'polynomial': 0.70, 'interaction': 0.60, 'binning': 0.40, 
                             'log_transform': 0.50, 'aggregation': 0.60}
                
                for i, name in enumerate(flag_names):
                    score = float(engineering_flags[i])
                    threshold = thresholds[name]
                    engineering_info[name] = {
                        'score': score,
                        'threshold': threshold,
                        'activated': score > threshold
                    }
            
            # Ajouter les compteurs de features
            engineering_info['counts'] = {
                'ai_predicted': features_count.get('ai_predicted', 0),
                'always_on': features_count.get('always_on', 0),
                'total': features_count.get('total', 0)
            }
            
            # Nom du modèle
            dataset_name = Path(dataset_path).stem
            
            # Utiliser le model_name fourni par l'utilisateur, sinon générer un nom unique
            if not model_name:
                model_name = f"automl_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                # Ajouter timestamp pour garantir l'unicité
                model_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Sauvegarder le modèle dans APP_autoML/models/xgboost_models/
            model_save_path = self.models_dir / f"{model_name}"
            automl.save_model(str(model_save_path))
            
            return {
                'success': True,
                'model_name': model_name,
                'model_path': str(model_save_path),
                'dataset_name': dataset_name,
                'dataset_size': int(len(df)),
                'num_features': int(performance.get('n_features', df.shape[1] - 1)),
                'fraud_rate': fraud_rate,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': float(performance.get('precision', 0)),
                    'recall': float(performance.get('recall', 0)),
                    'f1_score': float(performance.get('test_f1', 0)),
                    'roc_auc': float(performance.get('test_auc', 0))
                },
                'hyperparameters': json.dumps(performance.get('hyperparameters', {})),
                'features_engineered': json.dumps(engineering_info),  # Résumé structuré des features
                'training_time_seconds': float(training_time),
                'meta_transformer_used': use_meta_transformer
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'training_time_seconds': time.time() - start_time
            }
    
    def predict_unlabeled(self, dataset_path, top_k=3, threshold=0.20):
        """
        Faire des prédictions sur un dataset NON ÉTIQUETÉ (production)
        
        Mode COMPLET activé par défaut (recommandé pour production):
        - Ensemble: Combine top-3 modèles similaires (plus robuste)
        - Anomaly Detection: Détecte patterns nouveaux jamais vus
        - Calibration: Ajuste probabilités pour plus de confiance
        
        Pipeline:
        1. CHARGEMENT du dataset
        2. AUTO-MATCH: Trouve les 3 meilleurs modèles similaires
        3. ENSEMBLE: Applique les 3 modèles + moyenne pondérée
        4. ANOMALY DETECTION: Isolation Forest (70% XGBoost + 30% Anomaly)
        5. CALIBRATION: Ajuste les probabilités extrêmes
        
        Args:
            dataset_path: Chemin vers le dataset de production (sans labels)
            top_k: Nombre de modèles à combiner (défaut: 3)
            threshold: Seuil de classification (défaut: 0.20 optimisé fraude)
        
        Returns:
            dict: Résultats des prédictions avec métriques complètes
        """
        start_time = time.time()
        
        try:
            print(f"\n� MODE COMPLET: Prédiction sur {Path(dataset_path).name}")
            print(f"📊 Ensemble (top-{top_k}) + Anomaly Detection + Calibration")
            
            # Charger le dataset
            df = pd.read_csv(dataset_path)
            
            # Créer l'applicateur AutoML
            applicator = AutoMLProductionApplicator()
            
            # ÉTAPE 1: ENSEMBLE PREDICTIONS (top-k modèles)
            print(f"\n🔄 ÉTAPE 1: Ensemble predictions (top-{top_k} modèles)")
            results = applicator.apply_ensemble_predictions(
                df=df,
                top_k=top_k,
                threshold=threshold
            )
            
            # ÉTAPE 2: ANOMALY DETECTION
            print(f"\n🔍 ÉTAPE 2: Anomaly detection (Isolation Forest)")
            results = applicator.add_anomaly_detection(
                df=df,
                results=results,
                contamination=0.1,  # 10% de contamination attendue
                weight_xgboost=0.7,  # 70% XGBoost
                weight_anomaly=0.3   # 30% Anomaly
            )
            
            # ÉTAPE 3: CALIBRATION
            print(f"\n⚖️ ÉTAPE 3: Calibration des probabilités")
            results = applicator.calibrate_probabilities(
                results=results,
                method='isotonic'  # Calibration isotonique (meilleure pour XGBoost)
            )
            
            prediction_time = time.time() - start_time
            
            # Extraire les statistiques
            dataset_name = Path(dataset_path).stem
            total_transactions = len(results)
            
            # Statistiques par niveau de risque
            high_risk = (results['fraud_probability_calibrated'] >= 0.7).sum()
            medium_risk = ((results['fraud_probability_calibrated'] >= 0.5) & 
                          (results['fraud_probability_calibrated'] < 0.7)).sum()
            low_risk = (results['fraud_probability_calibrated'] < 0.5).sum()
            
            # Fraudes détectées (seuil)
            frauds_detected = (results['fraud_prediction'] == 1).sum()
            fraud_rate = (frauds_detected / total_transactions) * 100 if total_transactions > 0 else 0
            
            # Anomalies détectées
            anomalies = (results.get('anomaly_score', pd.Series([0]*len(results))) > 0.5).sum()
            
            # Variance moyenne (stabilité des prédictions)
            avg_variance = results.get('prediction_variance', pd.Series([0]*len(results))).mean()
            stability = 1 - avg_variance  # 1 = très stable, 0 = instable
            
            return {
                'success': True,
                'dataset_name': dataset_name,
                'mode': 'complet',  # Ensemble + Anomaly + Calibration
                'total_transactions': int(total_transactions),
                'frauds_detected': int(frauds_detected),
                'fraud_rate': round(fraud_rate, 2),
                'risk_breakdown': {
                    'high_risk': int(high_risk),      # >70%
                    'medium_risk': int(medium_risk),  # 50-70%
                    'low_risk': int(low_risk)         # <50%
                },
                'anomalies_detected': int(anomalies),
                'prediction_stability': round(float(stability), 3),  # 0-1 (plus haut = mieux)
                'predictions': results['fraud_prediction'].tolist(),
                'probabilities': results['fraud_probability'].tolist(),
                'probabilities_calibrated': results['fraud_probability_calibrated'].tolist(),
                'anomaly_scores': results.get('anomaly_score', pd.Series([0]*len(results))).tolist(),
                'prediction_time_seconds': round(prediction_time, 2)
            }
        
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'prediction_time_seconds': time.time() - start_time
            }
    
    def load_user_model(self, model_path):
        """
        Charger un modèle précédemment entraîné par l'utilisateur
        (pour réutilisation ou prédictions futures)
        
        Args:
            model_path: Chemin vers le dossier du modèle
        
        Returns:
            FullAutoML instance ou None
        """
        try:
            automl = FullAutoML()
            success = automl.load_model(model_path)
            return automl if success else None
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            return None
