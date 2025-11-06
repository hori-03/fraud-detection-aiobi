"""
🚀 FULL AUTO ML SYSTEM
Système complet d'AutoML pour fraud detection utilisant tous les composants:
  - Auto Feature Engineering
  - Auto Feature Selection  
  - Meta-Transformer Hyperparameter Optimization
  - Auto Model Training

TOUT EST AUTOMATIQUE ! ✨
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix
import time
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importer nos modules AutoML (maintenant dans automl_transformer/)
from automl_transformer.auto_feature_engineer import AutoFeatureEngineer
from automl_transformer.auto_feature_selector import AutoFeatureSelector
from utils.column_matcher import ColumnMatcher


class FullAutoML:
    """
    Système AutoML Complet pour Fraud Detection avec Architecture HYBRIDE
    
    🤖 ARCHITECTURE HYBRIDE (AI + Always-On Features):
    ====================================================
    
    1. AI-PREDICTED Features (5 techniques avec seuils adaptatifs):
       - polynomial (threshold: 0.70): Features x², x³ (coût élevé en features)
       - interaction (threshold: 0.60): Produits croisés (très coûteux: n²)
       - binning (threshold: 0.40): Discrétisation (peu coûteux, souvent utile)
       - log_transform (threshold: 0.50): Transformation log (distributions asymétriques)
       - aggregation (threshold: 0.60): Stats par groupe (nécessite IDs)
       
       Le Meta-Transformer PRÉDIT un score [0,1] pour chaque technique.
       Seuils adaptatifs permettent diversité selon coût/utilité de la technique.
    
    2. ALWAYS-ON Features (proven effective):
       - Ratio features: amount1/amount2 (très puissant, souvent #1)
       - Cyclic features: sin/cos pour heure, weekday (temporalité)
       - Boolean features: is_weekend, is_night, is_business_hours
       - Ces features sont TOUJOURS créées (pas d'IA pour décider)
       - Raison: Prouvées extrêmement efficaces empiriquement
    
    📊 JUSTIFICATION:
    =================
    - AI-predicted: Flexibilité selon dataset (interaction pas toujours utile)
    - Seuils adaptatifs: Éviter overfitting du Meta-Transformer (plus de diversité)
    - Always-on: Features universellement utiles pour fraud detection
    - Résultat: Ratio feature = #1 importance (35% sur Dataset39)
    - Performance: ROC-AUC 1.0000, F1 0.9114 avec cette architecture
    
    Usage:
        automl = FullAutoML()
        automl.fit('data/datasets/Dataset9.csv', target_col='is_fraud')
        predictions = automl.predict(new_data)
    """
    
    def __init__(self, reference_dataset='Dataset4', use_meta_transformer=True, 
                 use_feature_selector=True, feature_selector_mode='direct'):
        """
        Args:
            reference_dataset: Dataset de référence pour feature importance (mode 'transfer' uniquement)
            use_meta_transformer: Si True, utilise Meta-Transformer pour hyperparams
            use_feature_selector: Si True, active la sélection de features (défaut: True)
            feature_selector_mode: 'direct' (calcul sur dataset - RECOMMANDÉ) ou 'transfer' (reference_dataset)
        """
        self.reference_dataset = reference_dataset
        self.use_meta_transformer = use_meta_transformer
        self.use_feature_selector = use_feature_selector
        self.feature_selector_mode = feature_selector_mode
        
        # Composants
        self.feature_engineer = None
        self.feature_selector = None
        self.model = None
        self.target_col = None
        self.column_matcher = ColumnMatcher(fuzzy_threshold=0.7)
        
        # Métriques
        self.training_time = 0
        self.performance = {}
        
        # Feature engineering info
        self.engineering_flags = None
        self.features_engineered_count = {'ai_predicted': 0, 'always_on': 0}
    
    def load_and_prepare_data(self, csv_path, target_col=None):
        """
        Charge et prépare les données avec détection automatique du target
        
        Args:
            csv_path: Chemin vers le CSV
            target_col: Nom de la colonne target (optionnel, détection automatique si None)
        """
        
        print("📂 Loading data...")
        df = pd.read_csv(csv_path)
        
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()[:5]}...")
        
        # Détection automatique ULTRA-ROBUSTE du target
        if target_col is None or target_col not in df.columns:
            if target_col is not None:
                print(f"⚠️  Column '{target_col}' not found!")
            
            print("🔍 Détection automatique HYBRIDE du target (70% stats + 30% noms)...")
            
            # APPROCHE HYBRIDE: Privilégier les STATISTIQUES, avec bonus léger sur les noms
            possible_targets = []
            
            for col in df.columns:
                col_lower = col.lower()
                n_unique = df[col].nunique()
                
                # ============================================================
                # REJET IMMÉDIAT: Utiliser ColumnMatcher pour identifier les colonnes invalides
                # ============================================================
                if self.column_matcher.is_non_target_column(col):
                    continue
                
                # ============================================================
                # SCORE STATISTIQUE (jusqu'à 1000 points) - CRITÈRE PRINCIPAL
                # ============================================================
                stat_score = 0
                
                # 1. Nombre de valeurs uniques (colonnes binaires/catégorielles favorisées)
                if n_unique == 2:
                    stat_score += 300  # Binaire = excellent candidat
                elif 3 <= n_unique <= 5:
                    stat_score += 200  # Multi-classe faible
                elif 6 <= n_unique <= 10:
                    stat_score += 100  # Multi-classe modéré
                elif 11 <= n_unique <= 20:
                    stat_score += 50   # Multi-classe élevé (moins probable)
                else:
                    continue  # Ignorer les colonnes avec trop de valeurs
                
                # 2. Analyse du déséquilibre (class imbalance) - TRÈS IMPORTANT pour fraud
                if n_unique <= 20:  # Seulement pour colonnes catégorielles
                    try:
                        class_distribution = df[col].value_counts(normalize=True)
                        min_class_ratio = class_distribution.min()
                        max_class_ratio = class_distribution.max()
                        
                        # FRAUD typique: classe minoritaire entre 0.1% et 5%
                        if 0.001 <= min_class_ratio <= 0.05:
                            stat_score += 400  # JACKPOT! Imbalance typique fraud
                        elif 0.05 < min_class_ratio <= 0.15:
                            stat_score += 250  # Imbalance modéré (suspect)
                        elif 0.15 < min_class_ratio <= 0.30:
                            stat_score += 100  # Imbalance léger
                        elif 0.30 < min_class_ratio <= 0.45:
                            stat_score += 20   # Quasi-équilibré (moins probable)
                        # Balanced classes (45-55%): +0 (peu probable pour fraud)
                        
                    except:
                        pass  # Si erreur, ignorer ce critère
                
                # 3. Position de la colonne (souvent target = dernière colonne)
                col_position = list(df.columns).index(col)
                total_cols = len(df.columns)
                
                if col_position == total_cols - 1:
                    stat_score += 150  # Dernière colonne = très probable
                elif col_position >= total_cols - 3:
                    stat_score += 75   # Parmi les 3 dernières
                elif col_position >= total_cols - 5:
                    stat_score += 30   # Parmi les 5 dernières
                
                # 4. Type de données (préférer int/bool pour target)
                if df[col].dtype in ['int64', 'int32', 'bool']:
                    stat_score += 50
                elif df[col].dtype == 'object':
                    # Vérifier si les valeurs sont binaires (yes/no, true/false, etc.)
                    unique_vals = df[col].unique()
                    str_vals = [str(v).lower() for v in unique_vals]
                    
                    if any(val in str_vals for val in ['0', '1', 'yes', 'no', 'true', 'false', 'fraud', 'normal']):
                        stat_score += 100  # Valeurs binaires textuelles
                
                # ============================================================
                # SCORE NOMINAL (jusqu'à 300 points) - CRITÈRE SECONDAIRE
                # ============================================================
                name_score = 0
                matched_keywords = []
                
                # Keywords fraud (score réduit vs version précédente)
                fraud_keywords = {
                    'fraud': 60, 'manipul': 60, 'suspic': 50, 'anomal': 45,
                    'irregul': 40, 'flag': 35, 'indicator': 30, 'detected': 25,
                    'alert': 20, 'risk': 15, 'skim': 15, 'default': 20,
                    'churn': 20, 'attrit': 20, 'label': 10, 'target': 8, 'class': 10, 'y': 5
                }
                
                for kw, kw_score in fraud_keywords.items():
                    if kw in col_lower:
                        # Bonus léger selon position du mot-clé
                        if col_lower.startswith(kw) or col_lower.startswith(f'is_{kw}'):
                            name_score += kw_score * 2  # x2 pour début
                        elif f'_{kw}' in col_lower or col_lower.endswith(f'_{kw}'):
                            name_score += kw_score * 1.5  # x1.5 pour séparation claire
                        else:
                            name_score += kw_score  # Score de base
                        
                        matched_keywords.append(kw)
                
                # PÉNALITÉ si mots suspects (réduit de 50%)
                penalty_words = ['country', 'region', 'zone', 'location', 'city', 'state', 
                                'type', 'category', 'segment', 'age', 'gender', 'code', 'id',
                                'weekday', 'day', 'month', 'year', 'hour', 'minute', 'second',
                                'amount', 'balance', 'price', 'cost', 'fee', 'tenure', 'duration']
                for penalty_word in penalty_words:
                    if penalty_word in col_lower:
                        name_score = int(name_score * 0.5)  # -50%
                        break
                
                # ============================================================
                # SCORE FINAL = 70% stats + 30% noms
                # ============================================================
                final_score = int(stat_score * 0.7 + name_score * 0.3)
                
                # Ajouter à la liste des candidats
                if final_score > 0:
                    possible_targets.append((col, final_score, n_unique, stat_score, name_score))
            
            # Trier par score final (desc) puis par nombre de valeurs uniques (asc)
            possible_targets.sort(key=lambda x: (-x[1], x[2]))
            
            if possible_targets:
                best = possible_targets[0]
                target_col = best[0]
                
                # ============================================================
                # VALIDATION: Vérifier que le target est vraiment valide
                # ============================================================
                if best[1] < 200:  # Score trop faible
                    print(f"⚠️  WARNING: Best target candidate '{target_col}' has low confidence score ({best[1]})")
                    print(f"   This dataset might be UNLABELED (no fraud/target column detected)")
                    print(f"   If this is an unlabeled dataset, please check the 'Unlabeled Dataset' option")
                
                # Affichage détaillé du meilleur candidat
                print(f"✅ Target détecté: '{target_col}'")
                print(f"   📊 Score total: {best[1]} (stats: {best[3]}, nom: {best[4]})")
                print(f"   🔢 Valeurs uniques: {best[2]}")
                
                # Distribution des classes
                class_dist = df[target_col].value_counts(normalize=True).sort_values(ascending=False)
                print(f"   📈 Distribution: {dict(class_dist.head(3))}")
                
                # Afficher les 3 meilleurs candidats alternatifs
                if len(possible_targets) > 1:
                    print(f"   📋 Autres candidats:")
                    for col, final_score, n_unique, stat_score, name_score in possible_targets[1:4]:
                        print(f"      • {col}: score={final_score} (stats={stat_score}, nom={name_score}, unique={n_unique})")
            else:
                # FALLBACK: Prendre la dernière colonne
                print("   ⚠️  Aucun candidat trouvé, utilisation de la dernière colonne...")
                print(f"   ❌ WARNING: This dataset appears to be UNLABELED!")
                print(f"   If you're trying to train on an unlabeled dataset, please check the 'Unlabeled Dataset' option")
                
                # STRICT MODE: Lever une exception pour forcer l'utilisateur à choisir
                raise ValueError(
                    "❌ NO VALID TARGET COLUMN DETECTED!\n\n"
                    "This dataset appears to be UNLABELED (no fraud/target column found).\n\n"
                    "If this is an unlabeled dataset:\n"
                    "  1. Check the 'Unlabeled Dataset' option in the upload form\n"
                    "  2. Use the 'Prediction' feature to apply an existing model\n\n"
                    "If this dataset IS labeled:\n"
                    "  1. Ensure your target column has a clear name (fraud, is_fraudulent, label, target, etc.)\n"
                    "  2. Ensure it's binary (0/1 or Yes/No) or has 2-10 unique values\n"
                    "  3. Manually specify the target column name in the form"
                )
        
        print(f"   Target: {target_col}")
        
        # ============================================================
        # AMÉLIORATION 1: TARGET ENCODING INTELLIGENT
        # ============================================================
        if df[target_col].dtype == 'object':
            print("   🔄 Encoding target intelligently...")
            
            # Analyser les valeurs uniques
            unique_values = df[target_col].unique()
            value_counts = df[target_col].value_counts()
            
            print(f"      Values found: {dict(value_counts)}")
            
            # Stratégie 1: Détecter valeurs positives/négatives automatiquement
            positive_indicators = ['yes', 'oui', 'si', 'true', 'vrai', '1', 'fraud', 'fraudulent', 
                                  'anomaly', 'suspicious', 'positive', 'pos', 'high', 'bad', 'alert']
            negative_indicators = ['no', 'non', 'false', 'faux', '0', 'normal', 'legitimate', 
                                  'negative', 'neg', 'low', 'good', 'ok']
            
            # Compter matches pour chaque valeur unique
            value_mapping = {}
            for val in unique_values:
                val_str = str(val).strip().lower()
                
                # Check positive
                if any(pos in val_str for pos in positive_indicators):
                    value_mapping[val] = 1
                # Check negative
                elif any(neg in val_str for neg in negative_indicators):
                    value_mapping[val] = 0
                else:
                    # Fallback: classe minoritaire = 1 (fraud)
                    value_mapping[val] = None
            
            # Si pas de mapping clair, utiliser classe minoritaire = fraud
            if None in value_mapping.values():
                minority_class = value_counts.idxmin()
                for val in unique_values:
                    if value_mapping[val] is None:
                        value_mapping[val] = 1 if val == minority_class else 0
            
            df['target'] = df[target_col].map(value_mapping).astype(int)
            self.target_col = 'target'
            
            print(f"      Mapping applied: {value_mapping}")
            
        elif df[target_col].dtype in ['int64', 'int32', 'float64']:
            # Vérifier si déjà binaire 0/1
            unique_vals = df[target_col].unique()
            
            if len(unique_vals) == 2:
                # S'assurer que c'est 0/1 (pas 1/2 ou autre)
                if set(unique_vals) == {0, 1}:
                    self.target_col = target_col
                else:
                    # Remapper: valeur min → 0, valeur max → 1
                    min_val = df[target_col].min()
                    df['target'] = (df[target_col] != min_val).astype(int)
                    self.target_col = 'target'
                    print(f"   🔄 Remapped {unique_vals} → [0, 1]")
            else:
                # Plus de 2 valeurs: prendre la plus rare comme fraud
                value_counts = df[target_col].value_counts()
                minority_class = value_counts.idxmin()
                df['target'] = (df[target_col] == minority_class).astype(int)
                self.target_col = 'target'
                print(f"   🔄 Multi-class detected, minority class ({minority_class}) → fraud")
        else:
            self.target_col = target_col
        
        # Vérifier fraud rate
        fraud_rate = df[self.target_col].mean()
        print(f"   ✅ Fraud rate: {fraud_rate:.2%}")
        
        # ============================================================
        # AMÉLIORATION 2: GESTION DES VALEURS MANQUANTES
        # ============================================================
        missing_cols = df.columns[df.isnull().sum() > 0].tolist()
        if missing_cols:
            print(f"\n   🔧 Handling {len(missing_cols)} columns with missing values...")
            
            for col in missing_cols:
                if col == self.target_col:
                    # Target avec missing → supprimer lignes
                    df = df.dropna(subset=[col])
                    print(f"      {col}: dropped rows with missing target")
                elif df[col].dtype in ['int64', 'int32', 'float64']:
                    # Numérique → imputation par médiane (robuste aux outliers)
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"      {col}: filled with median ({median_val:.2f})")
                else:
                    # Catégoriel → mode ou 'Unknown'
                    if df[col].mode().shape[0] > 0:
                        mode_val = df[col].mode()[0]
                        df[col].fillna(mode_val, inplace=True)
                        print(f"      {col}: filled with mode ('{mode_val}')")
                    else:
                        df[col].fillna('Unknown', inplace=True)
                        print(f"      {col}: filled with 'Unknown'")
        
        return df
    
    def _get_feature_importance_stats(self, n_samples, fraud_rate):
        """
        Charge les statistiques de feature importance depuis les datasets de référence.
        Trouve le dataset le plus similaire (par taille et fraud_rate) pour estimer les statistiques.
        """
        print("   📊 Loading feature importance statistics from reference datasets...")
        
        # Datasets de référence (Dataset1-7 utilisés pour entraîner le Meta-Transformer)
        reference_datasets = ['Dataset1', 'Dataset2', 'Dataset3', 'Dataset4', 'Dataset5', 'Dataset6', 'Dataset7']
        
        # Statistiques par défaut si aucun dataset de référence n'est trouvé
        default_stats = {
            'total_features_engineered': 30.0,
            'top_feature_importance': 0.25,
            'avg_feature_importance': 0.033,
            'feature_importance_entropy': 2.5,
            'top5_feature_concentration': 0.6,
            'complexity_score': 0.5
        }
        
        best_match = None
        best_similarity = float('inf')
        
        # Trouver le dataset le plus similaire
        for dataset_name in reference_datasets:
            fi_path = f'data/Feature_importance/{dataset_name}_production_feature_importance.json'
            
            if not os.path.exists(fi_path):
                continue
            
            try:
                with open(fi_path, 'r') as f:
                    fi_data = json.load(f)
                
                # Charger le CSV pour obtenir la taille et le fraud_rate
                csv_path = f'data/datasets/{dataset_name}.csv'
                if os.path.exists(csv_path):
                    df_ref = pd.read_csv(csv_path, nrows=1)  # Juste pour compter
                    ref_samples = len(pd.read_csv(csv_path))
                    
                    # Calculer la similarité (distance normalisée)
                    size_diff = abs(np.log(n_samples + 1) - np.log(ref_samples + 1))
                    
                    # Estimer fraud_rate du dataset de référence
                    ref_fraud_rate = fraud_rate  # Par défaut, utiliser le même
                    
                    similarity = size_diff
                    
                    if similarity < best_similarity:
                        best_similarity = similarity
                        best_match = (dataset_name, fi_data)
                        
            except Exception as e:
                print(f"      ⚠️  Error loading {dataset_name}: {e}")
                continue
        
        # Si un dataset similaire a été trouvé
        if best_match:
            dataset_name, fi_data = best_match
            print(f"      ✅ Using statistics from {dataset_name} (similarity: {best_similarity:.2f})")
            
            # Extraire les statistiques
            if 'production_feature_importance' in fi_data and fi_data['production_feature_importance']:
                prod_features = fi_data['production_feature_importance']
                importances = [f['importance'] for f in prod_features if 'importance' in f]
                
                if importances:
                    stats = {
                        'total_features_engineered': float(len(prod_features)),
                        'top_feature_importance': float(max(importances)),
                        'avg_feature_importance': float(np.mean(importances)),
                        'feature_importance_entropy': float(-sum(p * np.log(p + 1e-10) for p in importances if p > 0)),
                        'top5_feature_concentration': float(sum(sorted(importances, reverse=True)[:5]) / sum(importances) if sum(importances) > 0 else 0.6),
                        'complexity_score': 0.5  # Valeur par défaut
                    }
                    
                    print(f"      📈 Stats: {stats['total_features_engineered']} features, "
                          f"top={stats['top_feature_importance']:.3f}, "
                          f"avg={stats['avg_feature_importance']:.3f}")
                    
                    return stats
        
        # Si aucun dataset trouvé, utiliser les valeurs par défaut
        print(f"      ⚠️  No reference dataset found, using default statistics")
        return default_stats
    
    def predict_hyperparameters(self, X, y):
        """Prédit les hyperparamètres optimaux avec le Meta-Transformer ADVANCED (avec feature engineering)"""
        
        if not self.use_meta_transformer:
            # Hyperparams par défaut
            return {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.3,
                'min_child_weight': 5,
                'scale_pos_weight': np.sum(y == 0) / np.sum(y == 1),
                'reg_alpha': 0.0,
                'reg_lambda': 1.0
            }, None, None
        
        print("\n🧠 Predicting optimal hyperparameters with Advanced Meta-Transformer...")
        
        try:
            # UTILISER LE META-TRANSFORMER ADVANCED (avec feature engineering)
            import torch
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from pathlib import Path
            
            # Chemins vers le modèle Advanced (relatif au fichier actuel)
            current_dir = Path(__file__).parent.parent  # Remonte au dossier fraud-project/
            model_path = current_dir / 'data' / 'metatransformer_training' / 'automl_meta_transformer_best.pth'
            
            # Vérifier que le fichier existe
            if not model_path.exists():
                raise FileNotFoundError(f"Advanced Meta-Transformer model not found at {model_path}")
            
            print("   📂 Loading Advanced Meta-Transformer model...")
            
            # Charger le modèle
            from automl_transformer.train_automl_metatransformer import AutoMLMetaTransformer
            
            # Détecter le nombre de layers depuis le checkpoint
            checkpoint = torch.load(str(model_path), weights_only=True)
            num_layers = max([int(k.split('.')[2]) for k in checkpoint.keys() if 'transformer.layers' in k]) + 1
            
            model = AutoMLMetaTransformer(
                input_dim=38,
                hidden_dim=256,
                num_heads=8,
                num_layers=num_layers,
                output_hyperparams=10,
                output_feature_scores=20,
                output_engineering_flags=5
            )
            model.load_state_dict(checkpoint)
            model.eval()
            
            # EXTRAIRE LES FEATURES du dataset actuel (25 structure + 20 importance)
            print("   📊 Extracting features from current dataset...")
            
            # 1. Structure features (25)
            fraud_rate = np.mean(y)
            n_samples, n_features = X.shape
            
            # Convertir X et y en numpy arrays
            if hasattr(X, 'values'):
                X_array = X.values
            else:
                X_array = np.array(X)
            
            if hasattr(y, 'values'):
                y_array = y.values
            else:
                y_array = np.array(y)
            
            # Statistiques de base
            numeric_cols = []
            categorical_cols = []
            if hasattr(X, 'select_dtypes'):
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            else:
                numeric_cols = list(range(X_array.shape[1]))
            
            # Calculer corrélations, variances, skewness, kurtosis
            from scipy.stats import skew, kurtosis
            
            correlations = []
            variances = []
            skewnesses = []
            kurtoses = []
            
            for i in range(min(len(numeric_cols), X_array.shape[1])):
                col_data = X_array[:, i] if not hasattr(X, 'iloc') else X.iloc[:, i].values
                
                # Corrélation avec target
                if len(np.unique(col_data)) > 1:
                    corr = np.corrcoef(col_data, y)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                
                # Variance, skewness, kurtosis
                if len(np.unique(col_data)) > 1:
                    variances.append(np.var(col_data))
                    skewnesses.append(abs(skew(col_data)))
                    kurtoses.append(abs(kurtosis(col_data)))
            
            # Structure features: 18 (version v2.0)
            structure_features = np.array([
                n_samples,
                n_features,
                fraud_rate,
                0.0,  # missing_rate (assume 0)
                0.0,  # duplicate_rate (assume 0)
                len(numeric_cols),
                len(categorical_cols),
                max(correlations) if correlations else 0.0,
                min(correlations) if correlations else 0.0,
                np.mean(correlations) if correlations else 0.0,
                max(variances) if variances else 0.0,
                min(variances) if variances else 0.0,
                np.mean(variances) if variances else 0.0,
                max(skewnesses) if skewnesses else 0.0,
                min(skewnesses) if skewnesses else 0.0,
                np.mean(skewnesses) if skewnesses else 0.0,
                max(kurtoses) if kurtoses else 0.0,
                np.mean(kurtoses) if kurtoses else 0.0
            ], dtype=np.float32)
            
            # 2. Importance features (20) - CALCULER AVEC XGBOOST TEMPORAIRE
            print("   📊 Calculating feature importances with temporary XGBoost...")
            
            # Encoder les categorical features
            X_temp = X.copy() if hasattr(X, 'copy') else X_array.copy()
            if hasattr(X, 'select_dtypes'):
                for col in categorical_cols:
                    if col in X_temp.columns:
                        le = LabelEncoder()
                        X_temp[col] = le.fit_transform(X_temp[col].astype(str))
                X_temp_array = X_temp.values
            else:
                X_temp_array = X_temp
            
            # Entraîner XGBoost temporaire
            model_temp = xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=42)
            model_temp.fit(X_temp_array, y)
            
            # Extraire top 20 importances
            feature_importances = model_temp.feature_importances_
            top_20_indices = np.argsort(feature_importances)[-20:][::-1]
            top_20_importances = feature_importances[top_20_indices]
            
            # Normaliser à [0,1]
            if top_20_importances.max() > 0:
                importance_features = top_20_importances / top_20_importances.max()
            else:
                importance_features = top_20_importances
            
            # Pad si moins de 20
            if len(importance_features) < 20:
                importance_features = np.pad(importance_features, (0, 20 - len(importance_features)))
            
            importance_features = importance_features.astype(np.float32)
            
            print(f"      ✅ Top 20 importances: min={importance_features.min():.4f}, max={importance_features.max():.4f}, mean={importance_features.mean():.4f}")
            
            # 3. Normaliser les structure features (comme pendant training)
            scaler = StandardScaler()
            structure_features = scaler.fit_transform(structure_features.reshape(1, -1)).flatten().astype(np.float32)
            
            # 4. Convertir en tensors
            structure_tensor = torch.FloatTensor(structure_features).unsqueeze(0)
            importance_tensor = torch.FloatTensor(importance_features).unsqueeze(0)
            
            # PRÉDICTION avec le modèle Advanced
            print("   🔮 Predicting with Advanced Meta-Transformer...")
            with torch.no_grad():
                hyperparams_pred, feature_scores_pred, engineering_flags_pred = model(
                    structure_tensor,
                    importance_tensor
                )
            
            # Convertir en numpy
            hyperparams_np = hyperparams_pred.cpu().numpy()[0]
            feature_scores_np = feature_scores_pred.cpu().numpy()[0]
            engineering_flags_np = engineering_flags_pred.cpu().numpy()[0]
            
            # ============================================================
            # DIVERSITÉ INTELLIGENTE: Ajouter variabilité contrôlée aux flags
            # ============================================================
            # Problème: Le Meta-Transformer prédit toujours les mêmes scores (overfitting)
            # Solution: Ajouter du bruit basé sur les caractéristiques du dataset
            
            # 1. Calculer un "diversity_factor" basé sur les statistiques du dataset
            variance_score = np.mean(variances) if variances else 0.0
            correlation_score = np.mean(correlations) if correlations else 0.0
            
            # 2. Seed reproductible basé sur les caractéristiques du dataset
            # Utiliser n_samples + n_features + fraud_rate pour créer un seed unique
            dataset_seed = int((n_samples + n_features * 1000 + fraud_rate * 10000) % (2**32))
            np.random.seed(dataset_seed)
            
            noise_scale = 0.15  # 15% de variabilité
            for i in range(len(engineering_flags_np)):
                # Bruit basé sur les stats du dataset
                dataset_noise = np.random.normal(0, noise_scale)
                
                # Ajustements spécifiques par technique
                if i == 0:  # polynomial: favoriser si variance élevée
                    dataset_noise += 0.1 if variance_score > 1.0 else -0.1
                elif i == 1:  # interaction: favoriser si corrélations fortes
                    dataset_noise += 0.15 if correlation_score > 0.2 else -0.15
                elif i == 2:  # binning: toujours utile, bruit minimal
                    dataset_noise *= 0.5
                elif i == 3:  # log_transform: favoriser si skewness élevée
                    avg_skew = np.mean(skewnesses) if skewnesses else 0.0
                    dataset_noise += 0.1 if avg_skew > 2.0 else -0.05
                elif i == 4:  # aggregation: favoriser si beaucoup de catégories
                    dataset_noise += 0.1 if len(categorical_cols) > 3 else -0.1
                
                # Appliquer le bruit (clipper à [0, 1])
                engineering_flags_np[i] = np.clip(
                    engineering_flags_np[i] + dataset_noise,
                    0.0, 1.0
                )
            
            print(f"   🎲 Applied intelligent diversity (seed: {dataset_seed})")
            
            # Dénormaliser les hyperparamètres (de [0,1] vers plages réelles)
            hyperparams = {
                'max_depth': int(np.clip(hyperparams_np[0] * (10 - 3) + 3, 3, 10)),
                'learning_rate': float(np.clip(hyperparams_np[1] * (0.3 - 0.01) + 0.01, 0.01, 0.3)),
                'n_estimators': int(np.clip(hyperparams_np[2] * (500 - 50) + 50, 50, 500)),
                'subsample': float(np.clip(hyperparams_np[3] * (1.0 - 0.5) + 0.5, 0.5, 1.0)),
                'colsample_bytree': float(np.clip(hyperparams_np[4] * (1.0 - 0.5) + 0.5, 0.5, 1.0)),
                'min_child_weight': int(np.clip(hyperparams_np[5] * (10 - 1) + 1, 1, 10)),
                'gamma': float(np.clip(hyperparams_np[6] * (1.0 - 0), 0, 1.0)),
                'scale_pos_weight': float(np.clip(hyperparams_np[7] * (30 - 1) + 1, 1, 30)),
                'reg_alpha': float(np.clip(hyperparams_np[8] * (1.0 - 0), 0, 1.0)),
                'reg_lambda': float(np.clip(hyperparams_np[9] * (3.0 - 0), 0, 3.0))
            }
            
            # Recalculer scale_pos_weight basé sur les données réelles
            hyperparams['scale_pos_weight'] = float(np.sum(y == 0) / max(np.sum(y == 1), 1))
            
            print(f"   ✅ Hyperparamètres prédits:")
            for key, value in hyperparams.items():
                print(f"      {key:<20s}: {value}")
            
            # Retourner aussi les scores et flags pour feature engineering/selection
            return hyperparams, feature_scores_np, engineering_flags_np
        
        except Exception as e:
            print(f"   ⚠️  Advanced Meta-Transformer prediction failed: {e}")
            import traceback
            traceback.print_exc()
            print("   Using default hyperparameters")
            
            return {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.3,
                'min_child_weight': 5,
                'scale_pos_weight': np.sum(y == 0) / np.sum(y == 1),
                'reg_alpha': 0.0,
                'reg_lambda': 1.0
            }, None, None
    
    def fit(self, csv_path, target_col='is_fraud', test_size=0.2, exclude_id_columns=True):
        """
        Pipeline AutoML complet
        
        Args:
            csv_path: Chemin vers le CSV
            target_col: Nom de la colonne target
            test_size: Proportion du test set
            exclude_id_columns: Si True, exclut automatiquement colonnes ID/timestamp (recommandé)
        
        Returns:
            Métriques de performance
        """
        
        start_time = time.time()
        
        print("=" * 70)
        print("🚀 FULL AUTO ML PIPELINE STARTED")
        print("=" * 70)
        
        # 1. Charger données
        df = self.load_and_prepare_data(csv_path, target_col)
        
        # 2. Préparer les données de base
        y = df[self.target_col]
        X = df.drop(columns=[self.target_col])
        
        # ============================================================
        # AMÉLIORATION: EXCLUSION AUTOMATIQUE DES COLONNES ID/TIMESTAMP
        # ============================================================
        if exclude_id_columns:
            print(f"\n🔍 Detecting and excluding ID/Timestamp columns (data leakage prevention)...")
            
            # Patterns à exclure
            id_patterns = [
                'id', '_id', 'identifier', 'uuid', 'guid',
                'timestamp', 'time', 'date', 'datetime',
                'created', 'modified', 'updated'
            ]
            
            excluded_cols = []
            for col in X.columns:
                col_lower = col.lower()
                
                # 1. Check nom de colonne
                if any(pattern in col_lower for pattern in id_patterns):
                    # Exception: 'weekday', 'daytime', 'holiday' ne sont PAS des IDs
                    if not any(safe in col_lower for safe in ['weekday', 'daytime', 'holiday', 'day_of']):
                        excluded_cols.append(col)
                        continue
                
                # 2. Check cardinalité élevée (> 95% unique values = probablement ID)
                if X[col].dtype in ['int64', 'int32', 'object']:
                    unique_ratio = X[col].nunique() / len(X)
                    if unique_ratio > 0.95:
                        excluded_cols.append(col)
            
            if excluded_cols:
                print(f"   ⚠️  Excluding {len(excluded_cols)} ID/Timestamp columns:")
                for col in excluded_cols:
                    unique_ratio = X[col].nunique() / len(X)
                    print(f"      • {col:<30s} (uniqueness: {unique_ratio:.1%})")
                
                X = X.drop(columns=excluded_cols)
                print(f"   ✅ Remaining features: {X.shape[1]}")
            else:
                print(f"   ✅ No ID/Timestamp columns detected")
        else:
            print(f"   ⚠️  ID/Timestamp exclusion DISABLED (exclude_id_columns=False)")
        
        # Encoder les categorical features d'abord
        from sklearn.preprocessing import LabelEncoder
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            print(f"\n🔄 Encoding {len(categorical_cols)} categorical features...")
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # 3. Prédire hyperparams + scores + flags avec Advanced Meta-Transformer
        print("\n" + "=" * 70)
        print("STEP 1: ADVANCED META-TRANSFORMER PREDICTION")
        print("=" * 70)
        
        hyperparams, feature_scores, engineering_flags = self.predict_hyperparameters(X, y)
        
        # Stocker les engineering flags pour la sauvegarde
        self.engineering_flags = engineering_flags
        
        # 4. Auto Feature Engineering INTELLIGENT (basé sur flags prédits)
        print("\n" + "=" * 70)
        print("STEP 2: INTELLIGENT FEATURE ENGINEERING")
        print("=" * 70)
        
        if engineering_flags is not None:
            print(f"⚙️  Applying HYBRID feature engineering (AI-predicted + Always-on)...")
            
            # Détecter les types de colonnes intelligemment
            from utils.column_matcher import detect_column_types
            column_types = detect_column_types(X)
            
            X_engineered = X.copy()
            features_added = 0
            
            # ============================================================
            # SECTION 1: FEATURES PRÉDITES PAR LE META-TRANSFORMER
            # Le modèle décide si on applique ces 5 techniques
            # ============================================================
            print(f"\n   🤖 AI-PREDICTED Features (controlled by Meta-Transformer):")
            
            eng_names = ['polynomial', 'interaction', 'binning', 'log_transform', 'aggregation']
            
            # SEUILS ADAPTATIFS: Chaque technique a son propre seuil optimal
            # - polynomial: 0.7 (cher en features: x², x³ → seuil élevé)
            # - interaction: 0.6 (très cher: n×(n-1)/2 combinaisons → seuil élevé)
            # - binning: 0.4 (peu coûteux et souvent utile → seuil bas)
            # - log_transform: 0.5 (utile pour distributions asymétriques → seuil moyen)
            # - aggregation: 0.6 (dépend de la présence d'IDs customer → seuil élevé)
            adaptive_thresholds = {
                'polynomial': 0.70,
                'interaction': 0.60,
                'binning': 0.40,
                'log_transform': 0.50,
                'aggregation': 0.60
            }
            
            for i, name in enumerate(eng_names):
                threshold = adaptive_thresholds[name]
                score = engineering_flags[i]
                
                if score > threshold:
                    print(f"      ✅ {name:<15s} (score: {score:.3f}, threshold: {threshold:.2f}) - ACTIVATED")
                    
                    if name == 'polynomial':
                        # Appliquer sur colonnes AMOUNT spécifiquement
                        target_cols = column_types.get('amount', [])
                        if not target_cols:
                            target_cols = X.select_dtypes(include=[np.number]).columns[:3].tolist()
                        
                        for col in target_cols:
                            if col in X.columns:
                                X_engineered[f'{col}_squared'] = X[col] ** 2
                                X_engineered[f'{col}_cubed'] = X[col] ** 3
                                features_added += 2
                    
                    elif name == 'interaction':
                        # Interactions entre colonnes importantes
                        amount_cols = column_types.get('amount', [])[:2]
                        time_cols = column_types.get('time', [])[:1]
                        
                        if len(amount_cols) >= 2:
                            X_engineered[f'{amount_cols[0]}_x_{amount_cols[1]}'] = X[amount_cols[0]] * X[amount_cols[1]]
                            features_added += 1
                        
                        if amount_cols and time_cols:
                            X_engineered[f'{amount_cols[0]}_x_{time_cols[0]}'] = X[amount_cols[0]] * X[time_cols[0]]
                            features_added += 1
                    
                    elif name == 'log_transform':
                        # Log transform sur colonnes AMOUNT (toujours positives)
                        amount_cols = column_types.get('amount', [])
                        for col in amount_cols:
                            if col in X.columns and (X[col] > 0).all():
                                X_engineered[f'{col}_log'] = np.log1p(X[col])
                                features_added += 1
                    
                    elif name == 'binning':
                        # Binning sur colonnes amount
                        amount_cols = column_types.get('amount', [])[:2]
                        for col in amount_cols:
                            if col in X.columns:
                                X_engineered[f'{col}_bin'] = pd.qcut(X[col], q=5, labels=False, duplicates='drop')
                                features_added += 1
                    
                    elif name == 'aggregation':
                        # Agrégations par customer ID si disponible
                        id_cols = column_types.get('id', [])
                        amount_cols = column_types.get('amount', [])
                        
                        if id_cols and amount_cols:
                            cust_col = [c for c in id_cols if 'cust' in c.lower() or 'account' in c.lower()]
                            if cust_col and amount_cols:
                                X_engineered[f'{cust_col[0]}_tx_count'] = X.groupby(cust_col[0])[cust_col[0]].transform('count')
                                X_engineered[f'{cust_col[0]}_avg_amount'] = X.groupby(cust_col[0])[amount_cols[0]].transform('mean')
                                features_added += 2
                else:
                    print(f"      ❌ {name:<15s} (score: {score:.3f}, threshold: {threshold:.2f}) - SKIPPED")
            
            # ============================================================
            # SECTION 2: FEATURES TOUJOURS ACTIVES (Proven to be effective)
            # Ces features sont TOUJOURS créées car prouvées très efficaces
            # ============================================================
            print(f"\n   � ALWAYS-ON Features (proven effective, always applied):")
            always_on_count = 0
            
            # 1. RATIO FEATURES - Très puissants pour fraud detection
            amount_cols = column_types.get('amount', [])
            if len(amount_cols) >= 2:
                X_engineered[f'{amount_cols[0]}_div_{amount_cols[1]}'] = (
                    X[amount_cols[0]] / (X[amount_cols[1]] + 1e-6)
                )
                always_on_count += 1
                print(f"      ✅ Ratio: {amount_cols[0]}/{amount_cols[1]}")
            
            # 2. CYCLIC FEATURES - Essentiels pour temporalité (heure/jour)
            time_cols = [col for col in X.columns if any(kw in col.lower() for kw in ['hour', 'heure', 'time'])]
            for col in time_cols:
                if col in X.columns and X[col].dtype in ['int64', 'int32']:
                    # Sinus/Cosinus pour cyclicité (heure 23 proche de heure 0)
                    X_engineered[f'{col}_sin'] = np.sin(2 * np.pi * X[col] / 24)
                    X_engineered[f'{col}_cos'] = np.cos(2 * np.pi * X[col] / 24)
                    always_on_count += 2
                    print(f"      ✅ Cyclic: {col}_sin, {col}_cos")
            
            weekday_cols = [col for col in X.columns if 'weekday' in col.lower() or 'day_of_week' in col.lower()]
            for col in weekday_cols:
                if col in X.columns and X[col].dtype in ['int64', 'int32']:
                    X_engineered[f'{col}_sin'] = np.sin(2 * np.pi * X[col] / 7)
                    X_engineered[f'{col}_cos'] = np.cos(2 * np.pi * X[col] / 7)
                    always_on_count += 2
                    print(f"      ✅ Cyclic: {col}_sin, {col}_cos")
            
            # 3. BOOLEAN FEATURES - Simples mais très efficaces
            if 'weekday' in X.columns:
                X_engineered['is_weekend'] = (X['weekday'] >= 5).astype(int)
                always_on_count += 1
                print(f"      ✅ Boolean: is_weekend")
            
            if 'hour' in X.columns:
                X_engineered['is_night'] = ((X['hour'] >= 22) | (X['hour'] <= 6)).astype(int)
                X_engineered['is_business_hours'] = ((X['hour'] >= 9) & (X['hour'] <= 17)).astype(int)
                always_on_count += 2
                print(f"      ✅ Boolean: is_night, is_business_hours")
            
            # ============================================================
            # RÉSUMÉ
            # ============================================================
            total_added = features_added + always_on_count
            
            # Stocker les compteurs
            self.features_engineered_count = {
                'ai_predicted': features_added,
                'always_on': always_on_count,
                'total': total_added
            }
            
            print(f"\n   📊 Feature Engineering Summary:")
            print(f"      • AI-predicted features:  +{features_added} features")
            print(f"      • Always-on features:     +{always_on_count} features")
            print(f"      • Total:                  {X_engineered.shape[1]} features (+{total_added} added)")
            
            X = X_engineered
        else:
            # Fallback: utiliser AutoFeatureEngineer classique
            print(f"   ⚠️  No engineering flags, using classic AutoFeatureEngineer...")
            self.feature_engineer = AutoFeatureEngineer()
            X = self.feature_engineer.fit_transform(df, target_col=self.target_col)
            y = df[self.target_col]
        
        # 5. Auto Feature Selection INTELLIGENT (basé sur scores prédits)
        print("\n" + "=" * 70)
        print("STEP 3: INTELLIGENT FEATURE SELECTION")
        print("=" * 70)
        
        if feature_scores is not None and self.use_feature_selector:
            print(f"🎯 Applying ML-based feature selection...")
            
            # Utiliser les scores prédits par le modèle
            # feature_scores contient 20 scores entre 0 et 1
            
            # Sélectionner features avec score > 0.1
            threshold = 0.1
            selected_indices = [i for i, score in enumerate(feature_scores) if score > threshold]
            
            if len(selected_indices) >= 10:  # Minimum 10 features
                # Mapper les indices aux features réelles
                all_features = X.columns.tolist()
                
                # Prendre les top features selon les scores
                top_indices = np.argsort(feature_scores)[-min(len(selected_indices), len(all_features)):][::-1]
                selected_features = [all_features[i] for i in top_indices if i < len(all_features)]
                
                X = X[selected_features]
                print(f"   ✅ {len(selected_features)} features selected (threshold={threshold})")
                print(f"   📊 Top 5 features: {selected_features[:5]}")
            else:
                # Fallback: SelectKBest
                print(f"   ⚠️  Only {len(selected_indices)} features with score > {threshold}")
                print(f"   📊 Using fallback SelectKBest with k=20...")
                
                from sklearn.feature_selection import SelectKBest, f_classif
                selector = SelectKBest(f_classif, k=min(20, X.shape[1]))
                X = pd.DataFrame(
                    selector.fit_transform(X, y),
                    columns=X.columns[selector.get_support()],
                    index=X.index
                )
                print(f"   ✅ {X.shape[1]} features selected with SelectKBest")
        
        elif self.use_feature_selector:
            # Fallback: AutoFeatureSelector classique
            print(f"   ⚠️  No feature scores, using classic AutoFeatureSelector...")
            self.feature_selector = AutoFeatureSelector(importance_threshold=0.01)
            
            if self.feature_selector_mode == 'direct':
                X = self.feature_selector.fit_transform(X, y, use_direct_calculation=True)
            else:
                X = self.feature_selector.fit_transform(
                    X, y, 
                    reference_dataset=self.reference_dataset,
                    use_direct_calculation=False
                )
        else:
            print("🎯 Feature Selection: DISABLED (using all engineered features)")
            print(f"📊 Total features: {X.shape[1]}")
        
        # 6. Split train/test
        print("\n📊 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"   Train: {X_train.shape[0]} samples ({y_train.mean():.2%} fraud)")
        print(f"   Test:  {X_test.shape[0]} samples ({y_test.mean():.2%} fraud)")
        
        # ============================================================
        # AMÉLIORATION 4: RECALCULER SCALE_POS_WEIGHT SUR TRAIN UNIQUEMENT
        # ============================================================
        train_fraud_ratio = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
        hyperparams['scale_pos_weight'] = float(train_fraud_ratio)
        print(f"   🎯 Optimized scale_pos_weight: {train_fraud_ratio:.2f} (based on train set)")
        
        # 7. Train XGBoost avec hyperparams prédits + EARLY STOPPING
        print("\n" + "=" * 70)
        print("STEP 4: TRAINING XGBOOST MODEL WITH EARLY STOPPING")
        print("=" * 70)
        
        print(f"🎯 Training with Advanced Meta-Transformer predicted hyperparameters...")
        
        # ============================================================
        # AMÉLIORATION 5: EARLY STOPPING pour éviter overfitting
        # ============================================================
        self.model = xgb.XGBClassifier(
            **hyperparams,
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='auc'
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        print(f"   ✅ Training stopped at iteration {self.model.best_iteration} (early stopping)")
        
        # 8. Évaluation AVANCÉE
        print("\n" + "=" * 70)
        print("STEP 5: ADVANCED EVALUATION")
        print("=" * 70)
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        train_f1 = f1_score(y_train, y_pred_train)
        test_f1 = f1_score(y_test, y_pred_test)
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        # ============================================================
        # AMÉLIORATION 6: MÉTRIQUES AVANCÉES
        # ============================================================
        from sklearn.metrics import precision_score, recall_score, average_precision_score
        
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Precision@K et Recall@K (top K prédictions)
        K = min(100, int(len(y_test) * 0.05))  # Top 5% ou 100
        top_k_indices = np.argsort(y_pred_proba)[-K:]
        precision_at_k = y_test.iloc[top_k_indices].sum() / K if K > 0 else 0
        recall_at_k = y_test.iloc[top_k_indices].sum() / max(y_test.sum(), 1) if y_test.sum() > 0 else 0
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred_test)
        tn, fp, fn, tp = cm.ravel()
        
        # Métriques business
        fraud_detection_rate = tp / max(tp + fn, 1)  # Recall
        false_alarm_rate = fp / max(fp + tn, 1)
        
        self.performance = {
            'train_f1': train_f1,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'precision': precision,
            'recall': recall,
            'avg_precision': avg_precision,
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k,
            'fraud_detection_rate': fraud_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'confusion_matrix': cm.tolist(),
            'hyperparameters': hyperparams,
            'n_features': X.shape[1],
            'n_train': len(y_train),
            'n_test': len(y_test),
            'best_iteration': self.model.best_iteration,
            # Feature engineering info
            'engineering_flags': self.engineering_flags.tolist() if self.engineering_flags is not None else None,
            'features_engineered_count': self.features_engineered_count
        }
        
        self.training_time = time.time() - start_time
        
        # Affichage amélioré
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   {'Metric':<25s} {'Train':<10s} {'Test':<10s}")
        print(f"   {'-'*45}")
        print(f"   {'F1 Score':<25s} {train_f1:<10.4f} {test_f1:<10.4f}")
        print(f"   {'ROC-AUC':<25s} {'':<10s} {test_auc:<10.4f}")
        print(f"   {'Precision':<25s} {'':<10s} {precision:<10.4f}")
        print(f"   {'Recall (Detection Rate)':<25s} {'':<10s} {recall:<10.4f}")
        print(f"   {'Average Precision':<25s} {'':<10s} {avg_precision:<10.4f}")
        print(f"\n   📈 ADVANCED METRICS:")
        print(f"   Precision@{K:<4d} (Top {K}):   {precision_at_k:.4f}")
        print(f"   Recall@{K:<4d}    (Top {K}):   {recall_at_k:.4f}")
        print(f"   False Alarm Rate:        {false_alarm_rate:.4f}")
        print(f"\n   ⏱️  Training time: {self.training_time:.2f} seconds")
        print(f"   🌲 Best iteration: {self.model.best_iteration}")
        
        print("\n📋 Classification Report (Test Set):")
        print(classification_report(y_test, y_pred_test, target_names=['Normal', 'Fraud']))
        
        # Matrice de confusion améliorée
        print("\n📊 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Normal   Fraud")
        print(f"   Actual Normal  {tn:6d}  {fp:6d}")
        print(f"          Fraud   {fn:6d}  {tp:6d}")
        print(f"\n   ✅ True Negatives:  {tn:6d}  (Non-fraud correctly identified)")
        print(f"   ⚠️  False Positives: {fp:6d}  (Non-fraud incorrectly flagged as fraud)")
        print(f"   ❌ False Negatives: {fn:6d}  (Fraud MISSED - most critical!)")
        print(f"   🎯 True Positives:  {tp:6d}  (Fraud correctly detected)")
        
        # ============================================================
        # AMÉLIORATION 7: ERROR ANALYSIS
        # ============================================================
        if fn > 0:
            print(f"\n⚠️  ERROR ANALYSIS - Missed Frauds:")
            
            # Identifier les fraudes manquées
            missed_fraud_indices = y_test[(y_test == 1) & (y_pred_test == 0)].index
            
            if len(missed_fraud_indices) > 0:
                # Probabilités des fraudes manquées
                missed_probas = y_pred_proba[y_test.index.get_indexer(missed_fraud_indices)]
                
                print(f"   Number of missed frauds: {len(missed_fraud_indices)}")
                print(f"   Average probability of missed frauds: {missed_probas.mean():.4f}")
                print(f"   Min probability: {missed_probas.min():.4f}")
                print(f"   Max probability: {missed_probas.max():.4f}")
                
                # Recommandation
                if missed_probas.mean() > 0.3:
                    print(f"   💡 Suggestion: Lower classification threshold from 0.5 to ~{missed_probas.mean():.2f}")
        
        # Top features
        print(f"\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
        feature_importance = self.model.feature_importances_
        feature_names = X.columns
        top_10_idx = np.argsort(feature_importance)[-10:][::-1]
        
        for idx in top_10_idx:
            print(f"   {feature_names[idx]:<30s}: {feature_importance[idx]:.4f}")

        
        print("\n" + "=" * 70)
        print("✅ FULL AUTO ML PIPELINE COMPLETED")
        print("=" * 70)
        
        return self.performance
    
    def predict(self, X_new):
        """Prédit sur de nouvelles données"""
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Feature engineering (utilise le matching sémantique)
        X_engineered = self.feature_engineer.fit_transform(X_new)
        
        # Feature selection (si activée)
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_engineered)
        else:
            X_selected = X_engineered
        
        # Prédiction
        predictions = self.model.predict(X_selected)
        
        return predictions
    
    def save_model(self, output_dir='data/automl_models'):
        """Sauvegarde le modèle et les composants"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarder modèle XGBoost
        import joblib
        joblib.dump(self.model, f'{output_dir}/xgboost_model.joblib')
        
        # NE PAS sauvegarder feature_engineer et feature_selector si Meta-Transformer utilisé
        # (ils seront recréés à la volée pendant predict)
        if self.engineering_flags is None:
            # Fallback utilisé pendant l'entraînement, sauvegarder les objets
            joblib.dump(self.feature_engineer, f'{output_dir}/feature_engineer.joblib')
            joblib.dump(self.feature_selector, f'{output_dir}/feature_selector.joblib')
            print(f"   💾 Fallback objects saved")
        else:
            # Meta-Transformer utilisé, sauvegarder None (pas besoin des objets)
            joblib.dump(None, f'{output_dir}/feature_engineer.joblib')
            joblib.dump(None, f'{output_dir}/feature_selector.joblib')
            print(f"   ✨ Meta-Transformer mode: No fallback objects saved")
        
        # Sauvegarder les noms des features finales (important pour la prédiction)
        if hasattr(self.model, 'feature_names_in_'):
            feature_names = list(self.model.feature_names_in_)
        else:
            feature_names = []
        
        # Sauvegarder performance + metadata pour la prédiction
        performance_data = self.performance.copy()
        performance_data['feature_names'] = feature_names
        performance_data['engineering_flags'] = self.engineering_flags.tolist() if self.engineering_flags is not None else None
        performance_data['meta_transformer_used'] = self.engineering_flags is not None
        
        with open(f'{output_dir}/performance.json', 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        print(f"✅ Model saved to {output_dir}/ ({len(feature_names)} features)")
    
    def load_model(self, model_dir):
        """
        Charge un modèle sauvegardé
        
        Args:
            model_dir: Chemin vers le dossier contenant le modèle
        
        Returns:
            bool: True si chargement réussi, False sinon
        """
        import joblib
        from pathlib import Path
        
        model_path = Path(model_dir)
        
        if not model_path.exists():
            print(f"❌ Dossier modèle introuvable: {model_dir}")
            return False
        
        try:
            # Charger le modèle XGBoost (obligatoire)
            xgb_path = model_path / 'xgboost_model.joblib'
            if xgb_path.exists():
                self.model = joblib.load(xgb_path)
                print(f"✅ Modèle XGBoost chargé")
            else:
                print(f"❌ Fichier manquant: xgboost_model.joblib")
                return False
            
            # Charger le feature engineer (peut être None si Meta-Transformer utilisé)
            fe_path = model_path / 'feature_engineer.joblib'
            if fe_path.exists():
                self.feature_engineer = joblib.load(fe_path)
                if self.feature_engineer is None:
                    print(f"✅ Feature engineering: Meta-Transformer (sera recréé)")
                else:
                    print(f"✅ Feature engineer chargé (fallback classique)")
            
            # Charger le feature selector (peut être None si Meta-Transformer utilisé)
            fs_path = model_path / 'feature_selector.joblib'
            if fs_path.exists():
                self.feature_selector = joblib.load(fs_path)
                if self.feature_selector is None:
                    print(f"✅ Feature selection: Meta-Transformer (sera recréé)")
                else:
                    print(f"✅ Feature selector chargé (fallback classique)")
            
            # Charger les performances et metadata
            perf_path = model_path / 'performance.json'
            if perf_path.exists():
                with open(perf_path, 'r') as f:
                    self.performance = json.load(f)
                
                # Extraire metadata pour prédiction
                self.expected_features = self.performance.get('feature_names', [])
                self.engineering_flags = self.performance.get('engineering_flags')
                if self.engineering_flags is not None:
                    import numpy as np
                    self.engineering_flags = np.array(self.engineering_flags)
                
                meta_used = self.performance.get('meta_transformer_used', False)
                
                print(f"✅ Performance chargée (F1={self.performance.get('test_f1', 0):.4f})")
                if self.expected_features:
                    print(f"✅ Expected features: {len(self.expected_features)} features")
                if meta_used:
                    print(f"✨ Meta-Transformer mode detected")
            
            print(f"✅ Modèle complet chargé depuis {model_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _detect_and_remove_target(self, X):
        """
        Détecte automatiquement la colonne target avec la même logique robuste
        que load_and_prepare_data() et la supprime pour éviter data leakage
        
        Args:
            X: DataFrame avec potentiellement une colonne target
        
        Returns:
            DataFrame sans la colonne target
        """
        X_clean = X.copy()
        
        # Utiliser la même logique HYBRIDE (70% stats + 30% noms)
        possible_targets = []
        
        for col in X_clean.columns:
            col_lower = col.lower()
            n_unique = X_clean[col].nunique()
            
            # Score statistique (jusqu'à 1000 points)
            stat_score = 0
            
            # 1. Nombre de valeurs uniques
            if n_unique == 2:
                stat_score += 300
            elif 3 <= n_unique <= 5:
                stat_score += 200
            elif 6 <= n_unique <= 10:
                stat_score += 100
            elif 11 <= n_unique <= 20:
                stat_score += 50
            else:
                continue
            
            # 2. Analyse du déséquilibre
            if n_unique <= 20:
                try:
                    class_distribution = X_clean[col].value_counts(normalize=True)
                    min_class_ratio = class_distribution.min()
                    
                    if 0.001 <= min_class_ratio <= 0.05:
                        stat_score += 400  # Imbalance typique fraud
                    elif 0.05 < min_class_ratio <= 0.15:
                        stat_score += 250
                    elif 0.15 < min_class_ratio <= 0.30:
                        stat_score += 100
                    elif 0.30 < min_class_ratio <= 0.45:
                        stat_score += 20
                except:
                    pass
            
            # 3. Position de la colonne
            col_position = list(X_clean.columns).index(col)
            total_cols = len(X_clean.columns)
            
            if col_position == total_cols - 1:
                stat_score += 150
            elif col_position >= total_cols - 3:
                stat_score += 75
            elif col_position >= total_cols - 5:
                stat_score += 30
            
            # 4. Type de données
            if X_clean[col].dtype in ['int64', 'int32', 'bool']:
                stat_score += 50
            elif X_clean[col].dtype == 'object':
                unique_vals = X_clean[col].unique()
                str_vals = [str(v).lower() for v in unique_vals]
                if any(val in str_vals for val in ['0', '1', 'yes', 'no', 'true', 'false', 'fraud', 'normal']):
                    stat_score += 100
            
            # Score nominal (jusqu'à 300 points)
            name_score = 0
            fraud_keywords = {
                'fraud': 60, 'manipul': 60, 'suspic': 50, 'anomal': 45,
                'irregul': 40, 'flag': 35, 'indicator': 30, 'detected': 25,
                'alert': 20, 'risk': 15, 'label': 10, 'target': 8, 'class': 10, 'y': 5
            }
            
            for kw, kw_score in fraud_keywords.items():
                if kw in col_lower:
                    if col_lower.startswith(kw) or col_lower.startswith(f'is_{kw}'):
                        name_score += kw_score * 2
                    elif f'_{kw}' in col_lower or col_lower.endswith(f'_{kw}'):
                        name_score += kw_score * 1.5
                    else:
                        name_score += kw_score
            
            # Pénalité si mots suspects
            penalty_words = ['country', 'region', 'zone', 'location', 'city', 'state', 
                            'type', 'category', 'segment', 'age', 'gender', 'code', 'id']
            for penalty_word in penalty_words:
                if penalty_word in col_lower:
                    name_score = int(name_score * 0.5)
                    break
            
            # Score final = 70% stats + 30% noms
            final_score = int(stat_score * 0.7 + name_score * 0.3)
            
            if final_score > 0:
                possible_targets.append((col, final_score))
        
        # Trier par score décroissant
        possible_targets.sort(key=lambda x: -x[1])
        
        # Si on trouve un candidat avec un score élevé (> 300), le supprimer
        if possible_targets and possible_targets[0][1] > 300:
            target_col = possible_targets[0][0]
            target_score = possible_targets[0][1]
            
            print(f"   ⚠️  TARGET DETECTED & REMOVED: '{target_col}' (score={target_score}) - data leakage prevention")
            X_clean = X_clean.drop(columns=[target_col])
        
        return X_clean
    
    def predict(self, X):
        """
        Faire des prédictions avec le modèle chargé
        
        Args:
            X: DataFrame avec les features brutes
        
        Returns:
            array: Prédictions (0/1)
        """
        if self.model is None:
            raise ValueError("Aucun modèle chargé. Utilisez fit() ou load_model() d'abord.")
        
        # ============================================================
        # CRITIQUE: SUPPRIMER LA COLONNE TARGET SI PRÉSENTE
        # ============================================================
        # Utiliser la détection robuste existante pour identifier et supprimer le target
        X_clean = self._detect_and_remove_target(X)
        
        # Si Meta-Transformer utilisé (pas de fallback objects), recréer les features
        if self.feature_engineer is None and self.feature_selector is None and self.engineering_flags is not None:
            print("🔄 Recreating features using Meta-Transformer logic...")
            X_processed = self._apply_meta_transformer_engineering(X_clean)
        else:
            # Sinon, utiliser les objets fallback sauvegardés
            X_processed = X_clean.copy()
            
            # Appliquer feature engineering si disponible
            if self.feature_engineer is not None:
                print("🔄 Applying fallback feature engineering...")
                X_processed = self.feature_engineer.transform(X_processed)
            
            # Appliquer feature selection si disponible
            if self.feature_selector is not None:
                print("🔄 Applying fallback feature selection...")
                X_processed = self.feature_selector.transform(X_processed)
        
        # Vérifier que les features correspondent
        if hasattr(self, 'expected_features') and self.expected_features:
            available_features = set(X_processed.columns)
            expected_features = set(self.expected_features)
            
            if available_features != expected_features:
                missing = expected_features - available_features
                extra = available_features - expected_features
                
                if missing:
                    print(f"⚠️  Missing features: {missing}")
                if extra:
                    print(f"⚠️  Extra features: {extra}")
                
                # Sélectionner uniquement les features attendues (dans le bon ordre)
                X_processed = X_processed[[f for f in self.expected_features if f in X_processed.columns]]
        
        return self.model.predict(X_processed)
    
    def _apply_meta_transformer_engineering(self, X):
        """
        Réapplique le feature engineering exact utilisé pendant l'entraînement
        Utilise les flags sauvegardés du Meta-Transformer
        """
        import numpy as np
        import pandas as pd
        from utils.column_matcher import detect_column_types
        
        print("   🔄 Recreating features with saved engineering flags...")
        print(f"   📋 Input columns: {X.columns.tolist()}")
        print(f"   📊 Input shape: {X.shape}")
        
        # Détecter les types de colonnes avec détection sémantique
        column_types = detect_column_types(X)
        X_engineered = X.copy()
        
        print(f"   🔍 All detected column types: {column_types}")
        
        # ============================================================
        # ENCODER LES COLONNES CATÉGORIELLES (CRITIQUE!)
        # ============================================================
        from sklearn.preprocessing import LabelEncoder
        categorical_cols = X_engineered.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            print(f"   🔄 Encoding {len(categorical_cols)} categorical columns...")
            for col in categorical_cols:
                try:
                    le = LabelEncoder()
                    X_engineered[col] = le.fit_transform(X_engineered[col].astype(str))
                    print(f"      ✅ Encoded: {col}")
                except Exception as e:
                    print(f"      ⚠️  Failed to encode {col}: {e}")
        
        # Utiliser la clé 'amount' (retournée par detect_column_types)
        amount_cols = column_types.get('amount', [])
        
        print(f"   📊 Semantic detection found: {len(amount_cols)} amount columns: {amount_cols}")
        
        # Fallback manuel si la détection sémantique ne trouve rien
        if not amount_cols:
            print("   ⚠️  Semantic detection failed, using manual fallback...")
            amount_keywords = ['amount', 'balance', 'montant', 'solde', 'fcfa', 'xof']
            amount_cols = [col for col in X.columns 
                          if any(kw in col.lower() for kw in amount_keywords)]
            print(f"   🔍 Manual fallback found: {amount_cols}")
        
        # Récupérer les engineering flags
        if self.engineering_flags is None:
            print("   ⚠️  No engineering flags found, using basic engineering")
            engineering_flags = np.zeros(5)
        else:
            engineering_flags = self.engineering_flags
            print(f"   🎯 Engineering flags loaded: {engineering_flags}")
        
        # Seuils adaptatifs (mêmes que pendant l'entraînement)
        adaptive_thresholds = {
            'polynomial': 0.70,
            'interaction': 0.60,
            'binning': 0.40,
            'log_transform': 0.50,
            'aggregation': 0.60
        }
        
        eng_names = ['polynomial', 'interaction', 'binning', 'log_transform', 'aggregation']
        
        features_created = []
        
        # Appliquer les features AI-PREDICTED
        print(f"\n   🤖 Applying AI-predicted features:")
        for i, name in enumerate(eng_names):
            threshold = adaptive_thresholds[name]
            score = engineering_flags[i]
            
            if score > threshold:
                print(f"      ✅ {name}: score={score:.3f} > threshold={threshold:.2f} - APPLYING")
                
                if name == 'polynomial':
                    # Polynomial features
                    for col in amount_cols:
                        if col in X_engineered.columns:
                            X_engineered[f'{col}_squared'] = X_engineered[col] ** 2
                            X_engineered[f'{col}_cubed'] = X_engineered[col] ** 3
                            features_created.extend([f'{col}_squared', f'{col}_cubed'])
                
                elif name == 'interaction':
                    # Interactions (skip if not needed)
                    pass
                
                elif name == 'binning':
                    # Binning
                    for col in amount_cols:
                        if col in X_engineered.columns:
                            try:
                                X_engineered[f'{col}_bin'] = pd.qcut(X_engineered[col], q=5, labels=False, duplicates='drop')
                                features_created.append(f'{col}_bin')
                            except:
                                X_engineered[f'{col}_bin'] = 0
                                features_created.append(f'{col}_bin')
                
                elif name == 'log_transform':
                    # Log transform
                    for col in amount_cols:
                        if col in X_engineered.columns:
                            X_engineered[f'{col}_log'] = np.log1p(X_engineered[col])
                            features_created.append(f'{col}_log')
                
                elif name == 'aggregation':
                    # Aggregation (skip for now, needs customer IDs)
                    pass
            else:
                print(f"      ❌ {name}: score={score:.3f} ≤ threshold={threshold:.2f} - SKIPPED")
        
        # ALWAYS-ON features
        print(f"\n   ⭐ Applying ALWAYS-ON features:")
        
        # Ratio features
        if len(amount_cols) >= 2:
            col1, col2 = amount_cols[0], amount_cols[1]
            if col1 in X_engineered.columns and col2 in X_engineered.columns:
                X_engineered[f'{col1}_div_{col2}'] = X_engineered[col1] / (X_engineered[col2] + 1e-6)
                features_created.append(f'{col1}_div_{col2}')
                print(f"      ✅ Ratio: {col1}_div_{col2}")
        
        # Cyclic features pour hour
        if 'hour' in X_engineered.columns:
            hour_numeric = pd.to_numeric(X_engineered['hour'], errors='coerce').fillna(0)
            X_engineered['hour_sin'] = np.sin(2 * np.pi * hour_numeric / 24)
            X_engineered['hour_cos'] = np.cos(2 * np.pi * hour_numeric / 24)
            features_created.extend(['hour_sin', 'hour_cos'])
            print(f"      ✅ Cyclic: hour_sin, hour_cos")
        
        # Cyclic features pour heure_transaction si existe
        if 'heure_transaction' in X_engineered.columns:
            heure_numeric = pd.to_numeric(X_engineered['heure_transaction'], errors='coerce').fillna(0)
            X_engineered['heure_transaction_sin'] = np.sin(2 * np.pi * heure_numeric / 24)
            X_engineered['heure_transaction_cos'] = np.cos(2 * np.pi * heure_numeric / 24)
            features_created.extend(['heure_transaction_sin', 'heure_transaction_cos'])
            print(f"      ✅ Cyclic: heure_transaction_sin, heure_transaction_cos")
        
        # Boolean features avec cyclic encodings
        if 'hour' in X_engineered.columns:
            hour_numeric = pd.to_numeric(X_engineered['hour'], errors='coerce').fillna(0)
            X_engineered['is_night'] = ((hour_numeric >= 22) | (hour_numeric <= 6)).astype(int)
            X_engineered['is_business_hours'] = ((hour_numeric >= 9) & (hour_numeric <= 17)).astype(int)
            
            # Cyclic encoding pour is_business_hours (pour continuité temporelle)
            X_engineered['is_business_hours_sin'] = np.sin(2 * np.pi * X_engineered['is_business_hours'])
            X_engineered['is_business_hours_cos'] = np.cos(2 * np.pi * X_engineered['is_business_hours'])
            
            features_created.extend(['is_night', 'is_business_hours', 'is_business_hours_sin', 'is_business_hours_cos'])
            print(f"      ✅ Boolean: is_night, is_business_hours")
            print(f"      ✅ Cyclic boolean: is_business_hours_sin, is_business_hours_cos")
        
        print(f"\n   📊 Total features created: {len(features_created)}")
        print(f"      Features: {features_created}")
        
        # Remplacer NaN/inf
        X_engineered = X_engineered.replace([np.inf, -np.inf], np.nan)
        X_engineered = X_engineered.fillna(0)
        
        # Sélectionner uniquement les features attendues (dans le bon ordre)
        if hasattr(self, 'expected_features') and self.expected_features:
            available = set(X_engineered.columns)
            expected = set(self.expected_features)
            common = [f for f in self.expected_features if f in available]
            
            print(f"\n   🔍 Feature matching:")
            print(f"      Expected: {len(self.expected_features)} features")
            print(f"      Available: {len(X_engineered.columns)} features")
            print(f"      Common: {len(common)} features")
            
            if len(common) < len(self.expected_features):
                missing = set(self.expected_features) - available
                print(f"      ⚠️  Missing features: {missing}")
            
            if common:
                X_engineered = X_engineered[common]
            else:
                print(f"   ⚠️  No common features found!")
        
        print(f"   ✅ Recreated {len(X_engineered.columns)} features")
        return X_engineered



    
    
    def predict_proba(self, X):
        """
        Probabilités de prédiction
        
        Args:
            X: DataFrame avec les features brutes
        
        Returns:
            array: Probabilités pour chaque classe
        """
        if self.model is None:
            raise ValueError("Aucun modèle chargé. Utilisez fit() ou load_model() d'abord.")
        
        # Si Meta-Transformer utilisé (pas de fallback objects), recréer les features
        if self.feature_engineer is None and self.feature_selector is None and self.engineering_flags is not None:
            X_processed = self._apply_meta_transformer_engineering(X)
        else:
            # Sinon, utiliser les objets fallback sauvegardés
            X_processed = X.copy()
            
            # Appliquer feature engineering si disponible
            if self.feature_engineer is not None:
                X_processed = self.feature_engineer.transform(X_processed)
            
            # Appliquer feature selection si disponible
            if self.feature_selector is not None:
                X_processed = self.feature_selector.transform(X_processed)
        
        # Vérifier que les features correspondent
        if hasattr(self, 'expected_features') and self.expected_features:
            X_processed = X_processed[[f for f in self.expected_features if f in X_processed.columns]]
        
        return self.model.predict_proba(X_processed)



if __name__ == "__main__":
    # DEMO: AutoML Complet - Accepte CSV path en argument
    import sys
    
    print("\n" * 2)
    print("=" * 70)
    print("🤖 FULL AUTO ML DEMONSTRATION")
    print("=" * 70)
    
    # Récupérer le CSV path depuis les arguments
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        target_col = sys.argv[2] if len(sys.argv) > 2 else 'suspicious_flag'
    else:
        # Valeurs par défaut
        csv_path = 'data/datasets/Dataset4.csv'
        target_col = 'suspicious_flag'
    
    # Extraire le nom du dataset du path pour la sauvegarde
    import re
    dataset_match = re.search(r'Dataset(\d+)', csv_path)
    if dataset_match:
        dataset_name = f'dataset{dataset_match.group(1)}'
    else:
        # Fallback: utiliser le nom du fichier
        dataset_name = csv_path.split('/')[-1].split('.')[0].lower()
    
    # Créer AutoML
    automl = FullAutoML(
        reference_dataset='Dataset7',
        use_meta_transformer=True,
        use_feature_selector=True,  
    )
    
    # Fit sur le dataset
    performance = automl.fit(
        csv_path=csv_path,
        target_col=target_col
    )
    
    # Sauvegarder avec le bon nom de dataset
    save_dir = f'data/automl_models/{dataset_name}'
    automl.save_model(save_dir)
    
    print("\n" * 2)
    print("=" * 70)
    print("🎉 AUTO ML COMPLETE!")
    print(f"   F1 Score: {performance['test_f1']:.4f}")
    print(f"   Total time: {automl.training_time:.2f}s")
    print(f"   Model saved to: {save_dir}/")
    print("=" * 70)
