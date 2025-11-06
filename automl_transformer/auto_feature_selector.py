"""
🎯 AUTO FEATURE SELECTION
Utilise les feature importance sauvegardées pour sélectionner automatiquement
les meilleures features
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
import os
import xgboost as xgb
import numpy as np
from utils.column_matcher import ColumnMatcher


class AutoFeatureSelector:
    """
    Sélectionne automatiquement les features importantes
    Basé sur les patterns appris de Dataset1-7
    """
    
    def __init__(self, importance_threshold=0.01):
        """
        Args:
            importance_threshold: Seuil minimum d'importance (défaut: 1%)
        """
        self.importance_threshold = importance_threshold
        self.selected_features = []
        self.feature_scores = {}
        self.column_matcher = ColumnMatcher(fuzzy_threshold=0.7)
    
    def detect_target_column(self, df):
        """
        Détecte automatiquement la colonne target (fraud) avec matching sémantique
        
        Args:
            df: DataFrame pandas
            
        Returns:
            str: Nom de la colonne target détectée, ou None
        """
        candidates = []
        
        # Étape 1: Chercher toutes les colonnes avec groupe sémantique 'fraud'
        for col in df.columns:
            semantic_group = self.column_matcher.get_semantic_group(col)
            if semantic_group == 'fraud':
                n_unique = df[col].nunique()
                if n_unique <= 5:
                    candidates.append((col, 100, n_unique))  # Score 100 pour semantic match
        
        # Étape 2: Chercher par mots-clés avec scoring
        keyword_scores = {
            'fraud_flag': 90,
            'is_fraud': 90,
            'fraudulent': 85,
            'skimming_detected': 85,  # Pour ATM skimming fraud
            'suspicious_flag': 80,
            'is_suspicious': 80,
            'is_anomaly': 80,  # Pour anomaly detection
            'anomaly': 75,
            'payment_irregularity': 75,  # Pour loan payment fraud
            'irregularity': 70,
            'fraud': 70,
            'suspicious': 60,
            'fraude': 70,
            'skimming': 60,
            'label': 40,
            'target': 40
        }
        
        for col in df.columns:
            col_lower = col.lower()
            n_unique = df[col].nunique()
            
            # Ne considérer que les colonnes binaires ou catégorielles (2-5 valeurs)
            if 2 <= n_unique <= 5:
                for keyword, score in keyword_scores.items():
                    if keyword in col_lower:
                        # Bonus si c'est vraiment binaire (2 valeurs exactement)
                        if n_unique == 2:
                            score += 20
                        # Bonus si c'est de type object (oui/non, yes/no)
                        if df[col].dtype == 'object':
                            score += 10
                        
                        candidates.append((col, score, n_unique))
                        break  # Ne compter qu'une fois par colonne
        
        # Étape 3: Choisir le meilleur candidat
        if not candidates:
            return None
        
        # Trier par score (desc), puis par nombre de valeurs uniques (asc)
        candidates.sort(key=lambda x: (-x[1], x[2]))
        
        return candidates[0][0]
    
    def load_reference_importance(self, dataset_name):
        """Charge le fichier feature importance d'un dataset de référence"""
        
        importance_path = f'data/Feature_importance/{dataset_name}_production_feature_importance.json'
        
        if not os.path.exists(importance_path):
            print(f"⚠️  No feature importance file found for {dataset_name}")
            return None
        
        with open(importance_path, 'r') as f:
            importance_data = json.load(f)
        
        return importance_data
    
    def select_from_importance_file(self, importance_data, top_n=None):
        """
        Sélectionne les features basées sur un fichier d'importance
        
        Args:
            importance_data: Données JSON de feature importance
            top_n: Nombre max de features à garder (None = toutes au-dessus du seuil)
        
        Returns:
            Liste des features sélectionnées
        """
        
        if not importance_data:
            return []
        
        # Extraire les features et leurs scores
        production_features = importance_data.get('production_feature_importance', [])
        
        # Trier par importance
        sorted_features = sorted(
            production_features,
            key=lambda x: x.get('importance', 0),
            reverse=True
        )
        
        # Filtrer par seuil
        selected = []
        for feat in sorted_features:
            feat_name = feat.get('feature_name')
            feat_importance = feat.get('importance', 0)
            
            if feat_importance >= self.importance_threshold:
                selected.append(feat_name)
                self.feature_scores[feat_name] = feat_importance
        
        # Limiter au top_n si spécifié
        if top_n:
            selected = selected[:top_n]
        
        self.selected_features = selected
        return selected
    
    def select_by_pattern_matching(self, available_features, reference_features):
        """
        Sélectionne les features disponibles qui matchent les patterns des references
        
        Args:
            available_features: Features disponibles dans le nouveau dataset
            reference_features: Features importantes des datasets de référence
        
        Returns:
            Liste des features sélectionnées
        """
        
        selected = []
        
        # Patterns de features importantes (appris de Dataset1-7)
        important_patterns = [
            '_log', '_sqrt', '_bin', '_is_high', '_is_low',
            'is_same_', 'is_international', 'transaction_hour',
            'transaction_weekday', 'is_business_hours', 'is_night',
            'interaction', '_length', '_words'
        ]
        
        for feat in available_features:
            # Match exact
            if feat in reference_features:
                selected.append(feat)
            
            # Match pattern
            elif any(pattern in feat for pattern in important_patterns):
                selected.append(feat)
        
        self.selected_features = selected
        return selected
    
    def select_top_k_features(self, X, y=None, k=30, method='variance'):
        """
        Sélectionne top K features basé sur une méthode statistique
        
        Args:
            X: DataFrame de features
            y: Target (optionnel)
            k: Nombre de features à sélectionner
            method: 'variance', 'correlation', ou 'mutual_info'
        
        Returns:
            Liste des top K features
        """
        
        if method == 'variance':
            # Sélectionner features avec plus grande variance
            variances = X.var()
            top_features = variances.nlargest(k).index.tolist()
        
        elif method == 'correlation' and y is not None:
            # Sélectionner features avec plus grande correlation au target
            correlations = X.corrwith(y).abs()
            top_features = correlations.nlargest(k).index.tolist()
        
        else:
            # Par défaut, garder toutes
            top_features = X.columns.tolist()[:k]
        
        self.selected_features = top_features
        return top_features
    
    def transform(self, X):
        """
        Applique la sélection de features
        
        Args:
            X: DataFrame avec toutes les features
        
        Returns:
            DataFrame avec features sélectionnées uniquement
        """
        
        # Features disponibles
        available = set(X.columns)
        selected = set(self.selected_features)
        
        # Intersection
        final_features = list(available & selected)
        
        if not final_features:
            print("⚠️  No selected features found in dataset, using all features")
            final_features = X.columns.tolist()
        
        print(f"📊 Feature Selection:")
        print(f"   Total features available: {len(available)}")
        print(f"   Features selected: {len(final_features)}")
        print(f"   Reduction: {len(available) - len(final_features)} features removed")
        
        return X[final_features]
    
    def calculate_importances_on_dataset(self, X, y, n_estimators=50):
        """
        Calcule les feature importances directement sur le dataset actuel
        (Alternative au transfer learning avec reference_dataset)
        
        Args:
            X: DataFrame de features
            y: Target (obligatoire)
            n_estimators: Nombre d'arbres pour le modèle rapide (défaut: 50)
        
        Returns:
            dict: {feature_name: importance_score}
        """
        print(f"   📊 Calculating importances on current dataset ({n_estimators} trees)...")
        
        # Entraîner un modèle rapide pour obtenir les importances
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        
        model.fit(X, y)
        
        # Récupérer les importances
        importances = dict(zip(X.columns, model.feature_importances_))
        
        # Normaliser (somme = 1)
        total_importance = sum(importances.values())
        if total_importance > 0:
            importances = {k: v/total_importance for k, v in importances.items()}
    def calculate_importances_on_dataset(self, X, y, n_estimators=50):
        """
        Calcule les feature importances directement sur le dataset actuel
        (Alternative au transfer learning avec reference_dataset)
        
        Args:
            X: DataFrame de features
            y: Target (obligatoire)
            n_estimators: Nombre d'arbres pour le modèle rapide (défaut: 50)
        
        Returns:
            dict: {feature_name: importance_score}
        """
        print(f"   📊 Calculating importances on current dataset ({n_estimators} trees)...")
        
        # Entraîner un modèle rapide pour obtenir les importances
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        
        model.fit(X, y)
        
        # Récupérer les importances
        importances = dict(zip(X.columns, model.feature_importances_))
        
        # Normaliser (somme = 1)
        total_importance = sum(importances.values())
        if total_importance > 0:
            importances = {k: v/total_importance for k, v in importances.items()}
        
        print(f"   ✅ Importances calculated: {len(importances)} features")
        
        return importances
    
    def select_features_intelligently(self, importances, X, min_cumulative=0.95, 
                                     min_features=10, max_features=None):
        """
        Sélection intelligente des features avec plusieurs critères
        
        Args:
            importances: dict {feature: importance}
            X: DataFrame pour référence
            min_cumulative: Garder features jusqu'à atteindre X% de l'importance totale (défaut: 95%)
            min_features: Nombre minimum de features à garder (défaut: 10)
            max_features: Nombre maximum de features à garder (défaut: None = illimité)
        
        Returns:
            list: Features sélectionnées
        """
        
        # Trier par importance décroissante
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        # Stratégie 1: Importance cumulée (garder features jusqu'à X% du total)
        cumulative_sum = 0
        cumulative_threshold_idx = 0
        
        for i, (feat, imp) in enumerate(sorted_features):
            cumulative_sum += imp
            if cumulative_sum >= min_cumulative:
                cumulative_threshold_idx = i + 1
                break
        
        # Stratégie 2: Supprimer seulement les features strictement inutiles (importance = 0)
        non_zero_features = [feat for feat, imp in sorted_features if imp > 0]
        
        # Stratégie 3: Seuil adaptatif basé sur la distribution
        import numpy as np
        importance_values = [imp for _, imp in sorted_features if imp > 0]
        if importance_values:
            mean_imp = np.mean(importance_values)
            std_imp = np.std(importance_values)
            adaptive_threshold = max(0.001, mean_imp - std_imp)  # moyenne - 1 écart-type
        else:
            adaptive_threshold = 0.001
        
        adaptive_features = [feat for feat, imp in sorted_features if imp >= adaptive_threshold]
        
        # Choisir la stratégie qui garde le plus de features (mais pas trop)
        candidates = [
            (cumulative_threshold_idx, f"cumulative {min_cumulative:.0%}"),
            (len(non_zero_features), "non-zero only"),
            (len(adaptive_features), f"adaptive (>{adaptive_threshold:.4f})")
        ]
        
        # Filtrer avec min/max
        valid_candidates = []
        for n_feat, strategy in candidates:
            if n_feat >= min_features:
                if max_features is None or n_feat <= max_features:
                    valid_candidates.append((n_feat, strategy))
        
        # Si aucun candidat valide, prendre min_features
        if not valid_candidates:
            n_selected = min_features
            strategy_used = f"minimum {min_features}"
        else:
            # Prendre le plus permissif (garde le plus de features)
            n_selected, strategy_used = max(valid_candidates, key=lambda x: x[0])
        
        selected_features = [feat for feat, _ in sorted_features[:n_selected]]
        
        print(f"   🎯 Selection strategy: {strategy_used}")
        print(f"   📊 Features selected: {n_selected}/{len(X.columns)}")
        
        # Statistiques
        if n_selected > 0:
            selected_importance = sum(importances[f] for f in selected_features)
            print(f"   📈 Coverage: {selected_importance:.1%} of total importance")
        
        return selected_features
    
    def fit_transform(self, X, y=None, reference_dataset='Dataset7', use_direct_calculation=False):
        """
        Pipeline complet: sélection + transformation
        
        Args:
            X: DataFrame de features
            y: Target (optionnel mais OBLIGATOIRE si use_direct_calculation=True)
            reference_dataset: Dataset de référence pour importance (ignoré si use_direct_calculation=True)
            use_direct_calculation: Si True, calcule les importances sur le dataset actuel
                                    Si False, utilise le transfer learning avec reference_dataset
        
        Returns:
            DataFrame avec features sélectionnées
        """
        
        print(f"🎯 AUTO FEATURE SELECTION STARTED")
        
        if use_direct_calculation:
            # Mode 1: Calculer importances directement sur le dataset actuel
            print(f"   Mode: DIRECT CALCULATION (dataset-specific, intelligent)")
            
            if y is None:
                raise ValueError("❌ y (target) is required when use_direct_calculation=True")
            
            # Calculer les importances
            importances = self.calculate_importances_on_dataset(X, y)
            
            # Sélection intelligente (au lieu du simple seuil)
            # Garde jusqu'à 95% de l'importance totale, min 15 features
            selected = self.select_features_intelligently(
                importances, 
                X,
                min_cumulative=0.95,  # Garder features couvrant 95% importance
                min_features=15,      # Minimum 15 features
                max_features=None     # Pas de maximum
            )
            
            # Sauvegarder les scores
            self.feature_scores = importances
            
            print(f"   ✅ Intelligent selection completed")

            
        else:
            # Mode 2: Transfer learning avec reference_dataset
            print(f"   Mode: TRANSFER LEARNING")
            print(f"   Reference dataset: {reference_dataset}")
            
            # Charger importance de référence
            importance_data = self.load_reference_importance(reference_dataset)
            
            if importance_data:
                # Sélection basée sur importance
                reference_features = self.select_from_importance_file(importance_data)
                selected = self.select_by_pattern_matching(X.columns.tolist(), reference_features)
                
                print(f"   Reference features: {len(reference_features)}")
                print(f"   Matched features: {len(selected)}")
            else:
                # Fallback: sélection statistique
                print("   ⚠️  Reference not found, using statistical selection (variance-based)")
                selected = self.select_top_k_features(X, y, k=30, method='variance')
        
        # Sauvegarder les features sélectionnées
        self.selected_features = selected
        
        # Transformer
        X_selected = self.transform(X)
        
        print(f"✅ AUTO FEATURE SELECTION COMPLETED")
        
        return X_selected


if __name__ == "__main__":
    # Test sur Dataset8
    print("=" * 60)
    print("TESTING AUTO FEATURE SELECTOR ON DATASET8")
    print("=" * 60)
    
    # D'abord générer les features
    from automl_transformer.auto_feature_engineer import AutoFeatureEngineer
    
    df = pd.read_csv('data/datasets/Dataset8.csv')
    
    # Auto feature engineering
    engineer = AutoFeatureEngineer()
    X = engineer.fit_transform(df, target_col='is_fraud')
    
    # Auto feature selection
    selector = AutoFeatureSelector(importance_threshold=0.01)
    X_selected = selector.fit_transform(X, reference_dataset='Dataset7')
    
    print(f"\n📊 FINAL RESULT:")
    print(f"   Features after engineering: {X.shape[1]}")
    print(f"   Features after selection: {X_selected.shape[1]}")
    print(f"\n   Top 10 selected features:")
    for i, feat in enumerate(X_selected.columns[:10], 1):
        score = selector.feature_scores.get(feat, 'N/A')
        print(f"     {i}. {feat} (importance: {score})")
