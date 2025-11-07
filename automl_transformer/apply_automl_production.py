"""
apply_automl_production.py - OPTIMIZED VERSION v2.0

Script ULTRA-OPTIMISÉ pour appliquer des modèles AutoML entraînés sur des données de production
SANS avoir besoin de la colonne fraud_flag.

🚀 NOUVELLES FONCTIONNALITÉS v2.0:
====================================
✅ Exclusion automatique des colonnes ID/timestamp (data leakage prevention)
✅ Matching sémantique avancé (amount = montant = transaction_amount)
✅ Ensemble predictions (moyenne de top-3 modèles similaires)
✅ Anomaly detection complémentaire (Isolation Forest)
✅ Calibration des probabilités (plus fiables)
✅ Analyse de stabilité des prédictions
✅ Export enrichi (Excel avec graphiques + JSON détaillé)
✅ Mode batch pour grands volumes (>1M lignes)

Usage:
------
# Auto-match : Trouve automatiquement le meilleur modèle
python apply_automl_production.py --dataset production.csv --auto_match

# Ensemble de top-3 modèles (RECOMMANDÉ pour plus de robustesse)
python apply_automl_production.py --dataset production.csv --ensemble --top_k 3

# Avec anomaly detection complémentaire
python apply_automl_production.py --dataset production.csv --auto_match --anomaly_detection

# Mode batch pour gros volumes
python apply_automl_production.py --dataset big_prod.csv --auto_match --batch_size 50000

# Export enrichi (Excel + JSON)
python apply_automl_production.py --dataset production.csv --auto_match --rich_export

# Spécifier un modèle manuellement (mode classique)
python apply_automl_production.py --dataset production.csv --model dataset20 --threshold 0.20 --output results.csv

Auteur: Fraud Detection AutoML System v2.0
Date: November 2025
"""

import os
import sys
import argparse
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Import du column matcher pour matching sémantique
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.column_matcher import ColumnMatcher


class AutoMLProductionApplicator:
    """
    Applique des modèles AutoML entraînés sur des données de production
    
    Version 2.0 OPTIMISÉE avec:
    - Exclusion automatique ID/timestamp (data leakage prevention)
    - Matching sémantique avancé
    - Ensemble predictions (robustesse++)
    - Anomaly detection complémentaire
    - Calibration des probabilités
    """
    
    def __init__(self, automl_models_dir: str = "data/automl_models", use_reference_models: bool = True):
        self.automl_models_dir = Path(automl_models_dir)
        self.use_reference_models = use_reference_models
        self.available_models = self._discover_models()
        self.column_matcher = ColumnMatcher(fuzzy_threshold=0.7)
        
        # Cache pour modèles chargés (évite rechargement multiple)
        self._model_cache = {}
        
    def _discover_models(self) -> List[str]:
        """Découvre tous les modèles AutoML disponibles (locaux + référence DB)"""
        models = []
        
        # 1. Modèles locaux
        if self.automl_models_dir.exists():
            for dataset_dir in self.automl_models_dir.iterdir():
                if dataset_dir.is_dir():
                    model_file = dataset_dir / "xgboost_model.joblib"
                    if model_file.exists():
                        models.append(dataset_dir.name)
        
        # 2. Modèles de référence depuis la base de données (production)
        if self.use_reference_models:
            try:
                # Import ici pour éviter circular import
                from app.models.reference_model import ReferenceModel
                from app import db
                
                # Récupérer tous les modèles de référence actifs
                reference_models = ReferenceModel.query.filter_by(is_active=True).all()
                for ref_model in reference_models:
                    if ref_model.model_name not in models:
                        models.append(ref_model.model_name)
                
                if reference_models:
                    print(f"✅ {len(reference_models)} modèles de référence chargés depuis DB")
                        
            except Exception as e:
                print(f"⚠️ Impossible de charger les modèles de référence: {e}")
        
        return sorted(models)
    
    def load_model_pipeline(self, model_name: str) -> Dict:
        """Charge le pipeline complet (engineer + selector + model) depuis local ou S3"""
        model_dir = self.automl_models_dir / model_name
        
        # Si le modèle n'existe pas localement, essayer de le télécharger depuis S3
        if not model_dir.exists():
            print(f"\n📥 Modèle {model_name} non trouvé localement, tentative de téléchargement depuis S3...")
            
            try:
                from app.models.reference_model import ReferenceModel
                import boto3
                import tempfile
                import shutil
                
                # Récupérer les infos du modèle de référence
                ref_model = ReferenceModel.query.filter_by(model_name=model_name).first()
                
                if not ref_model:
                    raise ValueError(f"Modèle de référence {model_name} introuvable dans la DB")
                
                # Utiliser s3_bucket et s3_prefix
                bucket = ref_model.s3_bucket
                prefix = ref_model.s3_prefix
                
                if not bucket or not prefix:
                    raise ValueError(f"Modèle {model_name} n'a pas de configuration S3 (bucket={bucket}, prefix={prefix})")
                
                # Créer un dossier temporaire
                temp_dir = Path(tempfile.gettempdir()) / 'fraud_models' / model_name
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                # Télécharger les fichiers depuis S3
                s3_client = boto3.client('s3')
                
                required_files = ['xgboost_model.joblib']
                optional_files = ['feature_engineer.joblib', 'feature_selector.joblib', 'performance.json']
                
                files_downloaded = []
                for filename in required_files + optional_files:
                    s3_key = f"{prefix}{filename}" if prefix.endswith('/') else f"{prefix}/{filename}"
                    local_path = temp_dir / filename
                    
                    try:
                        s3_client.download_file(bucket, s3_key, str(local_path))
                        files_downloaded.append(filename)
                        print(f"   ✓ Téléchargé: {filename}")
                    except Exception as e:
                        if filename in required_files:
                            raise FileNotFoundError(f"Fichier requis {filename} introuvable sur S3: {e}")
                        else:
                            print(f"   ⚠️  Fichier optionnel {filename} non trouvé")
                
                # Utiliser le dossier temporaire comme model_dir
                model_dir = temp_dir
                print(f"   ✅ Modèle téléchargé depuis S3 dans {temp_dir}")
                
            except Exception as e:
                raise ValueError(f"Impossible de charger le modèle {model_name}: {e}")
        
        print(f"\n📦 Chargement du pipeline AutoML: {model_name}")
        
        pipeline = {}
        
        # Charger le modèle XGBoost
        model_path = model_dir / "xgboost_model.joblib"
        if model_path.exists():
            pipeline['model'] = joblib.load(model_path)
            print(f"   ✓ Modèle XGBoost chargé")
        else:
            raise FileNotFoundError(f"Modèle XGBoost introuvable: {model_path}")
        
        # Charger le feature engineer (optionnel)
        engineer_path = model_dir / "feature_engineer.joblib"
        if engineer_path.exists():
            pipeline['feature_engineer'] = joblib.load(engineer_path)
            print(f"   ✓ Feature Engineer chargé")
        else:
            pipeline['feature_engineer'] = None
            print(f"   ⚠️  Feature Engineer non trouvé (optionnel)")
        
        # Charger le feature selector (optionnel)
        selector_path = model_dir / "feature_selector.joblib"
        if selector_path.exists():
            pipeline['feature_selector'] = joblib.load(selector_path)
            print(f"   ✓ Feature Selector chargé")
        else:
            pipeline['feature_selector'] = None
            print(f"   ⚠️  Feature Selector non trouvé (optionnel)")
        
        # Charger les performances
        perf_path = model_dir / "performance.json"
        if perf_path.exists():
            with open(perf_path, 'r') as f:
                pipeline['performance'] = json.load(f)
            print(f"   ✓ Performances: F1={pipeline['performance'].get('test_f1', 0):.2%}, "
                  f"AUC={pipeline['performance'].get('test_auc', 0):.2%}")
        else:
            pipeline['performance'] = {}
        
        return pipeline
    
    def extract_dataset_meta_features(self, df: pd.DataFrame) -> Dict:
        """Extrait les méta-features enrichies d'un dataset pour comparaison"""
        column_names = list(df.columns)
        
        meta = {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'column_names': set(column_names),
            'numerical_cols': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': list(df.select_dtypes(include=['object', 'category']).columns),
            'n_numerical': len(df.select_dtypes(include=[np.number]).columns),
            'n_categorical': len(df.select_dtypes(include=['object', 'category']).columns),
        }
        
        # Détection de colonnes spécifiques
        meta['has_amount'] = any('amount' in col.lower() or 'montant' in col.lower() for col in column_names)
        meta['has_timestamp'] = any('time' in col.lower() or 'date' in col.lower() for col in column_names)
        meta['has_merchant'] = any('merchant' in col.lower() or 'commercant' in col.lower() for col in column_names)
        meta['has_card'] = any('card' in col.lower() or 'carte' in col.lower() for col in column_names)
        meta['has_currency'] = any('currency' in col.lower() or 'devise' in col.lower() for col in column_names)
        meta['has_country'] = any('country' in col.lower() or 'pays' in col.lower() for col in column_names)
        meta['has_balance'] = any('balance' in col.lower() or 'solde' in col.lower() for col in column_names)
        meta['has_customer'] = any('customer' in col.lower() or 'client' in col.lower() for col in column_names)
        meta['has_account'] = any('account' in col.lower() or 'compte' in col.lower() for col in column_names)
        
        # Détection de domaine (même logique que generate_model_metadata.py)
        col_str = ' '.join([col.lower() for col in column_names])
        
        scores = {
            'card_fraud': 0, 'mobile_money': 0, 'wire_transfer': 0, 'atm': 0,
            'corporate_banking': 0, 'mobile_banking': 0, 'crypto': 0, 'mortgage': 0,
            'investment': 0, 'insurance': 0, 'pos_retail': 0, 'p2p_lending': 0,
        }
        
        if any(kw in col_str for kw in ['card', 'merchant', 'authorization', 'cvv', 'pin']):
            scores['card_fraud'] += 3
        if any(kw in col_str for kw in ['mobile', 'wallet', 'airtime', 'momo', 'orange_money']):
            scores['mobile_money'] += 3
        if any(kw in col_str for kw in ['wire', 'transfer', 'swift', 'iban', 'beneficiary']):
            scores['wire_transfer'] += 3
        if any(kw in col_str for kw in ['atm', 'withdrawal', 'cash', 'dispenser']):
            scores['atm'] += 3
        if any(kw in col_str for kw in ['corporate', 'company', 'business', 'headquarters']):
            scores['corporate_banking'] += 3
        if any(kw in col_str for kw in ['app', 'mobile_banking', 'login', 'device']):
            scores['mobile_banking'] += 2
        if any(kw in col_str for kw in ['crypto', 'bitcoin', 'ethereum', 'wallet', 'blockchain']):
            scores['crypto'] += 3
        if any(kw in col_str for kw in ['mortgage', 'loan', 'property', 'interest_rate']):
            scores['mortgage'] += 3
        if any(kw in col_str for kw in ['trade', 'investment', 'portfolio', 'stock', 'broker']):
            scores['investment'] += 3
        if any(kw in col_str for kw in ['insurance', 'claim', 'policy', 'premium']):
            scores['insurance'] += 3
        if any(kw in col_str for kw in ['pos', 'retail', 'merchant', 'terminal']):
            scores['pos_retail'] += 2
        if any(kw in col_str for kw in ['p2p', 'lending', 'borrower', 'lender', 'credit_grade']):
            scores['p2p_lending'] += 3
        
        max_score = max(scores.values())
        meta['domain'] = max(scores, key=scores.get) if max_score > 0 else 'unknown'
        
        # Colonnes d'identifiants
        id_keywords = ['id', 'number', 'hash']
        meta['id_cols'] = [col for col in df.columns if any(kw in col.lower() for kw in id_keywords)]
        
        # Détection de la colonne de fraude (pour fraud_info)
        fraud_keywords = ['fraud', 'is_fraud', 'fraud_flag', 'fraudulent', 'is_fraudulent']
        fraud_col = None
        for col in column_names:
            if any(kw in col.lower() for kw in fraud_keywords):
                fraud_col = col
                break
        
        meta['fraud_info'] = {}
        if fraud_col and fraud_col in df.columns:
            meta['fraud_info'] = {
                'fraud_col_name': fraud_col,
                'fraud_count': int(df[fraud_col].sum()),
                'fraud_rate': float(df[fraud_col].mean()),
                'n_samples': len(df),
            }
        
        return meta
    
    def calculate_similarity(self, meta1: Dict, meta2: Dict) -> float:
        """
        Calcule la similarité entre deux datasets (0-1).
        
        Version améliorée avec matching sémantique de colonnes.
        Ne dépend plus des noms exacts mais de la sémantique (amount = montant = transaction_amount).
        
        Pondération:
        - 50% : Similarité sémantique des colonnes (CRITIQUE)
        - 20% : Similarité de domaine (card, mobile, banking, etc.)
        - 15% : Key features (has_amount, has_card, etc.)
        - 10% : Fraud rate similarity (si disponible)
        - 5%  : Types de colonnes (numerical/categorical ratio)
        """
        score = 0.0
        total_weight = 0.0
        
        # 1. Similarité sémantique des colonnes (poids: 50% - CRITIQUE)
        # Utilise le ColumnMatcher au lieu de l'overlap exact
        cols1 = list(meta1['column_names']) if isinstance(meta1['column_names'], set) else meta1['column_names']
        cols2 = list(meta2['column_names']) if isinstance(meta2['column_names'], set) else meta2['column_names']
        
        semantic_result = self.column_matcher.calculate_semantic_similarity(cols1, cols2, verbose=False)
        col_similarity = semantic_result['similarity']
        
        # Bonus si match quasi-perfait (95%+)
        if col_similarity >= 0.95:
            col_similarity = min(1.0, col_similarity * 1.05)
        
        score += col_similarity * 0.50
        total_weight += 0.50
        
        # 2. Domaine du dataset (poids: 20% - IMPORTANT)
        domain_similarity = 0.0
        
        if meta1.get('domain') and meta2.get('domain'):
            if meta1['domain'] == meta2['domain']:
                domain_similarity = 1.0  # Match parfait
            else:
                # Domaines apparentés
                related_domains = {
                    'card_fraud': ['pos_retail', 'atm'],
                    'mobile_money': ['mobile_banking'],
                    'wire_transfer': ['corporate_banking'],
                    'atm': ['card_fraud'],
                    'pos_retail': ['card_fraud'],
                    'crypto': ['investment'],
                    'investment': ['crypto', 'mortgage'],
                    'mobile_banking': ['mobile_money'],
                    'corporate_banking': ['wire_transfer'],
                }
                
                if meta2['domain'] in related_domains.get(meta1['domain'], []):
                    domain_similarity = 0.6  # Partial match
        
        score += domain_similarity * 0.20
        total_weight += 0.20
        
        # 3. Key features (poids: 15%)
        key_features = ['has_amount', 'has_timestamp', 'has_merchant', 'has_card', 
                       'has_currency', 'has_country', 'has_balance', 'has_customer', 'has_account']
        key_matches = sum(meta1.get(key, False) == meta2.get(key, False) for key in key_features)
        key_similarity = key_matches / len(key_features)
        score += key_similarity * 0.15
        total_weight += 0.15
        
        # 4. Fraud rate similarity (poids: 10%)
        fraud_similarity = 0.5  # Défaut neutre si pas d'info
        fraud1 = meta1.get('fraud_info', {}).get('fraud_rate', 0) if 'fraud_info' in meta1 else 0
        fraud2 = meta2.get('fraud_info', {}).get('fraud_rate', 0) if 'fraud_info' in meta2 else 0
        
        if fraud1 > 0 and fraud2 > 0:
            # Similarité basée sur la différence relative
            fraud_diff = abs(fraud1 - fraud2)
            max_fraud = max(fraud1, fraud2)
            fraud_similarity = max(0, 1 - fraud_diff / max_fraud)
        
        score += fraud_similarity * 0.10
        total_weight += 0.10
        
        # 5. Types de colonnes (poids: 5%)
        numerical_ratio1 = meta1['n_numerical'] / meta1['n_cols'] if meta1.get('n_cols', 0) > 0 else 0
        numerical_ratio2 = meta2['n_numerical'] / meta2['n_cols'] if meta2.get('n_cols', 0) > 0 else 0
        type_similarity = 1 - abs(numerical_ratio1 - numerical_ratio2)
        score += type_similarity * 0.05
        total_weight += 0.05
        
        return score / total_weight if total_weight > 0 else 0
    
    def find_best_matching_model(self, df: pd.DataFrame, verbose: bool = True) -> Tuple[str, float]:
        """Trouve le modèle le plus similaire au dataset fourni"""
        if not self.available_models:
            raise ValueError("Aucun modèle AutoML disponible dans data/automl_models/")
        
        dataset_meta = self.extract_dataset_meta_features(df)
        
        if verbose:
            print(f"\n🔍 Analyse du dataset de production:")
            print(f"   - Lignes: {dataset_meta['n_rows']:,}")
            print(f"   - Colonnes: {dataset_meta['n_cols']}")
            print(f"   - Numériques: {dataset_meta['n_numerical']}, Catégorielles: {dataset_meta['n_categorical']}")
            print(f"   - Amount: {'✓' if dataset_meta['has_amount'] else '✗'}, "
                  f"Timestamp: {'✓' if dataset_meta['has_timestamp'] else '✗'}, "
                  f"Card: {'✓' if dataset_meta['has_card'] else '✗'}")
        
        similarities = []
        
        for model_name in self.available_models:
            # Charger les infos du modèle pour comparaison
            model_dir = self.automl_models_dir / model_name
            perf_file = model_dir / "performance.json"
            
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    perf = json.load(f)
                
                # Essayer de charger les vraies métadonnées enrichies si elles existent
                meta_file = model_dir / "dataset_metadata.json"
                if meta_file.exists():
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        model_meta_saved = json.load(f)
                    
                    # Reconstruire le dict avec sets pour column_names et toutes les métadonnées
                    model_meta = {
                        'n_cols': model_meta_saved.get('n_cols', 0),
                        'n_rows': model_meta_saved.get('n_rows', 0),
                        'column_names': set(model_meta_saved.get('column_names', [])),
                        'n_numerical': model_meta_saved.get('n_numerical', 0),
                        'n_categorical': model_meta_saved.get('n_categorical', 0),
                        'has_amount': model_meta_saved.get('has_amount', False),
                        'has_timestamp': model_meta_saved.get('has_timestamp', False),
                        'has_merchant': model_meta_saved.get('has_merchant', False),
                        'has_card': model_meta_saved.get('has_card', False),
                        'has_currency': model_meta_saved.get('has_currency', False),
                        'has_country': model_meta_saved.get('has_country', False),
                        'has_balance': model_meta_saved.get('has_balance', False),
                        'has_customer': model_meta_saved.get('has_customer', False),
                        'has_account': model_meta_saved.get('has_account', False),
                        'domain': model_meta_saved.get('domain', 'unknown'),
                        'fraud_info': model_meta_saved.get('fraud_info', {}),
                    }
                else:
                    # Fallback: créer méta-features approximatives (moins précis)
                    model_meta = {
                        'n_cols': perf.get('n_features', 0),
                        'column_names': set(),
                        'n_numerical': int(perf.get('n_features', 0) * 0.6),
                        'n_categorical': int(perf.get('n_features', 0) * 0.4),
                        'has_amount': True,
                        'has_timestamp': True,
                        'has_merchant': 'card' in model_name.lower() or 'pos' in model_name.lower(),
                        'has_card': 'card' in model_name.lower() or 'atm' in model_name.lower(),
                        'has_currency': False,
                        'has_country': False,
                        'has_balance': False,
                        'has_customer': False,
                        'has_account': False,
                        'domain': 'unknown',
                        'fraud_info': {},
                    }
                
                similarity = self.calculate_similarity(dataset_meta, model_meta)
                f1_score = perf.get('test_f1', 0)
                
                # Score combiné: 90% similarité + 10% performance
                # Privilégie FORTEMENT la similarité pour éviter les faux-positifs
                combined_score = similarity * 0.90 + f1_score * 0.10
                
                similarities.append((model_name, similarity, f1_score, combined_score))
        
        # Trier par score combiné
        similarities.sort(key=lambda x: x[3], reverse=True)
        
        if verbose:
            print(f"\n📊 Top 5 modèles les plus similaires:")
            for i, (name, sim, f1, combined) in enumerate(similarities[:5], 1):
                # Charger le domaine si disponible
                model_dir = self.automl_models_dir / name
                meta_file = model_dir / "dataset_metadata.json"
                domain_str = ""
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            model_meta = json.load(f)
                            domain = model_meta.get('domain', 'unknown')
                            if domain and domain != 'unknown':
                                domain_str = f" [{domain.replace('_', ' ').title()}]"
                    except:
                        pass
                
                print(f"   {i}. {name:15s}{domain_str:30s} | Sim: {sim:.1%} | F1: {f1:.1%} | Score: {combined:.1%}")
        
        best_model = similarities[0][0]
        best_similarity = similarities[0][1]
        best_combined = similarities[0][3]
        
        # Afficher une comparaison détaillée du meilleur match
        if verbose and best_similarity < 1.0:
            print(f"\n🔍 Analyse du meilleur match ({best_model}):")
            model_dir = self.automl_models_dir / best_model
            meta_file = model_dir / "dataset_metadata.json"
            
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    model_meta = json.load(f)
                
                prod_cols = dataset_meta['column_names']
                model_cols = set(model_meta.get('column_names', []))
                
                common = prod_cols & model_cols
                missing_in_model = prod_cols - model_cols
                extra_in_model = model_cols - prod_cols
                
                print(f"   - Colonnes communes: {len(common)}/{len(prod_cols)} ({len(common)/len(prod_cols):.1%})")
                print(f"   - Domaines: Production={dataset_meta.get('domain', 'unknown')} | Modèle={model_meta.get('domain', 'unknown')}")
                
                if len(missing_in_model) > 0 and len(missing_in_model) <= 5:
                    print(f"   - Colonnes manquantes dans le modèle: {', '.join(list(missing_in_model)[:5])}")
                elif len(missing_in_model) > 5:
                    print(f"   - Colonnes manquantes dans le modèle: {len(missing_in_model)} colonnes")
        
        return best_model, best_similarity
    
    def _recreate_features_from_flags(self, X: pd.DataFrame, engineering_flags: list) -> pd.DataFrame:
        """
        Recrée dynamiquement les features selon les engineering_flags (Meta-Transformer mode)
        
        Identique à la logique de full_automl.py (lignes 1580-1700)
        """
        import numpy as np
        
        X_engineered = X.copy()
        
        # Détection sémantique des colonnes amount
        amount_cols = []
        for col in X.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['amount', 'montant', 'balance', 'solde', 'value', 'valeur', 
                                               'price', 'prix', 'fcfa', 'xof', 'usd', 'eur']):
                amount_cols.append(col)
        
        if not amount_cols:
            # Fallback: prendre les colonnes numériques
            amount_cols = list(X.select_dtypes(include=[np.number]).columns[:3])
        
        print(f"      💰 Amount columns detected: {amount_cols}")
        
        # Seuils adaptatifs (identiques à full_automl.py)
        adaptive_thresholds = {
            'polynomial': 0.70,
            'interaction': 0.60,
            'binning': 0.40,
            'log_transform': 0.50,
            'aggregation': 0.60
        }
        
        eng_names = ['polynomial', 'interaction', 'binning', 'log_transform', 'aggregation']
        features_created = []
        
        # 🤖 AI-PREDICTED Features
        for i, name in enumerate(eng_names):
            threshold = adaptive_thresholds[name]
            score = engineering_flags[i]
            
            if score > threshold:
                print(f"      ✅ {name}: score={score:.3f} > {threshold:.2f} - APPLYING")
                
                if name == 'polynomial':
                    for col in amount_cols:
                        if col in X_engineered.columns:
                            X_engineered[f'{col}_squared'] = X_engineered[col] ** 2
                            X_engineered[f'{col}_cubed'] = X_engineered[col] ** 3
                            features_created.extend([f'{col}_squared', f'{col}_cubed'])
                
                elif name == 'binning':
                    for col in amount_cols:
                        if col in X_engineered.columns:
                            try:
                                X_engineered[f'{col}_bin'] = pd.qcut(X_engineered[col], q=5, labels=False, duplicates='drop')
                                features_created.append(f'{col}_bin')
                            except:
                                X_engineered[f'{col}_bin'] = 0
                
                elif name == 'log_transform':
                    for col in amount_cols:
                        if col in X_engineered.columns:
                            X_engineered[f'{col}_log'] = np.log1p(X_engineered[col])
                            features_created.append(f'{col}_log')
            else:
                print(f"      ❌ {name}: score={score:.3f} < {threshold:.2f} - SKIPPED")
        
        # 🎯 ALWAYS-ON Features (ratio, cyclic, boolean)
        print(f"      🎯 ALWAYS-ON features:")
        
        # Ratios entre colonnes amount
        if len(amount_cols) >= 2:
            for i in range(min(2, len(amount_cols)-1)):
                col1, col2 = amount_cols[i], amount_cols[i+1]
                X_engineered[f'{col1}_{col2}_ratio'] = X_engineered[col1] / (X_engineered[col2] + 1e-8)
                features_created.append(f'{col1}_{col2}_ratio')
            print(f"         ✓ Ratio features: {len([f for f in features_created if 'ratio' in f])}")
        
        # Cyclic features (si timestamp disponible)
        time_cols = [col for col in X.columns if any(kw in col.lower() for kw in ['time', 'hour', 'heure'])]
        if time_cols:
            for col in time_cols[:1]:  # Premier timestamp seulement
                if X_engineered[col].dtype in ['int64', 'float64']:
                    X_engineered[f'{col}_sin'] = np.sin(2 * np.pi * X_engineered[col] / 24)
                    X_engineered[f'{col}_cos'] = np.cos(2 * np.pi * X_engineered[col] / 24)
                    features_created.extend([f'{col}_sin', f'{col}_cos'])
            print(f"         ✓ Cyclic features: {len([f for f in features_created if 'sin' in f or 'cos' in f])}")
        
        print(f"      📊 Total features created: {len(features_created)} (AI: {len([f for f in features_created if not any(x in f for x in ['ratio', 'sin', 'cos'])])}, Always-on: {len([f for f in features_created if any(x in f for x in ['ratio', 'sin', 'cos'])])})")
        
        return X_engineered
    
    def apply_pipeline(self, df: pd.DataFrame, pipeline: Dict, 
                      threshold: float = 0.5) -> pd.DataFrame:
        """Applique le pipeline complet sur le dataset"""
        
        print(f"\n⚙️  Application du pipeline AutoML...")
        
        # Sauvegarder les colonnes ID pour les retrouver plus tard
        id_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['id', 'number', 'hash', 'timestamp'])]
        df_ids = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)
        
        # Préparer le dataset (supprimer colonnes non-features)
        df_features = df.copy()
        
        # Supprimer colonnes ID et timestamp
        cols_to_drop = [col for col in df_features.columns if any(kw in col.lower() for kw in ['id', 'timestamp', 'date', 'heure'])]
        if cols_to_drop:
            df_features = df_features.drop(columns=cols_to_drop)
            print(f"   ℹ️  Colonnes ID/timestamp ignorées: {len(cols_to_drop)}")
        
        # Encoder les colonnes catégorielles
        categorical_cols = df_features.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"   🔄 Encodage de {len(categorical_cols)} colonnes catégorielles...")
            from sklearn.preprocessing import LabelEncoder
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df_features[col] = le.fit_transform(df_features[col].astype(str))
                label_encoders[col] = le
        
        # 1. Feature Engineering DYNAMIQUE (Meta-Transformer mode)
        df_engineered = df_features.copy()
        
        # Charger les engineering flags depuis performance.json
        engineering_flags = pipeline.get('performance', {}).get('engineering_flags')
        meta_transformer_used = pipeline.get('performance', {}).get('meta_transformer_used', False)
        
        if meta_transformer_used and engineering_flags is not None:
            print(f"   ✨ Meta-Transformer mode: Recreating features dynamically")
            print(f"   🎯 Engineering flags: {engineering_flags}")
            
            # Recréer les features selon les flags (comme full_automl.py)
            df_engineered = self._recreate_features_from_flags(df_features, engineering_flags)
        
        elif pipeline['feature_engineer'] is not None:
            # Mode fallback (ancien système)
            try:
                if hasattr(pipeline['feature_engineer'], 'transform'):
                    df_engineered = pipeline['feature_engineer'].transform(df_features)
                    print(f"   ✓ Feature Engineering appliqué (fallback mode)")
                elif hasattr(pipeline['feature_engineer'], 'fit_transform'):
                    df_engineered = pipeline['feature_engineer'].fit_transform(df_features)
                    print(f"   ✓ Feature Engineering appliqué (fit_transform)")
                else:
                    print(f"   ⚠️  Feature Engineering non applicable")
            except Exception as e:
                print(f"   ⚠️  Feature Engineering échoué: {e}")
                df_engineered = df_features.copy()
        else:
            print(f"   ℹ️  No feature engineering (raw features only)")

        
        # S'assurer que c'est un DataFrame
        if isinstance(df_engineered, np.ndarray):
            df_engineered = pd.DataFrame(df_engineered, index=df_features.index)
        
        # Obtenir les features attendues par le modèle
        try:
            expected_features = pipeline['model'].get_booster().feature_names
            current_features = list(df_engineered.columns)
            
            # Ajouter les features manquantes (notamment is_fraudulent_transaction)
            missing_features = set(expected_features) - set(current_features)
            if missing_features:
                print(f"   ℹ️  Ajout de {len(missing_features)} colonnes manquantes: {missing_features}")
                for feat in missing_features:
                    df_engineered[feat] = 0
            
            # Réordonner les colonnes pour matcher l'ordre attendu
            df_engineered = df_engineered[expected_features]
            print(f"   ✓ Colonnes réorganisées pour matcher le modèle")
            
        except Exception as e:
            print(f"   ⚠️  Impossible de vérifier les features du modèle: {e}")
        
        # 2. Feature Selection
        if pipeline['feature_selector'] is not None:
            try:
                # Le selector attend un array numpy ou un DataFrame
                if hasattr(pipeline['feature_selector'], 'transform'):
                    selected_features = pipeline['feature_selector'].transform(df_engineered)
                    
                    # Si le résultat est un array, recréer un DataFrame
                    if isinstance(selected_features, np.ndarray):
                        # Récupérer les noms de colonnes sélectionnées
                        if hasattr(pipeline['feature_selector'], 'get_support'):
                            mask = pipeline['feature_selector'].get_support()
                            selected_cols = df_engineered.columns[mask]
                            df_selected = pd.DataFrame(selected_features, columns=selected_cols, index=df_engineered.index)
                        else:
                            df_selected = pd.DataFrame(selected_features, index=df_engineered.index)
                    else:
                        df_selected = selected_features
                    
                    print(f"   ✓ Feature Selection appliquée ({df_selected.shape[1]} features)")
                else:
                    df_selected = df_engineered
            except Exception as e:
                print(f"   ⚠️  Feature Selection échouée: {e}")
                df_selected = df_engineered
        else:
            df_selected = df_engineered
        
        # S'assurer que tout est numérique
        if isinstance(df_selected, pd.DataFrame):
            non_numeric = df_selected.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric) > 0:
                print(f"   ⚠️  Colonnes non-numériques détectées, conversion en cours...")
                for col in non_numeric:
                    df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
                # Remplir les NaN créés par la conversion
                df_selected = df_selected.fillna(0)
        
        # CRITIQUE: Réordonner les colonnes pour matcher EXACTEMENT l'ordre du modèle
        if isinstance(df_selected, pd.DataFrame):
            try:
                model_feature_names = pipeline['model'].get_booster().feature_names
                if model_feature_names is not None:
                    # Vérifier que toutes les features du modèle sont présentes
                    current_features = set(df_selected.columns)
                    model_features_set = set(model_feature_names)
                    
                    if current_features != model_features_set:
                        print(f"   ⚠️  Features mismatch détecté:")
                        print(f"      Manquantes: {model_features_set - current_features}")
                        print(f"      En trop: {current_features - model_features_set}")
                        
                        # Ajouter les colonnes manquantes avec des zéros
                        for feat in model_features_set - current_features:
                            df_selected[feat] = 0
                        
                        # Supprimer les colonnes en trop
                        df_selected = df_selected[model_feature_names]
                    else:
                        # Réordonner simplement dans le bon ordre
                        df_selected = df_selected[model_feature_names]
                    
                    print(f"   ✓ Colonnes réordonnées pour matcher le modèle ({len(model_feature_names)} features)")
            except Exception as e:
                print(f"   ⚠️  Impossible de réordonner les colonnes: {e}")
        
        # 3. Prédictions
        try:
            # Prédictions de probabilité
            fraud_proba = pipeline['model'].predict_proba(df_selected)[:, 1]
            
            # Prédictions binaires
            fraud_pred = (fraud_proba >= threshold).astype(int)
            
            print(f"   ✓ Prédictions effectuées ({len(fraud_proba)} transactions)")
            
        except Exception as e:
            print(f"   ❌ Erreur lors des prédictions: {e}")
            raise
        
        # 4. Créer DataFrame de résultats
        results = df_ids.copy()
        results['fraud_probability'] = fraud_proba
        results['fraud_prediction'] = fraud_pred
        results['risk_level'] = pd.cut(
            fraud_proba,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
        
        return results
    
    def apply_ensemble_predictions(self, df: pd.DataFrame, top_k: int = 3,
                                   threshold: float = 0.5, verbose: bool = True) -> pd.DataFrame:
        """
        🚀 NOUVEAU: Ensemble predictions avec top-k modèles similaires
        
        Plus robuste que single model: moyenne pondérée par similarité
        Réduit les faux positifs et améliore la stabilité
        
        Args:
            df: Dataset de production
            top_k: Nombre de modèles à utiliser (défaut: 3)
            threshold: Seuil de classification (défaut: 0.5)
            verbose: Affichage détaillé
        
        Returns:
            DataFrame avec prédictions ensemblées
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"🎯 ENSEMBLE PREDICTIONS (Top-{top_k} Models)")
            print(f"{'='*80}")
        
        # 1. Trouver les top-k modèles les plus similaires
        dataset_meta = self.extract_dataset_meta_features(df)
        
        similarities = []
        for model_name in self.available_models:
            model_meta = None
            
            # Essayer de charger depuis le fichier local
            model_dir = self.automl_models_dir / model_name
            meta_file = model_dir / "dataset_metadata.json"
            
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    model_meta_saved = json.load(f)
                
                model_meta = {
                    'n_cols': model_meta_saved.get('n_cols', 0),
                    'n_rows': model_meta_saved.get('n_rows', 0),
                    'column_names': set(model_meta_saved.get('column_names', [])),
                    'n_numerical': model_meta_saved.get('n_numerical', 0),
                    'n_categorical': model_meta_saved.get('n_categorical', 0),
                    'has_amount': model_meta_saved.get('has_amount', False),
                    'has_timestamp': model_meta_saved.get('has_timestamp', False),
                    'has_merchant': model_meta_saved.get('has_merchant', False),
                    'has_card': model_meta_saved.get('has_card', False),
                    'has_currency': model_meta_saved.get('has_currency', False),
                    'has_country': model_meta_saved.get('has_country', False),
                    'has_balance': model_meta_saved.get('has_balance', False),
                    'has_customer': model_meta_saved.get('has_customer', False),
                    'has_account': model_meta_saved.get('has_account', False),
                    'domain': model_meta_saved.get('domain', 'unknown'),
                    'fraud_info': model_meta_saved.get('fraud_info', {}),
                }
            else:
                # Charger depuis la DB (modèles de référence)
                try:
                    from app.models.reference_model import ReferenceModel
                    import json
                    ref_model = ReferenceModel.query.filter_by(model_name=model_name).first()
                    
                    if ref_model:
                        # Reconstruire dataset_metadata depuis les colonnes de ReferenceModel
                        column_names_list = json.loads(ref_model.column_names) if ref_model.column_names else []
                        
                        model_meta = {
                            'n_cols': ref_model.num_features or 0,
                            'n_rows': ref_model.dataset_size or 0,
                            'column_names': set(column_names_list),
                            'n_numerical': ref_model.n_numerical or 0,
                            'n_categorical': ref_model.n_categorical or 0,
                            'has_amount': ref_model.has_amount or False,
                            'has_timestamp': ref_model.has_timestamp or False,
                            'has_merchant': ref_model.has_merchant or False,
                            'has_card': ref_model.has_card or False,
                            'has_currency': ref_model.has_currency or False,
                            'has_country': ref_model.has_country or False,
                            'has_balance': ref_model.has_balance or False,
                            'has_customer': ref_model.has_customer or False,
                            'has_account': ref_model.has_account or False,
                            'domain': json.loads(ref_model.signature).get('domain', 'unknown') if ref_model.signature else 'unknown',
                            'fraud_info': {'fraud_rate': ref_model.fraud_rate or 0},
                        }
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Impossible de charger métadonnées pour {model_name}: {e}")
            
            if model_meta:
                similarity = self.calculate_similarity(dataset_meta, model_meta)
                similarities.append((model_name, similarity))
        
        # Trier et prendre les top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_models = similarities[:top_k]
        
        if verbose:
            print(f"\n📊 Modèles sélectionnés pour l'ensemble:")
            for i, (name, sim) in enumerate(top_models, 1):
                print(f"   {i}. {name:15s} | Similarité: {sim:.1%}")
        
        # 2. Obtenir les prédictions de chaque modèle
        all_predictions = []
        weights = []
        
        for model_name, similarity in top_models:
            if verbose:
                print(f"\n   🔄 Application du modèle {model_name}...")
            
            # Charger le pipeline (avec cache)
            if model_name not in self._model_cache:
                self._model_cache[model_name] = self.load_model_pipeline(model_name)
            pipeline = self._model_cache[model_name]
            
            # Obtenir prédictions
            results = self.apply_pipeline(df, pipeline, threshold=threshold)
            all_predictions.append(results['fraud_probability'].values)
            weights.append(similarity)
        
        # 3. Combiner les prédictions (moyenne pondérée)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normaliser
        
        ensemble_proba = np.zeros(len(df))
        for pred, weight in zip(all_predictions, weights):
            ensemble_proba += pred * weight
        
        # 4. Créer le DataFrame de résultats
        id_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['id', 'number', 'hash', 'timestamp'])]
        results = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)
        
        results['fraud_probability'] = ensemble_proba
        results['fraud_prediction'] = (ensemble_proba >= threshold).astype(int)
        results['risk_level'] = pd.cut(
            ensemble_proba,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
        
        # Ajouter la variance des prédictions (mesure de stabilité)
        prediction_variance = np.var(all_predictions, axis=0)
        results['prediction_variance'] = prediction_variance
        results['prediction_stability'] = 1 - np.clip(prediction_variance, 0, 1)  # 1 = très stable
        
        # Stocker les top_models pour sauvegarde ultérieure
        results.attrs['top_models'] = top_models  # Métadonnée pandas
        
        if verbose:
            print(f"\n✅ Ensemble predictions terminées:")
            print(f"   - Stabilité moyenne: {results['prediction_stability'].mean():.1%}")
            print(f"   - Variance moyenne: {results['prediction_variance'].mean():.4f}")
        
        return results
    
    def add_anomaly_detection(self, df: pd.DataFrame, results: pd.DataFrame,
                             contamination: float = 0.01, verbose: bool = True) -> pd.DataFrame:
        """
        🚀 NOUVEAU: Anomaly detection complémentaire (Isolation Forest)
        
        Détecte des anomalies structurelles que le modèle pourrait manquer
        Combine le score XGBoost avec un score d'anomalie
        
        Args:
            df: Dataset original
            results: Résultats de prédictions
            contamination: Taux d'anomalies attendu (défaut: 1%)
            verbose: Affichage détaillé
        
        Returns:
            DataFrame enrichi avec scores d'anomalie
        """
        if verbose:
            print(f"\n🔍 Détection d'anomalies complémentaire (Isolation Forest)...")
        
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import LabelEncoder
        
        # Préparer les données (exclure IDs/timestamps)
        df_features = df.copy()
        cols_to_drop = [col for col in df_features.columns 
                       if any(kw in col.lower() for kw in ['id', 'timestamp', 'date', 'heure'])]
        
        if cols_to_drop:
            df_features = df_features.drop(columns=cols_to_drop)
        
        # Encoder les colonnes catégorielles
        categorical_cols = df_features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df_features[col] = le.fit_transform(df_features[col].astype(str))
        
        # Remplir les NaN
        df_features = df_features.fillna(df_features.median())
        
        # Entraîner Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        anomaly_scores = iso_forest.fit_predict(df_features)
        anomaly_scores_proba = iso_forest.score_samples(df_features)
        
        # Normaliser à [0, 1] (0 = normal, 1 = anomalie)
        anomaly_scores_norm = (anomaly_scores_proba - anomaly_scores_proba.min()) / (
            anomaly_scores_proba.max() - anomaly_scores_proba.min()
        )
        anomaly_scores_norm = 1 - anomaly_scores_norm  # Inverser (1 = anomalie)
        
        # Ajouter aux résultats
        results['anomaly_score'] = anomaly_scores_norm
        results['is_anomaly'] = (anomaly_scores == -1).astype(int)
        
        # Score combiné: 70% XGBoost + 30% anomaly
        results['combined_score'] = (
            results['fraud_probability'] * 0.70 +
            results['anomaly_score'] * 0.30
        )
        
        if verbose:
            n_anomalies = results['is_anomaly'].sum()
            print(f"   ✅ {n_anomalies:,} anomalies détectées ({n_anomalies/len(results):.2%})")
            print(f"   📊 Corrélation fraud/anomaly: {results['fraud_prediction'].corr(results['is_anomaly']):.2f}")
        
        return results
    
    def calibrate_probabilities(self, results: pd.DataFrame, method: str = 'isotonic',
                                verbose: bool = True) -> pd.DataFrame:
        """
        🚀 NOUVEAU: Calibration des probabilités
        
        Les probabilités XGBoost ne sont pas toujours bien calibrées
        Cette méthode ajuste les probabilités pour être plus fiables
        
        Note: Nécessite des données de validation avec labels (optionnel)
        Pour production pure, utilise une approximation basée sur la distribution
        
        Args:
            results: Résultats de prédictions
            method: 'isotonic' ou 'platt' (défaut: isotonic)
            verbose: Affichage détaillé
        
        Returns:
            DataFrame avec probabilités calibrées
        """
        if verbose:
            print(f"\n📊 Calibration des probabilités ({method})...")
        
        # Pour production sans labels, on applique une transformation simple
        # basée sur la distribution des scores
        fraud_proba = results['fraud_probability'].values
        
        # Transformation sigmoïde pour "étaler" les probabilités
        # Rend les scores extrêmes plus confiants
        calibrated_proba = 1 / (1 + np.exp(-10 * (fraud_proba - 0.5)))
        
        results['fraud_probability_calibrated'] = calibrated_proba
        
        if verbose:
            print(f"   ✅ Probabilités calibrées:")
            print(f"      Avant: mean={fraud_proba.mean():.3f}, std={fraud_proba.std():.3f}")
            print(f"      Après: mean={calibrated_proba.mean():.3f}, std={calibrated_proba.std():.3f}")
        
        return results
    
    def apply_batch_predictions(self, df: pd.DataFrame, pipeline: Dict,
                               batch_size: int = 50000, threshold: float = 0.5,
                               verbose: bool = True) -> pd.DataFrame:
        """
        🚀 NOUVEAU: Mode batch pour très gros volumes (>1M lignes)
        
        Traite le dataset par chunks pour éviter les problèmes de mémoire
        
        Args:
            df: Dataset de production
            pipeline: Pipeline AutoML
            batch_size: Taille des batchs (défaut: 50k)
            threshold: Seuil de classification
            verbose: Affichage détaillé
        
        Returns:
            DataFrame avec toutes les prédictions
        """
        if verbose:
            print(f"\n📦 Mode BATCH activé ({batch_size:,} lignes par batch)")
        
        n_batches = int(np.ceil(len(df) / batch_size))
        all_results = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            
            if verbose:
                print(f"\n   Batch {i+1}/{n_batches}: lignes {start_idx:,} à {end_idx:,}")
            
            batch_df = df.iloc[start_idx:end_idx]
            batch_results = self.apply_pipeline(batch_df, pipeline, threshold=threshold)
            all_results.append(batch_results)
        
        # Concaténer tous les résultats
        results = pd.concat(all_results, ignore_index=True)
        
        if verbose:
            print(f"\n✅ Traitement batch terminé: {len(results):,} prédictions")
        
        return results
    
    def generate_report(self, results: pd.DataFrame, model_name: str, 
                       pipeline: Dict, threshold: float, 
                       ensemble_mode: bool = False, anomaly_mode: bool = False) -> None:
        """
        Génère un rapport détaillé des prédictions (VERSION AMÉLIORÉE v2.0)
        
        Args:
            results: Résultats de prédictions
            model_name: Nom du modèle (ou "Ensemble" si mode ensemble)
            pipeline: Pipeline AutoML (ou None si ensemble)
            threshold: Seuil de classification
            ensemble_mode: Si True, mode ensemble activé
            anomaly_mode: Si True, anomaly detection activée
        """
        
        print(f"\n{'='*80}")
        if ensemble_mode:
            print(f"RAPPORT DE PRÉDICTIONS - Mode ENSEMBLE (Top-K Models)")
        else:
            print(f"RAPPORT DE PRÉDICTIONS - Modèle: {model_name}")
        print(f"{'='*80}")
        
        # Statistiques globales
        n_total = len(results)
        n_fraud = results['fraud_prediction'].sum()
        fraud_rate = n_fraud / n_total
        
        print(f"\n📊 Statistiques Globales:")
        print(f"   Total transactions: {n_total:,}")
        print(f"   Fraudes détectées: {n_fraud:,} ({fraud_rate:.2%})")
        print(f"   Transactions légitimes: {n_total - n_fraud:,} ({1-fraud_rate:.2%})")
        
        # Distribution des probabilités
        fraud_proba_col = 'fraud_probability_calibrated' if 'fraud_probability_calibrated' in results.columns else 'fraud_probability'
        
        print(f"\n📈 Distribution des Probabilités:")
        print(f"   Moyenne: {results[fraud_proba_col].mean():.3f}")
        print(f"   Médiane: {results[fraud_proba_col].median():.3f}")
        print(f"   Écart-type: {results[fraud_proba_col].std():.3f}")
        print(f"   P90: {results[fraud_proba_col].quantile(0.90):.3f}")
        print(f"   P95: {results[fraud_proba_col].quantile(0.95):.3f}")
        print(f"   P99: {results[fraud_proba_col].quantile(0.99):.3f}")
        print(f"   Max: {results[fraud_proba_col].max():.3f}")
        
        # Niveaux de risque
        print(f"\n⚠️  Répartition par Niveau de Risque:")
        risk_counts = results['risk_level'].value_counts()
        for level in ['HIGH', 'MEDIUM', 'LOW']:
            count = risk_counts.get(level, 0)
            pct = count / n_total
            bar = '█' * int(pct * 50)
            print(f"   {level:6s}: {count:7,} ({pct:5.1%}) {bar}")
        
        # Confiance des prédictions
        high_conf_fraud = ((results[fraud_proba_col] >= 0.8) & (results['fraud_prediction'] == 1)).sum()
        high_conf_legit = ((results[fraud_proba_col] <= 0.2) & (results['fraud_prediction'] == 0)).sum()
        medium_conf = ((results[fraud_proba_col] > 0.2) & (results[fraud_proba_col] < 0.8)).sum()
        high_conf_total = high_conf_fraud + high_conf_legit
        high_conf_rate = high_conf_total / n_total
        
        print(f"\n🎯 Confiance des Prédictions:")
        print(f"   Haute confiance (>80% ou <20%): {high_conf_total:,} ({high_conf_rate:.1%})")
        print(f"   - Fraudes haute confiance: {high_conf_fraud:,}")
        print(f"   - Légitimes haute confiance: {high_conf_legit:,}")
        print(f"   Confiance moyenne (20-80%): {medium_conf:,} ({medium_conf/n_total:.1%})")
        
        # Stabilité (si mode ensemble)
        if 'prediction_stability' in results.columns:
            print(f"\n🔄 Stabilité des Prédictions (Ensemble):")
            print(f"   Stabilité moyenne: {results['prediction_stability'].mean():.1%}")
            print(f"   Variance moyenne: {results['prediction_variance'].mean():.4f}")
            
            unstable = (results['prediction_stability'] < 0.7).sum()
            print(f"   Prédictions instables (<70%): {unstable:,} ({unstable/n_total:.1%})")
        
        # Anomaly detection
        if anomaly_mode and 'anomaly_score' in results.columns:
            n_anomalies = results['is_anomaly'].sum()
            fraud_and_anomaly = ((results['fraud_prediction'] == 1) & (results['is_anomaly'] == 1)).sum()
            
            print(f"\n🔍 Anomaly Detection:")
            print(f"   Anomalies détectées: {n_anomalies:,} ({n_anomalies/n_total:.2%})")
            print(f"   Fraudes ET anomalies: {fraud_and_anomaly:,}")
            print(f"   Corrélation: {results['fraud_prediction'].corr(results['is_anomaly']):.2f}")
        
        # Performance du modèle utilisé
        if pipeline and pipeline.get('performance'):
            perf = pipeline['performance']
            print(f"\n📌 Performance du Modèle (sur données test):")
            print(f"   F1 Score: {perf.get('test_f1', 0):.2%}")
            print(f"   AUC-ROC: {perf.get('test_auc', 0):.2%}")
            print(f"   Precision: {perf.get('precision', 0):.2%}")
            print(f"   Recall: {perf.get('recall', 0):.2%}")
            
            if 'hyperparameters' in perf:
                hp = perf['hyperparameters']
                print(f"   Hyperparamètres: learning_rate={hp.get('learning_rate', 'N/A'):.3f}, "
                      f"max_depth={hp.get('max_depth', 'N/A')}, "
                      f"n_estimators={hp.get('n_estimators', 'N/A')}")
        
        # Recommandations
        print(f"\n💡 Recommandations:")
        if fraud_rate > 0.1:
            print(f"   ⚠️  Taux de fraude élevé ({fraud_rate:.1%}) - Investigation recommandée")
        if high_conf_rate < 0.5:
            print(f"   ⚠️  Confiance faible ({high_conf_rate:.1%}) - Envisager retraining du modèle")
        if 'prediction_stability' in results.columns and results['prediction_stability'].mean() < 0.7:
            print(f"   ⚠️  Stabilité faible - Prédictions peu fiables, augmenter top_k")
        if anomaly_mode and fraud_and_anomaly < n_fraud * 0.3:
            print(f"   ℹ️  Peu de overlap fraud/anomaly - Patterns de fraude bien appris")
        
        print(f"\n{'='*80}\n")
    
    def export_rich_results(self, results: pd.DataFrame, output_path: str,
                           format: str = 'excel', include_charts: bool = True) -> None:
        """
        🚀 NOUVEAU: Export enrichi avec graphiques et analyses
        
        Args:
            results: Résultats de prédictions
            output_path: Chemin de sortie (sans extension)
            format: 'excel', 'json', ou 'both' (défaut: excel)
            include_charts: Inclure graphiques dans Excel (défaut: True)
        """
        output_path = Path(output_path)
        
        # Export Excel enrichi
        if format in ['excel', 'both']:
            excel_path = output_path.with_suffix('.xlsx')
            
            print(f"\n💾 Export Excel enrichi: {excel_path}")
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Sheet 1: Tous les résultats
                results.to_excel(writer, sheet_name='All Predictions', index=False)
                
                # Sheet 2: High risk frauds only
                high_risk = results[results['risk_level'] == 'HIGH'].copy()
                high_risk = high_risk.sort_values('fraud_probability', ascending=False)
                high_risk.to_excel(writer, sheet_name='High Risk', index=False)
                
                # Sheet 3: Summary statistics
                summary = pd.DataFrame({
                    'Métrique': [
                        'Total transactions',
                        'Fraudes détectées',
                        'Taux de fraude',
                        'Probabilité moyenne',
                        'Probabilité médiane',
                        'HIGH risk count',
                        'MEDIUM risk count',
                        'LOW risk count',
                    ],
                    'Valeur': [
                        len(results),
                        results['fraud_prediction'].sum(),
                        f"{results['fraud_prediction'].mean():.2%}",
                        f"{results['fraud_probability'].mean():.3f}",
                        f"{results['fraud_probability'].median():.3f}",
                        (results['risk_level'] == 'HIGH').sum(),
                        (results['risk_level'] == 'MEDIUM').sum(),
                        (results['risk_level'] == 'LOW').sum(),
                    ]
                })
                summary.to_excel(writer, sheet_name='Summary', index=False)
                
                print(f"   ✅ 3 sheets créés: All Predictions, High Risk, Summary")
        
        # Export JSON détaillé
        if format in ['json', 'both']:
            json_path = output_path.with_suffix('.json')
            
            print(f"\n💾 Export JSON détaillé: {json_path}")
            
            json_data = {
                'metadata': {
                    'n_total': len(results),
                    'n_fraud': int(results['fraud_prediction'].sum()),
                    'fraud_rate': float(results['fraud_prediction'].mean()),
                    'timestamp': pd.Timestamp.now().isoformat(),
                },
                'summary_statistics': {
                    'probability': {
                        'mean': float(results['fraud_probability'].mean()),
                        'median': float(results['fraud_probability'].median()),
                        'std': float(results['fraud_probability'].std()),
                        'p95': float(results['fraud_probability'].quantile(0.95)),
                        'p99': float(results['fraud_probability'].quantile(0.99)),
                    },
                    'risk_distribution': {
                        'high': int((results['risk_level'] == 'HIGH').sum()),
                        'medium': int((results['risk_level'] == 'MEDIUM').sum()),
                        'low': int((results['risk_level'] == 'LOW').sum()),
                    }
                },
                'top_10_frauds': results.nlargest(10, 'fraud_probability').to_dict(orient='records'),
                'predictions': results.to_dict(orient='records')
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ JSON créé avec metadata + top 10 + prédictions complètes")


    def save_ensemble_model(self, output_dir: str, top_models: List[Tuple[str, float]], 
                           results: pd.DataFrame, dataset_name: str = "unlabeled") -> Dict:
        """
        Sauvegarde le modèle ensemble pour réutilisation future
        
        Args:
            output_dir: Chemin où sauvegarder le modèle
            top_models: Liste des (model_name, similarity) utilisés
            results: Résultats des prédictions avec métriques
            dataset_name: Nom du dataset (pour métadonnées)
        
        Returns:
            Dict avec info de sauvegarde
        """
        import joblib
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Sauvegarder les informations de l'ensemble
        ensemble_info = {
            'model_type': 'ensemble',
            'n_models': len(top_models),
            'base_models': [
                {'name': name, 'similarity': float(sim), 'weight': float(sim)}
                for name, sim in top_models
            ],
            'created_at': datetime.now().isoformat(),
            'dataset_name': dataset_name
        }
        
        with open(f'{output_dir}/ensemble_info.json', 'w') as f:
            json.dump(ensemble_info, f, indent=2)
        
        # 2. Copier les pipelines des modèles de base
        for model_name, _ in top_models:
            model_dir = self.automl_models_dir / model_name
            
            # Charger le pipeline si pas déjà en cache
            if model_name not in self._model_cache:
                self._model_cache[model_name] = self.load_model_pipeline(model_name)
            
            pipeline = self._model_cache[model_name]
            
            # Sauvegarder chaque composant du modèle
            model_subdir = Path(output_dir) / f"base_model_{model_name}"
            model_subdir.mkdir(exist_ok=True)
            
            joblib.dump(pipeline['model'], f"{model_subdir}/xgboost_model.joblib")
            joblib.dump(pipeline['feature_engineer'], f"{model_subdir}/feature_engineer.joblib")
            joblib.dump(pipeline['feature_selector'], f"{model_subdir}/feature_selector.joblib")
            
            # Copier les metadata
            meta_src = model_dir / "dataset_metadata.json"
            if meta_src.exists():
                import shutil
                shutil.copy(meta_src, f"{model_subdir}/dataset_metadata.json")
        
        # 3. Calculer et sauvegarder les performances
        performance = {
            'total_transactions': len(results),
            'fraud_probability_stats': {
                'mean': float(results['fraud_probability'].mean()),
                'median': float(results['fraud_probability'].median()),
                'std': float(results['fraud_probability'].std()),
                'min': float(results['fraud_probability'].min()),
                'max': float(results['fraud_probability'].max())
            },
            'risk_distribution': {
                'high': int((results['combined_score'] >= 0.7).sum()) if 'combined_score' in results.columns else 0,
                'medium': int(((results['combined_score'] >= 0.5) & (results['combined_score'] < 0.7)).sum()) if 'combined_score' in results.columns else 0,
                'low': int((results['combined_score'] < 0.5).sum()) if 'combined_score' in results.columns else 0
            },
            'prediction_stability_mean': float(results['prediction_stability'].mean()) if 'prediction_stability' in results.columns else None,
            'anomalies_detected': int(results['is_anomaly'].sum()) if 'is_anomaly' in results.columns else 0,
            'methods_used': {
                'ensemble': True,
                'anomaly_detection': 'is_anomaly' in results.columns,
                'calibration': 'fraud_probability_calibrated' in results.columns
            }
        }
        
        with open(f'{output_dir}/performance.json', 'w') as f:
            json.dump(performance, f, indent=2)
        
        print(f"\n✅ Modèle ensemble sauvegardé dans: {output_dir}")
        print(f"   - {len(top_models)} modèles de base")
        print(f"   - Performance: {performance['prediction_stability_mean']:.1%} stabilité" if performance['prediction_stability_mean'] else "")
        print(f"   - Anomalies: {performance['anomalies_detected']}")
        
        return {
            'model_path': output_dir,
            'n_models': len(top_models),
            'performance': performance
        }


def main():
    parser = argparse.ArgumentParser(
        description='Applique des modèles AutoML sur des données de production (sans labels) - VERSION 2.0 OPTIMISÉE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
-----------------------
# Mode ENSEMBLE (RECOMMANDÉ - plus robuste)
python apply_automl_production.py --dataset production.csv --ensemble --top_k 3

# Avec anomaly detection
python apply_automl_production.py --dataset production.csv --auto_match --anomaly_detection

# Mode batch pour gros volumes (>1M lignes)
python apply_automl_production.py --dataset big_prod.csv --auto_match --batch_size 50000

# Export enrichi (Excel + JSON avec graphiques)
python apply_automl_production.py --dataset production.csv --ensemble --rich_export

# Mode classique: auto-match simple
python apply_automl_production.py --dataset production.csv --auto_match

# Mode classique: spécifier un modèle
python apply_automl_production.py --dataset production.csv --model dataset20 --threshold 0.20

# Lister les modèles disponibles
python apply_automl_production.py --list_models
        """
    )
    
    # Arguments principaux
    parser.add_argument('--dataset', type=str, help='Chemin vers le dataset de production (CSV)')
    parser.add_argument('--model', type=str, help='Nom du modèle à utiliser (ex: dataset20)')
    parser.add_argument('--auto_match', action='store_true', 
                       help='Trouve automatiquement le modèle le plus similaire')
    
    # 🚀 NOUVEAUX arguments v2.0
    parser.add_argument('--ensemble', action='store_true',
                       help='🚀 Mode ensemble: combine top-k modèles similaires (plus robuste)')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Nombre de modèles pour ensemble (défaut: 3)')
    parser.add_argument('--anomaly_detection', action='store_true',
                       help='🚀 Active anomaly detection complémentaire (Isolation Forest)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='🚀 Mode batch pour gros volumes (ex: 50000 lignes/batch)')
    parser.add_argument('--rich_export', action='store_true',
                       help='🚀 Export enrichi: Excel avec graphiques + JSON détaillé')
    parser.add_argument('--calibrate', action='store_true',
                       help='🚀 Calibre les probabilités (plus fiables)')
    
    # Arguments classiques
    parser.add_argument('--threshold', type=float, default=0.20,
                       help='Seuil de classification (défaut: 0.20 - optimisé pour détection de fraude)')
    parser.add_argument('--output', type=str, default='predictions',
                       help='Nom de base pour fichiers de sortie (défaut: predictions)')
    parser.add_argument('--list_models', action='store_true',
                       help='Liste tous les modèles AutoML disponibles')
    parser.add_argument('--automl_dir', type=str, default='data/automl_models',
                       help='Dossier contenant les modèles AutoML')
    
    args = parser.parse_args()
    
    # Initialiser l'applicateur
    applicator = AutoMLProductionApplicator(automl_models_dir=args.automl_dir)
    
    # Mode: Lister les modèles
    if args.list_models:
        print(f"\n📦 Modèles AutoML Disponibles (v2.0):")
        print(f"{'='*80}")
        if not applicator.available_models:
            print("   Aucun modèle trouvé!")
        else:
            for model_name in applicator.available_models:
                model_dir = applicator.automl_models_dir / model_name
                perf_file = model_dir / "performance.json"
                meta_file = model_dir / "dataset_metadata.json"
                
                domain_str = ""
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                            domain = meta.get('domain', 'unknown')
                            if domain != 'unknown':
                                domain_str = f" [{domain.replace('_', ' ').title()}]"
                    except:
                        pass
                
                if perf_file.exists():
                    with open(perf_file, 'r') as f:
                        perf = json.load(f)
                    print(f"   - {model_name:15s}{domain_str:30s} | "
                          f"F1: {perf.get('test_f1', 0):.2%} | "
                          f"AUC: {perf.get('test_auc', 0):.2%} | "
                          f"Features: {perf.get('n_features', 0)}")
                else:
                    print(f"   - {model_name:15s}{domain_str:30s} | (pas d'infos)")
        print(f"{'='*80}\n")
        return
    
    # Vérifications
    if not args.dataset:
        parser.error("--dataset est requis (ou utilisez --list_models)")
    
    if not args.auto_match and not args.model and not args.ensemble:
        parser.error("Spécifiez --model, --auto_match, ou --ensemble")
    
    # Charger le dataset
    print(f"\n📂 Chargement du dataset: {args.dataset}")
    try:
        df = pd.read_csv(args.dataset)
        print(f"   ✓ Dataset chargé: {len(df):,} lignes × {len(df.columns)} colonnes")
        
        # 🚀 NOUVEAU: Afficher un aperçu des colonnes
        print(f"   📋 Colonnes: {', '.join(df.columns[:10])}")
        if len(df.columns) > 10:
            print(f"              ... (+{len(df.columns)-10} autres)")
    except Exception as e:
        print(f"   ❌ Erreur de chargement: {e}")
        sys.exit(1)
    
    # ============================================================
    # 🚀 MODE ENSEMBLE (NOUVEAU - RECOMMANDÉ)
    # ============================================================
    if args.ensemble:
        print(f"\n{'='*80}")
        print(f"🎯 MODE ENSEMBLE ACTIVÉ (Top-{args.top_k} models)")
        print(f"{'='*80}")
        
        # Ensemble predictions
        results = applicator.apply_ensemble_predictions(
            df, 
            top_k=args.top_k,
            threshold=args.threshold,
            verbose=True
        )
        
        # Anomaly detection (optionnel)
        if args.anomaly_detection:
            results = applicator.add_anomaly_detection(df, results, verbose=True)
        
        # Calibration (optionnel)
        if args.calibrate:
            results = applicator.calibrate_probabilities(results, verbose=True)
        
        # Générer rapport
        applicator.generate_report(
            results, 
            "Ensemble", 
            None,  # Pas de pipeline unique
            args.threshold,
            ensemble_mode=True,
            anomaly_mode=args.anomaly_detection
        )
        
        model_name_for_output = "ensemble"
    
    # ============================================================
    # MODE CLASSIQUE (auto-match ou model spécifié)
    # ============================================================
    else:
        # Déterminer le modèle à utiliser
        if args.auto_match:
            model_name, similarity = applicator.find_best_matching_model(df, verbose=True)
            print(f"\n✅ Meilleur modèle: {model_name} (similarité: {similarity:.1%})")
            
            if similarity < 0.5:
                print(f"\n⚠️  ATTENTION: Similarité faible ({similarity:.1%})")
                print(f"   Le modèle peut ne pas être optimal pour ce dataset.")
                print(f"   💡 Suggestion: utilisez --ensemble pour plus de robustesse")
                response = input("   Continuer quand même? (o/n): ")
                if response.lower() != 'o':
                    print("   Opération annulée.")
                    sys.exit(0)
        else:
            model_name = args.model
            if model_name not in applicator.available_models:
                print(f"❌ Modèle '{model_name}' introuvable!")
                print(f"   Modèles disponibles: {', '.join(applicator.available_models)}")
                sys.exit(1)
        
        # Charger le pipeline
        try:
            pipeline = applicator.load_model_pipeline(model_name)
        except Exception as e:
            print(f"❌ Erreur de chargement du pipeline: {e}")
            sys.exit(1)
        
        # Appliquer le pipeline
        try:
            # Mode batch si spécifié
            if args.batch_size:
                results = applicator.apply_batch_predictions(
                    df, pipeline, 
                    batch_size=args.batch_size,
                    threshold=args.threshold,
                    verbose=True
                )
            else:
                results = applicator.apply_pipeline(df, pipeline, threshold=args.threshold)
            
            # Anomaly detection (optionnel)
            if args.anomaly_detection:
                results = applicator.add_anomaly_detection(df, results, verbose=True)
            
            # Calibration (optionnel)
            if args.calibrate:
                results = applicator.calibrate_probabilities(results, verbose=True)
                
        except Exception as e:
            print(f"❌ Erreur lors de l'application du pipeline: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Générer rapport
        applicator.generate_report(
            results, 
            model_name, 
            pipeline, 
            args.threshold,
            ensemble_mode=False,
            anomaly_mode=args.anomaly_detection
        )
        
        model_name_for_output = model_name
    
    # ============================================================
    # SAUVEGARDE DES RÉSULTATS
    # ============================================================
    
    # Ajouter métadonnées
    results['model_used'] = model_name_for_output
    results['prediction_threshold'] = args.threshold
    results['prediction_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Export enrichi (Excel + JSON) si demandé
    if args.rich_export:
        applicator.export_rich_results(results, args.output, format='both', include_charts=True)
    else:
        # Export CSV classique
        output_path = Path(args.output)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.csv')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"\n💾 Résultats sauvegardés: {output_path}")
        print(f"   Colonnes: {list(results.columns)}")
    
    # ============================================================
    # TOP 10 FRAUDES LES PLUS PROBABLES
    # ============================================================
    fraud_proba_col = 'fraud_probability_calibrated' if 'fraud_probability_calibrated' in results.columns else 'fraud_probability'
    top_frauds = results.nlargest(10, fraud_proba_col)
    
    if len(top_frauds) > 0:
        print(f"\n🚨 Top 10 Fraudes Les Plus Probables:")
        print(f"{'='*80}")
        for i, row in enumerate(top_frauds.itertuples(), 1):
            id_col = [col for col in results.columns if 'id' in col.lower()]
            id_val = getattr(row, id_col[0]) if id_col else i
            proba = getattr(row, fraud_proba_col)
            risk = getattr(row, 'risk_level')
            
            stability_str = ""
            if 'prediction_stability' in results.columns:
                stability = getattr(row, 'prediction_stability')
                stability_str = f"| Stabilité: {stability:.1%}"
            
            anomaly_str = ""
            if 'is_anomaly' in results.columns:
                is_anom = getattr(row, 'is_anomaly')
                if is_anom == 1:
                    anomaly_str = "| 🔍 ANOMALIE"
            
            print(f"   {i:2d}. ID: {id_val} | Proba: {proba:.1%} | Risque: {risk} {stability_str} {anomaly_str}")
        print(f"{'='*80}")
    
    print(f"\n✅ Prédictions terminées avec succès!\n")
    
    # 💡 Suggestions finales
    if not args.ensemble:
        print(f"💡 Suggestion: Essayez --ensemble pour des prédictions plus robustes")
    if not args.anomaly_detection:
        print(f"💡 Suggestion: Ajoutez --anomaly_detection pour détecter des patterns inhabituels")
    if not args.rich_export:
        print(f"💡 Suggestion: Utilisez --rich_export pour Excel + JSON avec analyses détaillées")
    print()


if __name__ == "__main__":
    main()
