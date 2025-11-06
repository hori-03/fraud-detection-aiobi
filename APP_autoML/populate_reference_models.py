"""
Script de peuplement de la table reference_models

Ce script parcourt le dossier data/automl_models/ et ajoute tous les modèles
pré-entraînés dans la base de données PostgreSQL.

Usage:
    python populate_reference_models.py

⚠️ IMPORTANT: Exécuter depuis le dossier APP_autoML/
"""

import os
import sys
import json
from pathlib import Path
import joblib
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app import create_app, db
from app.models.reference_model import ReferenceModel


def load_model_metadata(model_dir: Path) -> dict:
    """
    Charge les métadonnées d'un modèle depuis son dossier
    
    Returns:
        dict avec toutes les infos du modèle, ou None si le modèle XGBoost n'existe pas
    """
    metadata = {
        'model_name': model_dir.name,
        'model_path': str(model_dir)
    }
    
    # ⚠️ VÉRIFICATION CRITIQUE : Le modèle XGBoost doit exister
    xgboost_file = model_dir / 'xgboost_model.joblib'
    if not xgboost_file.exists():
        print(f"   ❌ SKIPPED: xgboost_model.joblib not found")
        return None
    
    # Charger le modèle XGBoost pour vérifier et extraire num_features
    try:
        model = joblib.load(xgboost_file)
        print(f"   ✅ XGBoost model loaded successfully")
        
        # Extraire num_features du modèle avec plusieurs stratégies
        def _extract_num_features(m):
            """Try different heuristics to infer number of input features from the saved object."""
            if m is None:
                return None

            # sklearn Pipeline -> inspect last estimator
            try:
                from sklearn.pipeline import Pipeline
                if isinstance(m, Pipeline):
                    return _extract_num_features(m.steps[-1][1])
            except Exception:
                pass

            # sklearn GridSearchCV / wrapper with best_estimator_
            if hasattr(m, 'best_estimator_'):
                return _extract_num_features(m.best_estimator_)

            # dict-like containers
            if isinstance(m, dict):
                for v in m.values():
                    nf = _extract_num_features(v)
                    if nf:
                        return nf

            # xgboost sklearn API: get_booster()
            if hasattr(m, 'get_booster'):
                try:
                    booster = m.get_booster()
                    if hasattr(booster, 'feature_names') and booster.feature_names:
                        return len(booster.feature_names)
                except Exception:
                    pass

            # xgboost Booster saved directly
            if hasattr(m, 'feature_names') and getattr(m, 'feature_names'):
                try:
                    return len(m.feature_names)
                except Exception:
                    pass

            # Standard sklearn attrs
            if hasattr(m, 'n_features_in_'):
                try:
                    return int(m.n_features_in_)
                except Exception:
                    pass
            if hasattr(m, 'feature_names_in_'):
                try:
                    return len(m.feature_names_in_)
                except Exception:
                    pass
            if hasattr(m, 'feature_types_'):
                try:
                    return len(m.feature_types_)
                except Exception:
                    pass

            # Estimator coefficients (linear models)
            if hasattr(m, 'coef_'):
                try:
                    coef = m.coef_
                    if hasattr(coef, 'shape'):
                        return int(coef.shape[-1])
                except Exception:
                    pass

            return None

        nf = _extract_num_features(model)
        if nf is not None:
            metadata['num_features'] = nf
        else:
            # We'll try to infer from dataset metadata or feature_engineer later
            print(f"   ⚠️  Could not infer num_features from model for {model_dir.name}")
    except Exception as e:
        print(f"   ❌ Cannot load XGBoost model: {e}")
        return None
    
    # Charger dataset_metadata.json COMPLET pour garantir compatibilité apply_automl
    dataset_metadata_file = model_dir / 'dataset_metadata.json'
    if dataset_metadata_file.exists():
        with open(dataset_metadata_file, 'r', encoding='utf-8') as f:
            dataset_meta = json.load(f)
            
            # === STRUCTURE DE BASE ===
            if 'column_names' in dataset_meta:
                metadata['column_names'] = json.dumps(dataset_meta['column_names'])
            
            if 'numerical_cols' in dataset_meta:
                metadata['numerical_cols'] = json.dumps(dataset_meta['numerical_cols'])
                metadata['n_numerical'] = len(dataset_meta['numerical_cols'])
            
            if 'categorical_cols' in dataset_meta:
                metadata['categorical_cols'] = json.dumps(dataset_meta['categorical_cols'])
                metadata['n_categorical'] = len(dataset_meta['categorical_cols'])
            
            # === DATASET SIZE ===
            if 'fraud_info' in dataset_meta and 'n_samples' in dataset_meta['fraud_info']:
                metadata['dataset_size'] = dataset_meta['fraud_info']['n_samples']
            elif 'n_rows' in dataset_meta:
                metadata['dataset_size'] = dataset_meta['n_rows']
            
            # === FRAUD RATE ===
            if 'fraud_info' in dataset_meta and 'fraud_rate' in dataset_meta['fraud_info']:
                metadata['fraud_rate'] = dataset_meta['fraud_info']['fraud_rate']
            elif 'signature' in dataset_meta and 'fraud_rate_bucket' in dataset_meta['signature']:
                metadata['fraud_rate'] = dataset_meta['signature']['fraud_rate_bucket']
            
            # === FEATURES DISPONIBLES (pour matching sémantique) ===
            metadata['has_amount'] = dataset_meta.get('has_amount', False)
            metadata['has_timestamp'] = dataset_meta.get('has_timestamp', False)
            metadata['has_merchant'] = dataset_meta.get('has_merchant', False)
            metadata['has_card'] = dataset_meta.get('has_card', False)
            metadata['has_currency'] = dataset_meta.get('has_currency', False)
            metadata['has_country'] = dataset_meta.get('has_country', False)
            metadata['has_balance'] = dataset_meta.get('has_balance', False)
            metadata['has_customer'] = dataset_meta.get('has_customer', False)
            metadata['has_account'] = dataset_meta.get('has_account', False)
            
            # === FEATURES TEMPORELLES ===
            if 'temporal_features' in dataset_meta:
                metadata['temporal_features'] = json.dumps(dataset_meta['temporal_features'])
            
            # === PATTERNS DES MONTANTS ===
            if 'amount_patterns' in dataset_meta:
                metadata['amount_patterns'] = json.dumps(dataset_meta['amount_patterns'])
            
            # === PATTERNS CATÉGORIELS ===
            if 'categorical_patterns' in dataset_meta:
                metadata['categorical_patterns'] = json.dumps(dataset_meta['categorical_patterns'])
            
            # === SIGNATURE (pour comparaison rapide) ===
            if 'signature' in dataset_meta:
                metadata['signature'] = json.dumps(dataset_meta['signature'])
            
            # === DOMAIN ===
            if 'domain' in dataset_meta:
                metadata['domain'] = dataset_meta['domain']
            
            # Calculer num_engineered_features
            if 'num_features' in metadata:
                metadata['num_engineered_features'] = metadata['num_features']
    
    # Charger performance.json pour les métriques
    performance_file = model_dir / 'performance.json'
    if performance_file.exists():
        with open(performance_file, 'r', encoding='utf-8') as f:
            perf = json.load(f)
            
            # Ajouter les métriques directement dans metadata
            if 'precision' in perf:
                metadata['precision'] = perf['precision']
            if 'recall' in perf:
                metadata['recall'] = perf['recall']
            if 'test_f1' in perf:
                metadata['f1_score'] = perf['test_f1']
            if 'test_auc' in perf:
                metadata['roc_auc'] = perf['test_auc']
            if 'accuracy' in perf:
                metadata['accuracy'] = perf['accuracy']
            
            # Extraire hyperparameters
            if 'hyperparameters' in perf:
                metadata['hyperparameters'] = json.dumps(perf['hyperparameters'])
            
            # Extraire n_features si pas déjà récupéré
            if 'num_features' not in metadata and 'n_features' in perf:
                metadata['num_features'] = perf['n_features']
    
    # Charger feature_importance si disponible
    fi_file = model_dir.parent.parent / 'Feature_importance' / f'{model_dir.name}_production_feature_importance.json'
    if fi_file.exists():
        with open(fi_file, 'r', encoding='utf-8') as f:
            fi_data = json.load(f)
            if 'feature_importance' in fi_data:
                metadata['feature_importance'] = json.dumps(fi_data['feature_importance'])
    
    # Charger feature_engineer si disponible (meilleure source pour column names)
    engineer_file = model_dir / 'feature_engineer.joblib'
    if engineer_file.exists():
        try:
            engineer = joblib.load(engineer_file)
            cols = None
            if hasattr(engineer, 'original_columns'):
                cols = engineer.original_columns
            elif hasattr(engineer, 'feature_names_in_'):
                cols = list(engineer.feature_names_in_)
            elif hasattr(engineer, 'feature_names_generated'):
                cols = getattr(engineer, 'feature_names_generated')
            if cols:
                metadata['column_names'] = json.dumps(cols)
                # Preferer le nombre de colonnes provenant de l'engineer
                metadata['num_features'] = len(cols)
        except Exception as e:
            print(f"   ⚠️  Cannot load engineer: {e}")
    
    # Détecter le domaine basé sur le nom du dataset
    dataset_name = model_dir.name.lower()  # Convertir en minuscules
    dataset_num_str = dataset_name.replace('dataset', '')  # Retirer 'dataset'
    try:
        dataset_num = int(dataset_num_str)
    except ValueError:
        print(f"   ⚠️  Cannot extract dataset number from {model_dir.name}")
        metadata['domain'] = 'unknown'
        return metadata
    
    if dataset_num <= 10:
        metadata['domain'] = 'banking'
    elif dataset_num <= 20:
        metadata['domain'] = 'e-commerce'
    elif dataset_num <= 30:
        metadata['domain'] = 'telecom'
    else:
        metadata['domain'] = 'insurance'
    
    return metadata


def populate_reference_models(automl_models_dir: str = None):
    """
    Peuple la table reference_models avec tous les modèles disponibles
    """
    if automl_models_dir is None:
        automl_models_dir = Path(__file__).parent.parent / 'data' / 'automl_models'
    else:
        automl_models_dir = Path(automl_models_dir)
    
    if not automl_models_dir.exists():
        print(f"❌ Dossier {automl_models_dir} introuvable!")
        return
    
    print(f"\n🔍 Scanning {automl_models_dir}...")
    
    models_found = []
    for model_dir in sorted(automl_models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        
        # Vérifier que le modèle existe
        model_file = model_dir / 'xgboost_model.joblib'
        if not model_file.exists():
            print(f"   ⚠️  Skipping {model_dir.name}: no xgboost_model.joblib")
            continue
        
        models_found.append(model_dir)
    
    print(f"✅ Found {len(models_found)} models\n")
    
    if len(models_found) == 0:
        print("❌ No models found!")
        return
    
    # Créer l'application Flask
    app = create_app()
    
    with app.app_context():
        # Vérifier si la table existe
        try:
            db.create_all()
            print("✅ Database tables created/verified\n")
        except Exception as e:
            print(f"❌ Database error: {e}")
            return
        
        # Supprimer les anciens modèles (optionnel - décommenter si nécessaire)
        # ReferenceModel.query.delete()
        # db.session.commit()
        # print("🗑️  Cleared existing reference models\n")
        
        added_count = 0
        updated_count = 0
        skipped_count = 0
        
        for model_dir in models_found:
            model_name = model_dir.name
            
            print(f"📦 Processing {model_name}...")
            
            # Vérifier si le modèle existe déjà
            existing = ReferenceModel.query.filter_by(model_name=model_name).first()
            
            # Charger les métadonnées
            try:
                metadata = load_model_metadata(model_dir)
                if metadata is None:
                    # Le modèle XGBoost n'existe pas ou est invalide
                    skipped_count += 1
                    continue
            except Exception as e:
                print(f"   ❌ Error loading metadata: {e}")
                skipped_count += 1
                continue
            
            if existing:
                # Mettre à jour le modèle existant
                print(f"   🔄 Updating existing model...")
                
                for key, value in metadata.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                
                # Forcer storage_type='local' si pas déjà défini
                if not existing.storage_type:
                    existing.storage_type = 'local'
                    existing.s3_bucket = None
                    existing.s3_prefix = None
                
                existing.is_active = True
                updated_count += 1
            else:
                # Créer nouveau modèle
                print(f"   ✨ Creating new model...")
                
                new_model = ReferenceModel(
                    model_name=metadata.get('model_name'),
                    model_path=metadata.get('model_path'),
                    dataset_size=metadata.get('dataset_size'),
                    num_features=metadata.get('num_features'),
                    num_engineered_features=metadata.get('num_engineered_features'),
                    fraud_rate=metadata.get('fraud_rate'),
                    column_names=metadata.get('column_names'),
                    column_types=metadata.get('column_types'),
                    
                    # === MÉTADONNÉES COMPLÈTES ===
                    numerical_cols=metadata.get('numerical_cols'),
                    categorical_cols=metadata.get('categorical_cols'),
                    n_numerical=metadata.get('n_numerical'),
                    n_categorical=metadata.get('n_categorical'),
                    
                    # Features disponibles
                    has_amount=metadata.get('has_amount'),
                    has_timestamp=metadata.get('has_timestamp'),
                    has_merchant=metadata.get('has_merchant'),
                    has_card=metadata.get('has_card'),
                    has_currency=metadata.get('has_currency'),
                    has_country=metadata.get('has_country'),
                    has_balance=metadata.get('has_balance'),
                    has_customer=metadata.get('has_customer'),
                    has_account=metadata.get('has_account'),
                    
                    # Patterns et signature
                    temporal_features=metadata.get('temporal_features'),
                    amount_patterns=metadata.get('amount_patterns'),
                    categorical_patterns=metadata.get('categorical_patterns'),
                    signature=metadata.get('signature'),
                    
                    # Métriques et hyperparamètres
                    accuracy=metadata.get('accuracy'),
                    precision=metadata.get('precision'),
                    recall=metadata.get('recall'),
                    f1_score=metadata.get('f1_score'),
                    roc_auc=metadata.get('roc_auc'),
                    hyperparameters=metadata.get('hyperparameters'),
                    feature_importance=metadata.get('feature_importance'),
                    engineering_methods=metadata.get('engineering_methods'),
                    
                    # Métadonnées de gestion
                    domain=metadata.get('domain', 'general'),
                    data_quality=metadata.get('data_quality', 'high'),
                    imbalance_ratio=metadata.get('fraud_rate', 0.1),
                    is_active=True,
                    version='1.0',
                    description=f'Pre-trained model from {model_name}',
                    tags=f"{metadata.get('domain', 'general')},automl,production",
                    
                    # 🚀 Stockage (local par défaut, S3 après migration)
                    storage_type='local',
                    s3_bucket=None,
                    s3_prefix=None
                )
                
                db.session.add(new_model)
                added_count += 1
            
            # Afficher les infos chargées
            print(f"   📊 Dataset size: {metadata.get('dataset_size', 'N/A')}")
            print(f"   🔢 Features: {metadata.get('num_features', 'N/A')}")
            
            # Afficher les métriques
            metrics_json = metadata.get('metrics', '{}')
            try:
                metrics = json.loads(metrics_json) if isinstance(metrics_json, str) else metrics_json
                if metrics.get('roc_auc'):
                    print(f"   🎯 ROC-AUC: {metrics['roc_auc']:.4f}")
                if metrics.get('f1'):
                    print(f"   📈 F1-Score: {metrics['f1']:.4f}")
            except:
                pass
            
            print(f"   🏷️  Domain: {metadata.get('domain', 'unknown')}")
            print(f"   ✅ Done\n")
        
        # Commit tous les changements
        try:
            db.session.commit()
            print("\n" + "="*60)
            print(f"🎉 COMPLETED!")
            print(f"   ✨ Added: {added_count} models")
            print(f"   🔄 Updated: {updated_count} models")
            print(f"   ⏭️  Skipped: {skipped_count} models")
            print(f"   📊 Total in DB: {ReferenceModel.query.count()} models")
            print("="*60 + "\n")
        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Database commit failed: {e}")


def show_reference_models():
    """Affiche tous les modèles de référence dans la base de données"""
    app = create_app()
    
    with app.app_context():
        models = ReferenceModel.query.order_by(ReferenceModel.model_name).all()
        
        if not models:
            print("❌ No reference models found in database")
            return
        
        print("\n" + "="*80)
        print(f"📊 REFERENCE MODELS IN DATABASE ({len(models)} total)")
        print("="*80 + "\n")
        
        for model in models:
            status = "✅ ACTIVE" if model.is_active else "❌ INACTIVE"
            print(f"{status} | {model.model_name:15} | Domain: {model.domain:12} | "
                  f"ROC-AUC: {model.roc_auc or 0:.4f} | Used: {model.usage_count:3} times")
        
        print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Populate reference_models table')
    parser.add_argument('--dir', type=str, help='Path to automl_models directory')
    parser.add_argument('--show', action='store_true', help='Show existing models instead of populating')
    
    args = parser.parse_args()
    
    if args.show:
        show_reference_models()
    else:
        populate_reference_models(args.dir)
