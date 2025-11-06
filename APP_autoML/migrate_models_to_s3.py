"""
Script de migration: Local → S3 (Production)

Upload tous les modèles locaux (data/automl_models/) vers AWS S3
et met à jour la table reference_models avec les URLs S3.

Usage:
------
# 1. Configurer les credentials AWS
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# 2. Créer un bucket S3
aws s3 mb s3://fraud-detection-models

# 3. Lancer la migration
python migrate_models_to_s3.py --bucket fraud-detection-models

# 4. Vérifier
python migrate_models_to_s3.py --bucket fraud-detection-models --verify
"""

import os
import sys
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path (current dir = APP_autoML)
sys.path.insert(0, str(Path(__file__).parent))

from app import create_app, db
from app.models.reference_model import ReferenceModel
from app.services.model_storage import ModelStorageService


def migrate_models_to_s3(bucket_name: str, base_dir: str = 'data/automl_models', dry_run: bool = False):
    """
    Migre tous les modèles locaux vers S3
    
    Args:
        bucket_name: Nom du bucket S3
        base_dir: Dossier local des modèles
        dry_run: Si True, simule sans uploader
    """
    app = create_app()
    storage = ModelStorageService()
    
    with app.app_context():
        # Récupérer tous les modèles de référence
        models = ReferenceModel.query.filter_by(storage_type='local').all()
        
        logger.info(f"Found {len(models)} models to migrate")
        logger.info(f"Target bucket: s3://{bucket_name}")
        
        if dry_run:
            logger.info("DRY RUN MODE - No files will be uploaded")
        
        migrated_count = 0
        failed_count = 0
        
        for model in models:
            logger.info(f"\n{'='*80}")
            logger.info(f"Migrating: {model.model_name}")
            logger.info(f"  Local path: {model.model_path}")
            
            model_dir = Path(model.model_path)
            if not model_dir.exists():
                logger.error(f"  ❌ Model directory not found: {model_dir}")
                failed_count += 1
                continue
            
            # Définir le prefix S3
            s3_prefix = f"automl_models/{model.model_name}/"
            
            logger.info(f"  S3 path: s3://{bucket_name}/{s3_prefix}")
            
            if not dry_run:
                try:
                    # Upload vers S3
                    success = storage.upload_model_to_s3(
                        model_dir=model_dir,
                        bucket=bucket_name,
                        prefix=s3_prefix
                    )
                    
                    if success:
                        # Mettre à jour la BDD
                        model.storage_type = 's3'
                        model.s3_bucket = bucket_name
                        model.s3_prefix = s3_prefix
                        db.session.commit()
                        
                        logger.info(f"  ✅ Migration successful")
                        migrated_count += 1
                    else:
                        logger.error(f"  ❌ Upload failed")
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"  ❌ Error: {e}")
                    failed_count += 1
                    db.session.rollback()
            else:
                logger.info(f"  ✓ Would upload to s3://{bucket_name}/{s3_prefix}")
                migrated_count += 1
        
        # Résumé
        logger.info(f"\n{'='*80}")
        logger.info(f"MIGRATION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"  Total models: {len(models)}")
        logger.info(f"  ✅ Migrated: {migrated_count}")
        logger.info(f"  ❌ Failed: {failed_count}")
        
        if dry_run:
            logger.info(f"\n⚠️  DRY RUN - No actual changes made")
        else:
            logger.info(f"\n✅ Migration complete!")


def verify_s3_models(bucket_name: str):
    """
    Vérifie que tous les modèles S3 sont accessibles
    """
    app = create_app()
    storage = ModelStorageService()
    
    with app.app_context():
        models = ReferenceModel.query.filter_by(storage_type='s3').all()
        
        logger.info(f"Verifying {len(models)} S3 models...")
        
        success_count = 0
        failed_count = 0
        
        for model in models:
            logger.info(f"\n{'='*80}")
            logger.info(f"Verifying: {model.model_name}")
            logger.info(f"  S3 path: s3://{model.s3_bucket}/{model.s3_prefix}")
            
            try:
                # Tenter de charger le modèle
                pipeline = storage.load_model_pipeline(model)
                
                # Vérifier les composants
                has_xgboost = pipeline.get('xgboost_model') is not None
                has_engineer = pipeline.get('feature_engineer') is not None
                has_selector = pipeline.get('feature_selector') is not None
                
                logger.info(f"  ✓ XGBoost model: {has_xgboost}")
                logger.info(f"  ✓ Feature engineer: {has_engineer}")
                logger.info(f"  ✓ Feature selector: {has_selector}")
                
                if has_xgboost:
                    logger.info(f"  ✅ Model accessible")
                    success_count += 1
                else:
                    logger.error(f"  ❌ XGBoost model missing")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"  ❌ Error loading model: {e}")
                failed_count += 1
        
        # Résumé
        logger.info(f"\n{'='*80}")
        logger.info(f"VERIFICATION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"  Total models: {len(models)}")
        logger.info(f"  ✅ Accessible: {success_count}")
        logger.info(f"  ❌ Failed: {failed_count}")
        
        if failed_count == 0:
            logger.info(f"\n✅ All models verified successfully!")
        else:
            logger.error(f"\n⚠️  {failed_count} models failed verification")


def estimate_s3_costs(base_dir: str = 'data/automl_models'):
    """
    Estime les coûts de stockage S3
    """
    import os
    
    total_size = 0
    model_count = 0
    
    base_path = Path(base_dir)
    
    for model_dir in base_path.iterdir():
        if model_dir.is_dir():
            model_count += 1
            for file in model_dir.rglob('*'):
                if file.is_file():
                    total_size += file.stat().st_size
    
    total_size_mb = total_size / (1024 * 1024)
    total_size_gb = total_size / (1024 * 1024 * 1024)
    
    # Coûts S3 (US East - Ohio)
    storage_cost_per_gb = 0.023  # $/GB/month
    request_cost_per_1000 = 0.0004  # PUT requests
    transfer_cost_per_gb = 0.09  # Data transfer OUT (first 10 TB)
    
    monthly_storage_cost = total_size_gb * storage_cost_per_gb
    upload_cost = (model_count * 5) * (request_cost_per_1000 / 1000)  # ~5 files/model
    
    logger.info(f"\n{'='*80}")
    logger.info(f"S3 COST ESTIMATION")
    logger.info(f"{'='*80}")
    logger.info(f"  Total models: {model_count}")
    logger.info(f"  Total size: {total_size_mb:.2f} MB ({total_size_gb:.3f} GB)")
    logger.info(f"\n  Costs (monthly):")
    logger.info(f"    Storage: ${monthly_storage_cost:.4f}/month")
    logger.info(f"    Upload (one-time): ${upload_cost:.4f}")
    logger.info(f"    Yearly storage: ${monthly_storage_cost * 12:.2f}/year")
    logger.info(f"\n  💡 Tip: Use S3 Intelligent-Tiering for cost optimization")
    logger.info(f"{'='*80}")


def rollback_to_local():
    """
    Rollback: Repasse tous les modèles en mode local
    """
    app = create_app()
    
    with app.app_context():
        models = ReferenceModel.query.filter_by(storage_type='s3').all()
        
        logger.info(f"Rolling back {len(models)} models to local storage...")
        
        for model in models:
            # Remettre le chemin local
            local_path = f"data/automl_models/{model.model_name}"
            model.model_path = local_path
            model.storage_type = 'local'
            model.s3_bucket = None
            model.s3_prefix = None
        
        db.session.commit()
        logger.info(f"✅ Rollback complete - all models set to local storage")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate AutoML models from local to S3 (production)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
# Estimate costs
python migrate_models_to_s3.py --estimate

# Dry run (simulate)
python migrate_models_to_s3.py --bucket fraud-models --dry-run

# Migrate to S3
python migrate_models_to_s3.py --bucket fraud-models

# Verify S3 models
python migrate_models_to_s3.py --bucket fraud-models --verify

# Rollback to local
python migrate_models_to_s3.py --rollback
        """
    )
    
    parser.add_argument('--bucket', type=str, help='S3 bucket name')
    parser.add_argument('--base-dir', type=str, default='data/automl_models',
                       help='Local models directory')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate without uploading')
    parser.add_argument('--verify', action='store_true',
                       help='Verify S3 models accessibility')
    parser.add_argument('--estimate', action='store_true',
                       help='Estimate S3 storage costs')
    parser.add_argument('--rollback', action='store_true',
                       help='Rollback to local storage')
    
    args = parser.parse_args()
    
    if args.estimate:
        estimate_s3_costs(args.base_dir)
        return
    
    if args.rollback:
        response = input("⚠️  This will reset all models to local storage. Continue? (yes/no): ")
        if response.lower() == 'yes':
            rollback_to_local()
        else:
            logger.info("Rollback cancelled")
        return
    
    if args.verify:
        if not args.bucket:
            parser.error("--bucket required for verification")
        verify_s3_models(args.bucket)
        return
    
    if not args.bucket:
        parser.error("--bucket required (or use --estimate)")
    
    # Migration
    migrate_models_to_s3(args.bucket, args.base_dir, args.dry_run)


if __name__ == '__main__':
    main()
