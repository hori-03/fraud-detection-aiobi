"""
Force re-upload all models to S3 (even if storage_type='s3' in DB)
Used after retraining models locally with new engineering_flags
"""
import sys
import boto3
from pathlib import Path
import logging
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app import create_app, db
from app.models.reference_model import ReferenceModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def force_reupload_models(bucket_name: str = 'fraud-detection-ml-models'):
    """Force re-upload all models to S3"""
    
    app = create_app()
    
    with app.app_context():
        # Get all models from database
        models = ReferenceModel.query.all()
        logger.info(f"Found {len(models)} models in database")
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Get base directory
        base_dir = Path(app.config['AUTOML_MODELS_DIR'])
        logger.info(f"Base directory: {base_dir}")
        
        success_count = 0
        failed_count = 0
        
        for model in models:
            try:
                model_dir = base_dir / model.model_name
                
                if not model_dir.exists():
                    logger.warning(f"❌ {model.model_name}: Directory not found")
                    failed_count += 1
                    continue
                
                logger.info(f"\n📦 Uploading {model.model_name}...")
                
                # Get all files in model directory
                files_uploaded = 0
                for file_path in model_dir.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(base_dir)
                        # Add automl_models/ prefix
                        s3_key = f"automl_models/{str(relative_path).replace(chr(92), '/')}"
                        
                        s3_client.upload_file(
                            str(file_path),
                            bucket_name,
                            s3_key
                        )
                        files_uploaded += 1
                
                logger.info(f"   ✅ {files_uploaded} files uploaded")
                success_count += 1
                
            except Exception as e:
                logger.error(f"❌ {model.model_name}: {str(e)}")
                failed_count += 1
        
        logger.info("\n" + "="*80)
        logger.info("FORCE RE-UPLOAD SUMMARY")
        logger.info("="*80)
        logger.info(f"   Total models: {len(models)}")
        logger.info(f"   ✅ Uploaded: {success_count}")
        logger.info(f"   ❌ Failed: {failed_count}")
        logger.info("")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default='fraud-detection-ml-models')
    args = parser.parse_args()
    
    logger.info("🚀 FORCE RE-UPLOAD TO S3")
    logger.info(f"📂 Bucket: {args.bucket}")
    logger.info("⚠️  This will overwrite existing files on S3\n")
    
    time.sleep(2)  # Give user time to cancel
    
    force_reupload_models(args.bucket)
    
    logger.info("✅ Force re-upload complete!")
