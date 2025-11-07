"""
Test S3 upload avec les bonnes variables d'environnement
"""
import os
import boto3
from pathlib import Path
from dotenv import load_dotenv
import tempfile

# Load .env
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

def get_s3_bucket():
    """Même fonction que dans l'API"""
    return os.environ.get('S3_MODEL_BUCKET') or os.environ.get('AWS_S3_BUCKET', 'fraud-detection-ml-models')

print("🧪 Test S3 Upload\n")
print(f"S3_MODEL_BUCKET: {os.environ.get('S3_MODEL_BUCKET')}")
print(f"AWS_S3_BUCKET: {os.environ.get('AWS_S3_BUCKET')}")
print(f"get_s3_bucket() returns: {get_s3_bucket()}\n")

# Create temporary test file
test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
test_file.write("test_column_1,test_column_2\n")
test_file.write("value1,value2\n")
test_file.close()
test_path = Path(test_file.name)

print(f"📁 Test file created: {test_path}")
print(f"   Size: {test_path.stat().st_size} bytes")

# Test upload
try:
    s3_bucket = get_s3_bucket()
    s3_key = f"test_uploads/test_{Path(test_path).name}"
    
    print(f"\n📤 Uploading to S3...")
    print(f"   Bucket: {s3_bucket}")
    print(f"   Key: {s3_key}")
    
    s3_client = boto3.client('s3')
    s3_client.upload_file(str(test_path), s3_bucket, s3_key)
    
    print(f"✅ Upload successful!")
    print(f"   Location: s3://{s3_bucket}/{s3_key}")
    
    # Test download
    print(f"\n📥 Testing download...")
    download_path = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    download_path.close()
    
    s3_client.download_file(s3_bucket, s3_key, download_path.name)
    print(f"✅ Download successful!")
    
    # Cleanup
    test_path.unlink()
    Path(download_path.name).unlink()
    
    # Delete from S3
    s3_client.delete_object(Bucket=s3_bucket, Key=s3_key)
    print(f"🗑️  Cleaned up test files")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Cleanup
    if test_path.exists():
        test_path.unlink()
