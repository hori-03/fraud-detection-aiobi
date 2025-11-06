"""
Configuration de l'application Flask

Gère les configurations pour différents environnements:
- Development: Base de données SQLite locale
- Production: PostgreSQL Railway
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Répertoire de base de l'application
basedir = Path(__file__).parent.parent


class Config:
    """Configuration de base"""
    
    # Clé secrète pour les sessions et CSRF
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Base de données
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        f'sqlite:///{basedir / "instance" / "aml_dev.db"}'
    
    # Fix pour Railway PostgreSQL URL (postgres:// -> postgresql://)
    if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith('postgres://'):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace('postgres://', 'postgresql://', 1)
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Configuration uploads
    UPLOAD_FOLDER = basedir / 'uploads'
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_UPLOAD_SIZE', 500 * 1024 * 1024))  # 500MB par défaut
    ALLOWED_EXTENSIONS = {'csv', 'json'}
    
    # Configuration modèles AutoML
    MODELS_FOLDER = basedir / 'models' / 'xgboost_models'
    
    # Répertoire racine du projet (pour accéder à automl_transformer)
    PROJECT_ROOT = basedir.parent
    AUTOML_MODELS_DIR = PROJECT_ROOT / 'data' / 'automl_models'  # Tes 40 modèles existants
    DATASETS_DIR = PROJECT_ROOT / 'data' / 'datasets'
    
    # 🚀 PRODUCTION: Configuration stockage cloud
    STORAGE_TYPE = os.environ.get('STORAGE_TYPE', 'local')  # 'local', 's3', 'gcs', 'azure'
    
    # AWS S3 Configuration
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_DEFAULT_REGION = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    S3_MODEL_BUCKET = os.environ.get('S3_MODEL_BUCKET', 'fraud-detection-models')
    
    # Google Cloud Storage Configuration (optionnel)
    GCS_PROJECT_ID = os.environ.get('GCS_PROJECT_ID')
    GCS_BUCKET = os.environ.get('GCS_BUCKET')
    
    # Cache pour modèles téléchargés
    MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', '/tmp/model_cache')
    
    # Configuration Google OAuth2 (optionnel pour l'instant)
    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
    GOOGLE_REDIRECT_URI = os.environ.get('GOOGLE_REDIRECT_URI') or 'http://localhost:5000/auth/callback'
    
    # Logs
    LOG_FOLDER = basedir / 'logs'
    LOG_FILE = LOG_FOLDER / 'app.log'
    
    # Timezone
    TIMEZONE = 'UTC'


class DevelopmentConfig(Config):
    """Configuration développement"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Configuration production (Railway)"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Configuration tests"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


# Dictionnaire des configurations
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
