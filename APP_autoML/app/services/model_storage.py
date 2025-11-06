"""
Service de stockage des modèles - Support multi-cloud

Supporte:
- Local (développement)
- AWS S3 (production recommandé)
- Google Cloud Storage
- Azure Blob Storage
- Railway Volumes

Architecture:
1. PostgreSQL stocke les métadonnées + URLs
2. Cloud storage stocke les fichiers .joblib
3. Cache local pour optimisation
"""

import os
import joblib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelStorageService:
    """
    Service unifié pour charger des modèles depuis différentes sources
    
    Usage:
        storage = ModelStorageService()
        pipeline = storage.load_model_pipeline(reference_model)
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: Dossier pour cache local (défaut: /tmp/model_cache)
        """
        self.cache_dir = Path(cache_dir or tempfile.gettempdir()) / 'model_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Vérifier la disponibilité des clients cloud
        self.s3_available = self._check_s3_available()
        self.gcs_available = self._check_gcs_available()
        
        logger.info(f"ModelStorageService initialized (cache: {self.cache_dir})")
        logger.info(f"  S3 available: {self.s3_available}")
        logger.info(f"  GCS available: {self.gcs_available}")
    
    def _check_s3_available(self) -> bool:
        """Vérifie si boto3 (AWS S3) est disponible"""
        try:
            import boto3
            return True
        except ImportError:
            return False
    
    def _check_gcs_available(self) -> bool:
        """Vérifie si google-cloud-storage est disponible"""
        try:
            from google.cloud import storage
            return True
        except ImportError:
            return False
    
    def load_model_pipeline(self, reference_model) -> Dict:
        """
        Charge le pipeline complet (XGBoost + Engineer + Selector)
        
        Args:
            reference_model: Instance de ReferenceModel (from DB)
        
        Returns:
            Dict avec {
                'xgboost_model': model,
                'feature_engineer': engineer,
                'feature_selector': selector,
                'performance': perf_dict,
                'metadata': meta_dict
            }
        """
        storage_type = reference_model.storage_type or 'local'
        
        logger.info(f"Loading model {reference_model.model_name} (storage: {storage_type})")
        
        # Dispatcher selon le type de stockage
        if storage_type == 'local':
            return self._load_from_local(reference_model)
        elif storage_type == 's3':
            return self._load_from_s3(reference_model)
        elif storage_type == 'gcs':
            return self._load_from_gcs(reference_model)
        elif storage_type == 'azure':
            return self._load_from_azure(reference_model)
        else:
            raise ValueError(f"Stockage type non supporté: {storage_type}")
    
    def _load_from_local(self, reference_model) -> Dict:
        """Charge depuis le système de fichiers local"""
        model_dir = Path(reference_model.model_path)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Modèle introuvable: {model_dir}")
        
        logger.info(f"  Loading from local: {model_dir}")
        
        pipeline = {}
        
        # XGBoost model (REQUIS)
        xgboost_file = model_dir / 'xgboost_model.joblib'
        if not xgboost_file.exists():
            raise FileNotFoundError(f"XGBoost model introuvable: {xgboost_file}")
        pipeline['xgboost_model'] = joblib.load(xgboost_file)
        
        # Feature Engineer (optionnel)
        engineer_file = model_dir / 'feature_engineer.joblib'
        if engineer_file.exists():
            pipeline['feature_engineer'] = joblib.load(engineer_file)
        else:
            pipeline['feature_engineer'] = None
        
        # Feature Selector (optionnel)
        selector_file = model_dir / 'feature_selector.joblib'
        if selector_file.exists():
            pipeline['feature_selector'] = joblib.load(selector_file)
        else:
            pipeline['feature_selector'] = None
        
        # Performance metrics
        perf_file = model_dir / 'performance.json'
        if perf_file.exists():
            import json
            with open(perf_file, 'r') as f:
                pipeline['performance'] = json.load(f)
        else:
            pipeline['performance'] = {}
        
        # Metadata
        meta_file = model_dir / 'dataset_metadata.json'
        if meta_file.exists():
            import json
            with open(meta_file, 'r') as f:
                pipeline['metadata'] = json.load(f)
        else:
            pipeline['metadata'] = {}
        
        logger.info(f"  ✓ Model loaded from local filesystem")
        return pipeline
    
    def _load_from_s3(self, reference_model) -> Dict:
        """
        Charge depuis AWS S3
        
        Nécessite:
        - pip install boto3
        - Variables d'environnement: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        """
        if not self.s3_available:
            raise ImportError("boto3 non installé. Installez avec: pip install boto3")
        
        import boto3
        import json
        
        s3_client = boto3.client('s3')
        bucket = reference_model.s3_bucket
        prefix = reference_model.s3_prefix
        
        logger.info(f"  Loading from S3: s3://{bucket}/{prefix}")
        
        # Vérifier le cache local
        cache_model_dir = self.cache_dir / reference_model.model_name
        if cache_model_dir.exists():
            logger.info(f"  ✓ Using cached version: {cache_model_dir}")
            reference_model.model_path = str(cache_model_dir)
            return self._load_from_local(reference_model)
        
        # Télécharger depuis S3
        cache_model_dir.mkdir(parents=True, exist_ok=True)
        
        files_to_download = [
            'xgboost_model.joblib',
            'feature_engineer.joblib',
            'feature_selector.joblib',
            'performance.json',
            'dataset_metadata.json'
        ]
        
        for filename in files_to_download:
            s3_key = f"{prefix}{filename}"
            local_file = cache_model_dir / filename
            
            try:
                logger.info(f"    Downloading {filename}...")
                s3_client.download_file(bucket, s3_key, str(local_file))
                logger.info(f"    ✓ {filename} downloaded")
            except Exception as e:
                if filename == 'xgboost_model.joblib':
                    raise  # XGBoost model est REQUIS
                else:
                    logger.warning(f"    ⚠️  {filename} not found (optional): {e}")
        
        # Charger depuis le cache
        reference_model.model_path = str(cache_model_dir)
        logger.info(f"  ✓ Model downloaded and cached")
        return self._load_from_local(reference_model)
    
    def _load_from_gcs(self, reference_model) -> Dict:
        """
        Charge depuis Google Cloud Storage
        
        Nécessite:
        - pip install google-cloud-storage
        - Credentials GCP configurées
        """
        if not self.gcs_available:
            raise ImportError("google-cloud-storage non installé. Installez avec: pip install google-cloud-storage")
        
        from google.cloud import storage
        import json
        
        client = storage.Client()
        bucket = client.bucket(reference_model.s3_bucket)  # Réutilise le champ s3_bucket
        prefix = reference_model.s3_prefix
        
        logger.info(f"  Loading from GCS: gs://{reference_model.s3_bucket}/{prefix}")
        
        # Vérifier le cache local
        cache_model_dir = self.cache_dir / reference_model.model_name
        if cache_model_dir.exists():
            logger.info(f"  ✓ Using cached version: {cache_model_dir}")
            reference_model.model_path = str(cache_model_dir)
            return self._load_from_local(reference_model)
        
        # Télécharger depuis GCS
        cache_model_dir.mkdir(parents=True, exist_ok=True)
        
        files_to_download = [
            'xgboost_model.joblib',
            'feature_engineer.joblib',
            'feature_selector.joblib',
            'performance.json',
            'dataset_metadata.json'
        ]
        
        for filename in files_to_download:
            blob_name = f"{prefix}{filename}"
            local_file = cache_model_dir / filename
            
            try:
                blob = bucket.blob(blob_name)
                logger.info(f"    Downloading {filename}...")
                blob.download_to_filename(str(local_file))
                logger.info(f"    ✓ {filename} downloaded")
            except Exception as e:
                if filename == 'xgboost_model.joblib':
                    raise
                else:
                    logger.warning(f"    ⚠️  {filename} not found (optional): {e}")
        
        # Charger depuis le cache
        reference_model.model_path = str(cache_model_dir)
        logger.info(f"  ✓ Model downloaded and cached from GCS")
        return self._load_from_local(reference_model)
    
    def _load_from_azure(self, reference_model) -> Dict:
        """
        Charge depuis Azure Blob Storage
        
        Nécessite:
        - pip install azure-storage-blob
        - Connection string Azure
        """
        raise NotImplementedError("Azure Blob Storage support coming soon")
    
    def upload_model_to_s3(self, model_dir: Path, bucket: str, prefix: str) -> bool:
        """
        Upload un modèle local vers S3
        
        Utile pour migration locale → production
        
        Args:
            model_dir: Dossier local du modèle
            bucket: Nom du bucket S3
            prefix: Préfixe S3 (ex: "automl_models/dataset1/")
        
        Returns:
            True si succès
        """
        if not self.s3_available:
            raise ImportError("boto3 non installé")
        
        import boto3
        
        s3_client = boto3.client('s3')
        
        files_to_upload = [
            'xgboost_model.joblib',
            'feature_engineer.joblib',
            'feature_selector.joblib',
            'performance.json',
            'dataset_metadata.json'
        ]
        
        logger.info(f"Uploading model to S3: s3://{bucket}/{prefix}")
        
        for filename in files_to_upload:
            local_file = model_dir / filename
            if not local_file.exists():
                if filename == 'xgboost_model.joblib':
                    raise FileNotFoundError(f"Fichier requis manquant: {local_file}")
                else:
                    logger.warning(f"  ⚠️  {filename} not found (skipping)")
                    continue
            
            s3_key = f"{prefix}{filename}"
            logger.info(f"  Uploading {filename}...")
            s3_client.upload_file(str(local_file), bucket, s3_key)
            logger.info(f"  ✓ {filename} uploaded")
        
        logger.info(f"✓ Model uploaded successfully to S3")
        return True
    
    def clear_cache(self, model_name: Optional[str] = None):
        """
        Nettoie le cache local
        
        Args:
            model_name: Si spécifié, supprime uniquement ce modèle. Sinon tout le cache.
        """
        if model_name:
            cache_dir = self.cache_dir / model_name
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                logger.info(f"Cache cleared for {model_name}")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"All cache cleared")


# Singleton instance
_storage_service = None

def get_storage_service() -> ModelStorageService:
    """Factory pattern pour obtenir le service de stockage"""
    global _storage_service
    if _storage_service is None:
        _storage_service = ModelStorageService()
    return _storage_service
