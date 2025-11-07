"""
API routes for AutoML operations (AJAX endpoints)
"""
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from app.models.history import TrainingHistory
from app.services.automl_service import AutoMLService
from app.services.model_storage import ModelStorageService
import pandas as pd
import os
import json
import boto3
import shutil
from pathlib import Path
from datetime import datetime
import traceback

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Helper function to get S3 bucket name
def get_s3_bucket() -> str:
    """
    Récupère le nom du bucket S3 depuis les variables d'environnement.
    Supporte à la fois AWS_S3_BUCKET et S3_MODEL_BUCKET pour compatibilité.
    """
    return os.environ.get('S3_MODEL_BUCKET') or os.environ.get('AWS_S3_BUCKET', 'fraud-detection-ml-models')

# Helper function for S3 uploads
def upload_file_to_s3(local_path: Path, s3_bucket: str, s3_key: str) -> bool:
    """
    Upload un fichier vers S3 et supprime le fichier local
    
    Args:
        local_path: Chemin local du fichier
        s3_bucket: Nom du bucket S3
        s3_key: Clé S3 (chemin dans le bucket)
    
    Returns:
        True si succès, False sinon
    """
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(str(local_path), s3_bucket, s3_key)
        
        # Supprimer le fichier local après upload réussi
        local_path.unlink()
        current_app.logger.info(f"✅ {local_path.name} uploaded to s3://{s3_bucket}/{s3_key} and deleted locally")
        return True
    except Exception as e:
        current_app.logger.error(f"❌ Failed to upload {local_path.name} to S3: {e}")
        return False


def download_file_from_s3(s3_url: str, local_dir: Path) -> Path:
    """
    Télécharge un fichier depuis S3 vers un répertoire local temporaire
    
    Args:
        s3_url: URL S3 (format: s3://bucket/key)
        local_dir: Répertoire local où télécharger
    
    Returns:
        Path du fichier téléchargé
    
    Raises:
        Exception si le téléchargement échoue
    """
    try:
        # Parse S3 URL
        if not s3_url.startswith('s3://'):
            raise ValueError(f"Invalid S3 URL: {s3_url}")
        
        s3_url_parts = s3_url[5:].split('/', 1)
        s3_bucket = s3_url_parts[0]
        s3_key = s3_url_parts[1] if len(s3_url_parts) > 1 else ''
        
        # Create temp directory
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract filename from S3 key
        filename = s3_key.split('/')[-1]
        local_path = local_dir / filename
        
        # Download from S3
        s3_client = boto3.client('s3')
        s3_client.download_file(s3_bucket, s3_key, str(local_path))
        
        current_app.logger.info(f"✅ Downloaded {s3_url} to {local_path}")
        return local_path
        
    except Exception as e:
        current_app.logger.error(f"❌ Failed to download from S3: {e}")
        raise

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'parquet'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@api_bp.route('/analyze', methods=['POST'])
@login_required
def analyze_dataset():
    """
    Analyze uploaded dataset and detect target column
    
    Expected: multipart/form-data with 'file' field
    Returns: JSON with dataset info and detected target (filepath will be S3 URL)
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Format de fichier non supporté. Utilisez CSV, XLSX ou Parquet'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{current_user.id}_{timestamp}_{filename}"
        
        upload_dir = Path(current_app.config['PROJECT_ROOT']) / 'APP_autoML' / 'uploads'
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = upload_dir / temp_filename
        file.save(str(filepath))
        
        # Analyze dataset
        automl_service = AutoMLService(current_app.config)
        analysis_result = automl_service.analyze_dataset(str(filepath))
        
        if not analysis_result['success']:
            # Clean up temp file
            filepath.unlink(missing_ok=True)
            return jsonify({'error': analysis_result['message']}), 400
        
        # 📤 Upload to S3 and delete local file
        s3_bucket = 'fraud-detection-ml-models'
        s3_key = f"user_data/{current_user.id}/uploads/{temp_filename}"
        
        if upload_file_to_s3(filepath, s3_bucket, s3_key):
            s3_url = f"s3://{s3_bucket}/{s3_key}"
            current_app.logger.info(f"✅ Dataset uploaded to S3: {s3_url}")
            
            # Return S3 URL instead of local path
            analysis_result['filepath'] = s3_url
            analysis_result['original_filename'] = filename
            
            return jsonify(analysis_result), 200
        else:
            # If S3 upload fails, raise error
            filepath.unlink(missing_ok=True)
            raise Exception("S3 upload failed - cannot proceed without cloud storage")
        
    except Exception as e:
        current_app.logger.error(f"Error in analyze_dataset: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Erreur lors de l\'analyse: {str(e)}'}), 500


@api_bp.route('/train', methods=['POST'])
@login_required
def train_model():
    """
    Train AutoML model on labeled dataset
    
    Expected JSON:
    {
        "filepath": "s3://bucket/key or local/path.csv",
        "model_name": "my_fraud_model",
        "target_column": "is_fraud",
        "use_metatransformer": true
    }
    
    Returns: JSON with training status and history_id
    """
    temp_file = None
    try:
        data = request.get_json()
        
        # Validation
        required_fields = ['filepath', 'model_name', 'target_column']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Champ requis manquant: {field}'}), 400
        
        filepath = data['filepath']
        model_name = data['model_name']
        target_column = data['target_column']
        use_metatransformer = data.get('use_metatransformer', True)
        
        # 📥 Download from S3 if needed
        if filepath.startswith('s3://'):
            current_app.logger.info(f"📥 Downloading dataset from S3: {filepath}")
            temp_dir = Path(current_app.config['PROJECT_ROOT']) / 'APP_autoML' / 'temp' / 'datasets'
            temp_file = download_file_from_s3(filepath, temp_dir)
            filepath = str(temp_file)
            current_app.logger.info(f"✅ Using temporary file: {filepath}")
        
        # Check if file exists
        if not Path(filepath).exists():
            return jsonify({'error': 'Fichier introuvable'}), 404
        
        # Check license limits
        if not current_user.has_valid_license():
            return jsonify({
                'error': 'Licence requise',
                'message': 'Veuillez activer une licence pour utiliser AutoML',
                'redirect': '/auth/license'
            }), 403
        
        active_license = current_user.get_active_license()
        user_models_count = TrainingHistory.query.filter_by(
            user_id=current_user.id,
            status='completed'
        ).count()
        
        if user_models_count >= active_license.max_models:
            return jsonify({
                'error': f'Limite de modèles atteinte ({active_license.max_models} max). Upgradez votre licence.'
            }), 403
        
        # Create training history entry
        history = TrainingHistory(
            user_id=current_user.id,
            model_name=model_name,
            dataset_name=os.path.basename(filepath),
            status='training'
        )
        db.session.add(history)
        db.session.commit()
        
        # Train model
        automl_service = AutoMLService(current_app.config)
        training_result = automl_service.train_model(
            dataset_path=filepath,
            target_col=target_column,
            model_name=model_name,  # Passer le nom fourni par l'utilisateur
            use_meta_transformer=use_metatransformer,
            is_labeled=True
        )
        
        if training_result['success']:
            # Update history with results
            history.status = 'completed'
            model_local_path = training_result.get('model_path')
            
            # 🚀 Upload vers S3 ET supprimer local (mode cloud-only)
            s3_uploaded = False
            s3_bucket = get_s3_bucket()
            
            # Utiliser le nom du modèle généré (avec timestamp) pour éviter les collisions
            model_folder_name = Path(model_local_path).name if model_local_path else model_name
            s3_prefix = f"user_models/{current_user.id}/{model_folder_name}/"
            
            try:
                storage_service = ModelStorageService()
                
                if storage_service.s3_available and model_local_path:
                    current_app.logger.info(f"📤 Uploading trained model to S3: s3://{s3_bucket}/{s3_prefix}")
                    
                    # Upload tous les fichiers du modèle
                    s3_client = boto3.client('s3')
                    model_dir = Path(model_local_path)
                    
                    if model_dir.exists():
                        for file_path in model_dir.rglob('*'):
                            if file_path.is_file():
                                relative_path = file_path.relative_to(model_dir)
                                s3_key = f"{s3_prefix}{str(relative_path).replace(chr(92), '/')}"
                                s3_client.upload_file(str(file_path), s3_bucket, s3_key)
                        
                        s3_uploaded = True
                        
                        # ✨ CLOUD-ONLY: Supprimer le dossier local après upload
                        import shutil
                        shutil.rmtree(model_dir)
                        current_app.logger.info(f"🗑️  Local model deleted (cloud-only mode)")
                        
                        # Mettre à jour le chemin avec l'URL S3
                        history.model_path = f"s3://{s3_bucket}/{s3_prefix}"
                        current_app.logger.info(f"✅ Trained model uploaded to S3")
                    else:
                        raise FileNotFoundError(f"Model directory not found: {model_dir}")
                else:
                    raise ImportError("boto3 not available - S3 storage required")
            except Exception as e:
                current_app.logger.error(f"❌ S3 upload failed: {e}")
                history.status = 'failed'
                history.error_message = f"S3 upload failed: {str(e)}"
                db.session.commit()
                return jsonify({'error': f'Erreur lors de l\'upload S3: {str(e)}'}), 500
            
            # Extract metrics
            metrics = training_result.get('metrics', {})
            history.accuracy = metrics.get('accuracy')
            history.precision = metrics.get('precision')
            history.recall = metrics.get('recall')
            history.f1_score = metrics.get('f1_score')
            history.roc_auc = metrics.get('roc_auc')
            
            # Other info
            history.training_time_seconds = training_result.get('training_time_seconds')
            history.dataset_size = training_result.get('dataset_size')
            history.dataset_features = training_result.get('num_features')
            history.fraud_rate = training_result.get('fraud_rate')
            
            # Ajouter les infos S3 dans hyperparameters
            hyperparams = json.loads(training_result.get('hyperparameters', '{}'))
            hyperparams['storage_type'] = 's3' if s3_uploaded else 'local'
            if s3_uploaded:
                hyperparams['s3_bucket'] = s3_bucket
                hyperparams['s3_prefix'] = s3_prefix
            history.hyperparameters = json.dumps(hyperparams)
            
            history.features_engineered = training_result.get('features_engineered')
            history.meta_transformer_used = training_result.get('meta_transformer_used', use_metatransformer)
            history.automl_version = '1.0'
            
            db.session.commit()
            
            # 🗑️ Clean up temporary downloaded file
            if temp_file and temp_file.exists():
                temp_file.unlink()
                current_app.logger.info(f"🗑️  Temporary file deleted: {temp_file}")
            
            return jsonify({
                'success': True,
                'message': 'Modèle entraîné avec succès',
                'history_id': history.id,
                'metrics': metrics,
                'model_path': training_result.get('model_path')
            }), 200
        else:
            # Update history with error
            history.status = 'failed'
            db.session.commit()
            
            # Clean up temp file
            if temp_file and temp_file.exists():
                temp_file.unlink()
            
            return jsonify({'error': training_result.get('error', 'Erreur inconnue')}), 500
            
    except Exception as e:
        current_app.logger.error(f"Error in train_model: {str(e)}\n{traceback.format_exc()}")
        
        # Clean up temp file
        if 'temp_file' in locals() and temp_file and temp_file.exists():
            temp_file.unlink()
        
        # Update history if exists
        if 'history' in locals():
            history.status = 'failed'
            history.error_message = str(e)
            db.session.commit()
        
        return jsonify({'error': f'Erreur lors de l\'entraînement: {str(e)}'}), 500


@api_bp.route('/predict', methods=['POST'])
@login_required
def predict_unlabeled():
    """
    Make predictions on unlabeled dataset using trained model
    
    Expected JSON:
    {
        "model_id": 123,  # TrainingHistory ID
        "filepath": "s3://... or local/path.csv"
    }
    
    Returns: JSON with prediction results and file path
    """
    temp_dataset_file = None
    try:
        data = request.get_json()
        
        # Validation
        if 'model_id' not in data or 'filepath' not in data:
            return jsonify({'error': 'Champs requis: model_id, filepath'}), 400
        
        model_id = int(data['model_id'])  # Convert to integer
        filepath = data['filepath']
        
        # Get model from history
        model = TrainingHistory.query.filter_by(
            id=model_id,
            user_id=current_user.id,
            status='completed'
        ).first()
        
        if not model:
            return jsonify({'error': 'Modèle introuvable ou non disponible'}), 404
        
        # � Détecter le type de modèle (ensemble vs classique)
        is_ensemble = False
        try:
            if model.hyperparameters:
                hyperparams = json.loads(model.hyperparameters)
                is_ensemble = hyperparams.get('model_type') == 'ensemble_unlabeled'
                current_app.logger.info(f"📋 Type de modèle détecté: {'ENSEMBLE' if is_ensemble else 'CLASSIQUE'}")
        except:
            pass
        
        # �📥 Download dataset from S3 if needed
        if filepath.startswith('s3://'):
            current_app.logger.info(f"📥 Downloading dataset from S3: {filepath}")
            temp_dir = Path(current_app.config['PROJECT_ROOT']) / 'APP_autoML' / 'temp' / 'datasets'
            temp_dataset_file = download_file_from_s3(filepath, temp_dir)
            filepath = str(temp_dataset_file)
            current_app.logger.info(f"✅ Using temporary file: {filepath}")
        
        # Check file exists
        if not Path(filepath).exists():
            return jsonify({'error': 'Fichier de données introuvable'}), 404
        
        # Check license limits for predictions
        if not current_user.has_valid_license():
            return jsonify({
                'error': 'Licence requise',
                'message': 'Veuillez activer une licence pour faire des prédictions',
                'redirect': '/auth/license'
            }), 403
        
        # 🚀 FLUX DIFFÉRENT SELON LE TYPE DE MODÈLE
        if is_ensemble:
            # ========== MODÈLE ENSEMBLE ==========
            return _predict_with_ensemble(model, filepath, temp_dataset_file, current_user, current_app)
        else:
            # ========== MODÈLE CLASSIQUE ==========
            return _predict_with_classic_model(model, filepath, temp_dataset_file, current_user, current_app)
    
    except Exception as e:
        current_app.logger.error(f"Error in predict_unlabeled: {str(e)}\n{traceback.format_exc()}")
        
        # Clean up temp file
        if 'temp_dataset_file' in locals() and temp_dataset_file and temp_dataset_file.exists():
            temp_dataset_file.unlink()
            current_app.logger.info(f"🗑️  Cleaned up temp dataset after error")
        
        return jsonify({'error': f'Erreur lors de la prédiction: {str(e)}'}), 500


def _predict_with_classic_model(model, filepath, temp_dataset_file, current_user, current_app):
    """Prédictions avec un modèle classique (entraîné sur données étiquetées)"""
    try:
        # Make predictions using the specific model
        import sys
        import joblib
        import tempfile
        import shutil
        # Add parent directory to path for automl_transformer import
        project_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(project_root))
        from automl_transformer.full_automl import FullAutoML
        
        # 🚀 Charger le modèle (depuis S3 ou local)
        model_path = model.model_path
        model_id = model.id
        
        # Vérifier si c'est une URL S3
        if model_path.startswith('s3://'):
            current_app.logger.info(f"📥 Downloading model from S3: {model_path}")
            
            try:
                # Parser l'URL S3
                s3_url_parts = model_path.replace('s3://', '').split('/', 1)
                s3_bucket = s3_url_parts[0]
                s3_prefix = s3_url_parts[1] if len(s3_url_parts) > 1 else ''
                
                # Créer un dossier temporaire pour télécharger le modèle
                temp_model_dir = Path(tempfile.gettempdir()) / 'model_cache' / f"model_{model_id}"
                temp_model_dir.mkdir(parents=True, exist_ok=True)
                
                # Télécharger tous les fichiers du modèle depuis S3
                s3_client = boto3.client('s3')
                
                # Lister les fichiers dans S3
                response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
                
                if 'Contents' not in response:
                    return jsonify({'error': 'Modèle introuvable sur S3'}), 404
                
                # Télécharger chaque fichier
                for obj in response['Contents']:
                    s3_key = obj['Key']
                    # Construire le chemin local relatif
                    relative_path = s3_key[len(s3_prefix):].lstrip('/')
                    if not relative_path:  # Skip if empty (folder itself)
                        continue
                    
                    local_file = temp_model_dir / relative_path
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    current_app.logger.debug(f"  Downloading {s3_key} -> {local_file}")
                    s3_client.download_file(s3_bucket, s3_key, str(local_file))
                
                model_dir = temp_model_dir
                current_app.logger.info(f"✅ Model downloaded to {model_dir}")
                
            except Exception as e:
                current_app.logger.error(f"❌ Failed to download from S3: {e}")
                return jsonify({'error': f'Impossible de charger le modèle depuis S3: {str(e)}'}), 500
        else:
            # Chemin local
            model_dir = Path(model_path)
            if not model_dir.exists():
                return jsonify({'error': 'Modèle introuvable sur le disque'}), 404
        
        # Initialize FullAutoML and load model
        automl = FullAutoML(use_meta_transformer=True)
        automl.load_model(str(model_dir))
        
        # Load data to predict
        import pandas as pd
        if filepath.endswith('.csv'):
            data_to_predict = pd.read_csv(filepath)
        elif filepath.endswith('.parquet'):
            data_to_predict = pd.read_parquet(filepath)
        else:
            return jsonify({'error': 'Format de fichier non supporté (CSV ou Parquet uniquement)'}), 400
        
        # Make predictions
        predictions = automl.predict(data_to_predict)
        
        # Save predictions
        output_filename = f"predictions_{model.model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = Path(current_app.config['UPLOAD_FOLDER']) / output_filename
        
        # ============================================================
        # CSV SIMPLIFIÉ: Customer ID + Transaction ID + Timestamp + Fraude
        # Détection automatique des colonnes (même logique que full_automl)
        # ============================================================
        results_df = pd.DataFrame()
        
        print(f"\n🔍 Détection automatique des colonnes pour CSV simplifié...")
        print(f"   Colonnes disponibles: {data_to_predict.columns.tolist()}")
        
        # 1. DÉTECTER LA COLONNE CUSTOMER ID
        customer_id_col = None
        
        # Mots-clés qui indiquent un ID client (avec id/ref/code)
        customer_id_keywords = ['customer_id', 'cust_id', 'client_id', 'customer_ref', 'cust_ref', 
                                'client_ref', 'customer_code', 'cust_code', 'customerid', 'custid']
        
        # Mots-clés plus génériques (customer, cust, client sans suffixe)
        customer_general_keywords = ['customer', 'cust', 'client', 'user']
        
        for col in data_to_predict.columns:
            col_lower = col.lower()
            
            # Exclure les colonnes qui ne sont clairement PAS des IDs
            if any(word in col_lower for word in ['age', 'tenure', 'days', 'months', 'years', 'amount', 
                                                   'balance', 'type', 'region', 'status', 'date', 'time']):
                print(f"   ❌ Skipping '{col}': contains exclusion word")
                continue
            
            # Exclure les colonnes avec 'transaction' ou 'tx' (ce sont des IDs de transaction)
            if 'transaction' in col_lower or 'tx' in col_lower:
                print(f"   ❌ Skipping '{col}': contains transaction/tx")
                continue
            
            # Priorité 1: Mots-clés spécifiques (customer_id, cust_ref, etc.)
            if any(keyword in col_lower for keyword in customer_id_keywords):
                unique_ratio = data_to_predict[col].nunique() / len(data_to_predict)
                print(f"   🔍 Checking '{col}': keyword match, uniqueness={unique_ratio:.2%}")
                # Cardinalité entre 1% et 90% (1% = au moins 100 clients différents sur 10k transactions)
                if 0.01 < unique_ratio < 0.90:
                    customer_id_col = col
                    print(f"   ✅ SELECTED as Customer ID: {col}")
                    break
                else:
                    print(f"   ❌ Rejected: uniqueness out of range [1%, 90%]")
            
            # Priorité 2: Mots-clés génériques (customer, cust) SEULEMENT s'ils sont seuls ou avec _id/_ref
            elif any(keyword == col_lower or f'{keyword}_' in col_lower or f'_{keyword}' in col_lower 
                   for keyword in customer_general_keywords):
                unique_ratio = data_to_predict[col].nunique() / len(data_to_predict)
                print(f"   🔍 Checking '{col}': general keyword match, uniqueness={unique_ratio:.2%}")
                if 0.01 < unique_ratio < 0.90:
                    customer_id_col = col
                    print(f"   ✅ SELECTED as Customer ID: {col}")
                    break
                else:
                    print(f"   ❌ Rejected: uniqueness out of range [1%, 90%]")
        
        # Ajouter Customer_ID seulement si détecté
        if customer_id_col:
            results_df['Customer_ID'] = data_to_predict[customer_id_col]
            print(f"   ✅ Customer ID detected: {customer_id_col}")
        else:
            print(f"   ⚠️  No Customer ID column detected")
        
        # 2. DÉTECTER LA COLONNE TRANSACTION ID (cardinalité élevée)
        transaction_id_col = None
        transaction_patterns = ['tx', 'transaction', 'trans']
        
        # Rechercher d'abord par nom spécifique
        for col in data_to_predict.columns:
            if col == customer_id_col:
                continue  # Skip customer ID
            
            col_lower = col.lower()
            # Chercher des mots-clés transaction + id/ref
            if any(pattern in col_lower for pattern in transaction_patterns) and ('id' in col_lower or 'ref' in col_lower):
                unique_ratio = data_to_predict[col].nunique() / len(data_to_predict)
                if unique_ratio > 0.85:  # Haute cardinalité (probablement unique par transaction)
                    transaction_id_col = col
                    break
        
        # Si pas trouvé, chercher une colonne 'id' générique (différente de customer_id)
        if transaction_id_col is None:
            for col in data_to_predict.columns:
                if col == customer_id_col:
                    continue  # Skip customer ID
                col_lower = col.lower()
                if ('id' in col_lower or 'identifier' in col_lower) and 'customer' not in col_lower and 'cust' not in col_lower:
                    unique_ratio = data_to_predict[col].nunique() / len(data_to_predict)
                    if unique_ratio > 0.85:
                        transaction_id_col = col
                        break
        
        # Fallback: première colonne si pas trouvé (et différente de customer_id)
        if transaction_id_col is None:
            transaction_id_col = data_to_predict.columns[0] if data_to_predict.columns[0] != customer_id_col else data_to_predict.columns[1]
        
        results_df['Transaction_ID'] = data_to_predict[transaction_id_col]
        
        # 3. DÉTECTER LA COLONNE TIMESTAMP
        timestamp_col = None
        timestamp_patterns = ['date', 'time', 'timestamp', 'datetime', 'created', 
                             'transaction_date', 'trans_date', 'date_transaction']
        
        for col in data_to_predict.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in timestamp_patterns):
                # Vérifier si c'est une date/datetime
                if data_to_predict[col].dtype == 'object':
                    try:
                        pd.to_datetime(data_to_predict[col].iloc[0])
                        timestamp_col = col
                        break
                    except:
                        pass
                elif 'datetime' in str(data_to_predict[col].dtype):
                    timestamp_col = col
                    break
        
        # Fallback: créer timestamp si pas trouvé
        if timestamp_col is None:
            results_df['Timestamp'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        else:
            results_df['Timestamp'] = data_to_predict[timestamp_col]
        
        # 4. AJOUTER PRÉDICTION FRAUDE
        results_df['Fraude'] = predictions.astype(int)
        
        # Sauvegarder CSV simplifié
        results_df.to_csv(output_path, index=False)
        
        # Calculate summary
        fraud_count = int((predictions == 1).sum())
        total_count = len(predictions)
        fraud_rate = (fraud_count / total_count * 100) if total_count > 0 else 0
        
        # 🗑️ Clean up temporary dataset file
        if temp_dataset_file and temp_dataset_file.exists():
            temp_dataset_file.unlink()
            current_app.logger.info(f"🗑️  Temporary dataset deleted: {temp_dataset_file}")
        
        return jsonify({
            'success': True,
            'message': 'Prédictions effectuées avec succès',
            'output_path': str(output_path),
            'predictions_summary': {
                'total_transactions': total_count,
                'fraud_detected': fraud_count,
                'fraud_rate': round(fraud_rate, 2),
                'normal_transactions': total_count - fraud_count
            }
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error in _predict_with_classic_model: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Erreur lors de la prédiction: {str(e)}'}), 500


def _predict_with_ensemble(model, filepath, temp_dataset_file, current_user, current_app):
    """Prédictions avec un modèle ensemble (avec anomaly detection + calibration)"""
    try:
        import sys
        from pathlib import Path as PathLib
        sys.path.insert(0, str(PathLib(__file__).parent.parent.parent.parent))
        from automl_transformer.apply_automl_production import AutoMLProductionApplicator
        
        current_app.logger.info(f"🚀 Using ENSEMBLE model for predictions on: {filepath}")
        
        # Charger le dataset
        df_prod = pd.read_csv(filepath)
        
        # Initialize applicator
        applicator = AutoMLProductionApplicator(
            automl_models_dir=str(current_app.config['AUTOML_MODELS_DIR'])
        )
        
        # Charger le modèle ensemble depuis S3 ou local
        model_path = model.model_path
        ensemble_dir = None
        
        if model_path.startswith('s3://'):
            current_app.logger.info(f"📥 Downloading ensemble model from S3: {model_path}")
            
            # Parser l'URL S3
            s3_url_parts = model_path.replace('s3://', '').split('/', 1)
            s3_bucket = s3_url_parts[0]
            s3_prefix = s3_url_parts[1] if len(s3_url_parts) > 1 else ''
            
            # Créer un dossier temporaire
            import tempfile
            ensemble_dir = PathLib(tempfile.gettempdir()) / 'model_cache' / f"ensemble_{model.id}"
            ensemble_dir.mkdir(parents=True, exist_ok=True)
            
            # Télécharger tous les fichiers
            s3_client = boto3.client('s3')
            response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
            
            if 'Contents' not in response:
                return jsonify({'error': 'Modèle ensemble introuvable sur S3'}), 404
            
            for obj in response['Contents']:
                s3_key = obj['Key']
                relative_path = s3_key[len(s3_prefix):].lstrip('/')
                if not relative_path:
                    continue
                
                local_file = ensemble_dir / relative_path
                local_file.parent.mkdir(parents=True, exist_ok=True)
                s3_client.download_file(s3_bucket, s3_key, str(local_file))
            
            current_app.logger.info(f"✅ Ensemble model downloaded to {ensemble_dir}")
        else:
            ensemble_dir = PathLib(model_path)
        
        # Charger les informations de l'ensemble depuis les hyperparameters de la BDD
        hyperparams = json.loads(model.hyperparameters)
        top_models = [(m['name'], m['similarity']) for m in hyperparams.get('top_models', [])]
        
        if not top_models:
            current_app.logger.error(f"❌ No models found in hyperparameters: {hyperparams}")
            return jsonify({'error': 'Aucun modèle trouvé dans l\'ensemble'}), 400
        
        current_app.logger.info(f"✅ Loaded {len(top_models)} models from hyperparameters: {[name for name, _ in top_models]}")
        
        # Pour les datasets non étiquetés, on utilise un threshold adaptatif
        # basé sur la contamination attendue (% de fraudes dans les données)
        contamination = 0.04  # 4% de fraudes attendues par défaut
        
        # Calculer un threshold basé sur la contamination
        # Pour un ensemble, on utilise un threshold plus bas que 0.5
        threshold = 0.25  # 25% - plus permissif que le mode classique
        
        # Appliquer le pipeline complet (l'applicator va charger les modèles depuis automl_models_dir)
        current_app.logger.info(f"🤖 Applying ensemble predictions with {len(top_models)} models...")
        current_app.logger.info(f"📊 Using threshold={threshold}, contamination={contamination}")
        
        # L'applicator utilise les modèles de référence depuis automl_models_dir
        # Il va automatiquement les charger en fonction des noms dans top_models
        results = applicator.apply_ensemble_predictions(
            df=df_prod,
            top_k=len(top_models),  # Utiliser tous les modèles de l'ensemble
            threshold=threshold,  # Threshold pour la prédiction binaire
            verbose=True
        )
        
        # S'assurer que les bons modèles sont utilisés (override si nécessaire)
        # Note: les top_models sont déjà sélectionnés, on utilise juste leur config
        if 'top_models' not in results.attrs:
            results.attrs['top_models'] = top_models
        
        # Anomaly detection - crucial pour déterminer le threshold
        current_app.logger.info("🔍 Adding anomaly detection...")
        results = applicator.add_anomaly_detection(
            df=df_prod,
            results=results,
            contamination=contamination,
            verbose=True
        )
        
        # Calibration
        current_app.logger.info("📊 Calibrating probabilities...")
        results = applicator.calibrate_probabilities(
            results=results,
            method='isotonic',
            verbose=True
        )
        
        # Créer le CSV enrichi
        current_app.logger.info("📊 Creating enriched CSV output...")
        output_df = _create_simplified_output_unlabeled(df_prod, results)
        
        # Sauvegarder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"predictions_{model.model_name}_{timestamp}.csv"
        output_dir = PathLib(current_app.config['UPLOAD_FOLDER']) / 'predictions'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / output_filename
        
        output_df.to_csv(output_path, index=False)
        current_app.logger.info(f"✅ Predictions saved locally: {output_path}")
        
        # Calculer les statistiques avec les seuils optimisés (40% et 50%)
        fraud_detected = int((results['combined_score'] >= 0.40).sum())  # MEDIUM + HIGH
        stats = {
            'total_transactions': len(results),
            'fraud_detected': fraud_detected,
            'fraud_rate': round(fraud_detected / len(results) * 100, 2),
            'normal_transactions': len(results) - fraud_detected,
            'high_risk': int((results['combined_score'] >= 0.50).sum()),
            'medium_risk': int(((results['combined_score'] >= 0.40) & (results['combined_score'] < 0.50)).sum()),
            'low_risk': int((results['combined_score'] < 0.40).sum()),
            'anomalies_detected': int(results['is_anomaly'].sum()),
            'avg_fraud_probability': float(results['fraud_probability'].mean()),
            'avg_combined_score': float(results['combined_score'].mean())
        }
        
        # � Upload predictions to S3 for persistent storage
        s3_key = None
        download_url = None
        try:
            s3_bucket = get_s3_bucket()
            s3_key = f"user_data/{current_user.id}/predictions/{output_filename}"
            
            s3_client = boto3.client('s3')
            s3_client.upload_file(str(output_path), s3_bucket, s3_key)
            
            download_url = f"/api/download_s3_predictions?key={s3_key}"
            current_app.logger.info(f"✅ Predictions uploaded to S3: s3://{s3_bucket}/{s3_key}")
            
            # Clean up local file after upload
            output_path.unlink()
            current_app.logger.info(f"�🗑️  Local predictions file deleted (cloud-only mode)")
            
        except Exception as e:
            current_app.logger.warning(f"⚠️  S3 upload failed, keeping local file: {e}")
            download_url = str(output_path)  # Fallback to local path
        
        # 🗑️ Clean up temp dataset
        if temp_dataset_file and temp_dataset_file.exists():
            temp_dataset_file.unlink()
        
        return jsonify({
            'success': True,
            'message': f'Prédictions ensemble effectuées avec succès sur {len(results)} transactions',
            'output_path': f"s3://{s3_bucket}/{s3_key}" if s3_key else str(output_path),
            'download_url': download_url if download_url else f"/download/{output_filename}",
            'predictions_summary': stats,
            'model_type': 'ensemble'
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in _predict_with_ensemble: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Erreur lors de la prédiction ensemble: {str(e)}'}), 500


@api_bp.route('/models', methods=['GET'])
@login_required
def get_user_models():
    """
    Get list of user's trained models
    
    Returns: JSON list of models
    """
    try:
        models = TrainingHistory.query.filter_by(
            user_id=current_user.id,
            status='completed'
        ).order_by(
            TrainingHistory.created_at.desc()
        ).all()
        
        models_list = []
        for model in models:
            models_list.append({
                'id': model.id,
                'model_name': model.model_name,
                'model_type': model.model_type,
                'target_column': model.target_column,
                'metrics': model.metrics,
                'created_at': model.created_at.isoformat(),
                'prediction_count': model.prediction_count,
                'last_used': model.last_used.isoformat() if model.last_used else None
            })
        
        return jsonify({'models': models_list}), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in get_user_models: {str(e)}")
        return jsonify({'error': f'Erreur lors de la récupération des modèles: {str(e)}'}), 500


@api_bp.route('/history', methods=['GET'])
@login_required
def get_training_history():
    """
    Get user's training history
    
    Query params:
    - page: page number (default: 1)
    - per_page: items per page (default: 20)
    - status: filter by status (optional)
    
    Returns: JSON with paginated history
    """
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        status_filter = request.args.get('status', None)
        
        query = TrainingHistory.query.filter_by(user_id=current_user.id)
        
        if status_filter:
            query = query.filter_by(status=status_filter)
        
        pagination = query.order_by(
            TrainingHistory.created_at.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
        
        history_list = []
        for item in pagination.items:
            history_list.append({
                'id': item.id,
                'model_name': item.model_name,
                'model_type': item.model_type,
                'status': item.status,
                'target_column': item.target_column,
                'metrics': item.metrics,
                'created_at': item.created_at.isoformat(),
                'completed_at': item.completed_at.isoformat() if item.completed_at else None,
                'training_duration': item.training_duration,
                'error_message': item.error_message
            })
        
        return jsonify({
            'history': history_list,
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': pagination.page,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in get_training_history: {str(e)}")
        return jsonify({'error': f'Erreur lors de la récupération de l\'historique: {str(e)}'}), 500


@api_bp.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """
    Upload file for prediction (separate from analyze)
    
    Returns: JSON with S3 filepath for later use
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Format non supporté'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{current_user.id}_{timestamp}_{filename}"
        
        upload_dir = Path(current_app.config['PROJECT_ROOT']) / 'APP_autoML' / 'uploads'
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = upload_dir / temp_filename
        file.save(str(filepath))
        
        # Get basic file info
        df = pd.read_csv(filepath) if filepath.suffix == '.csv' else pd.read_excel(filepath)
        file_info = {
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist()
        }
        
        # Upload to S3 and delete local file
        s3_bucket = 'fraud-detection-ml-models'
        s3_key = f"user_data/{current_user.id}/uploads/{temp_filename}"
        
        if upload_file_to_s3(filepath, s3_bucket, s3_key):
            s3_url = f"s3://{s3_bucket}/{s3_key}"
            current_app.logger.info(f"✅ Dataset uploaded to S3: {s3_url}")
            
            return jsonify({
                'success': True,
                'filepath': s3_url,  # Return S3 URL for later use
                **file_info
            }), 200
        else:
            # If S3 upload fails, raise error
            raise Exception("S3 upload failed - cannot proceed without cloud storage")
        
    except Exception as e:
        current_app.logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': f'Erreur lors de l\'upload: {str(e)}'}), 500


@api_bp.route('/status/<int:history_id>', methods=['GET'])
@login_required
def get_training_status(history_id):
    """
    Get status of a training job
    
    Returns: JSON with current status
    """
    try:
        history = TrainingHistory.query.filter_by(
            id=history_id,
            user_id=current_user.id
        ).first()
        
        if not history:
            return jsonify({'error': 'Entraînement introuvable'}), 404
        
        return jsonify({
            'status': history.status,
            'model_name': history.model_name,
            'created_at': history.created_at.isoformat(),
            'completed_at': history.completed_at.isoformat() if history.completed_at else None,
            'metrics': history.metrics,
            'error_message': history.error_message
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in get_training_status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/apply_unlabeled', methods=['POST'])
@login_required
def apply_unlabeled():
    """
    Create ensemble model from reference models (WITHOUT making predictions)
    
    🚀 MODE CRÉATION ENSEMBLE:
    - Analyse le dataset pour trouver les meilleurs modèles de référence
    - Crée un modèle ensemble (3 meilleurs modèles)
    - Sauvegarde le modèle ensemble sur S3
    - NE FAIT PAS de prédictions (l'utilisateur les fera depuis la page "Faire des prédictions")
    
    Expected: JSON with {'filepath': 's3://... or local/path', 'model_name': '...'}
    Returns: JSON with ensemble model info (no predictions)
    """
    temp_file = None
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        model_name = data.get('model_name', 'unlabeled_predictions')
        
        if not filepath:
            return jsonify({'error': 'Aucun fichier spécifié'}), 400
        
        # 📥 Download from S3 if needed
        if filepath.startswith('s3://'):
            current_app.logger.info(f"📥 Downloading dataset from S3: {filepath}")
            temp_dir = Path(current_app.config['PROJECT_ROOT']) / 'APP_autoML' / 'temp' / 'datasets'
            temp_file = download_file_from_s3(filepath, temp_dir)
            filepath = str(temp_file)
            current_app.logger.info(f"✅ Using temporary file: {filepath}")
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Fichier introuvable'}), 404
        
        current_app.logger.info(f"🚀 Creating ensemble model based on dataset: {filepath}")
        
        # Import models
        from app.models.reference_model import ReferenceModel
        
        # Charger le dataset pour analyser les colonnes
        current_app.logger.info("📂 Loading dataset for analysis...")
        df_prod = pd.read_csv(filepath)
        
        # Trouver le meilleur modèle de référence basé sur la similarité des colonnes
        current_app.logger.info("🔍 Finding best matching reference models...")
        column_names = df_prod.columns.tolist()
        dataset_size = len(df_prod)
        
        # Calculer le fraud_rate si disponible (sinon None)
        fraud_rate = None
        
        best_model, similarity_score = ReferenceModel.find_best_match(
            column_names=column_names,
            dataset_size=dataset_size,
            fraud_rate=fraud_rate
        )
        
        if not best_model:
            return jsonify({'error': 'Aucun modèle de référence trouvé dans la base de données'}), 500
        
        current_app.logger.info(f"✅ Best match: {best_model.model_name} (similarity: {similarity_score:.2%})")
        
        # Incrémenter le compteur d'utilisation
        best_model.increment_usage(similarity_score)
        
        # Import apply_automl_production
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from automl_transformer.apply_automl_production import AutoMLProductionApplicator
        
        # Initialize applicator avec le chemin des modèles de référence
        applicator = AutoMLProductionApplicator(
            automl_models_dir=str(current_app.config['AUTOML_MODELS_DIR'])
        )
        
        # ✅ ÉTAPE 1: Ensemble predictions (top 3 models) - sur tout le dataset
        current_app.logger.info(f"🤖 Testing ensemble on {len(df_prod)} transactions to select top 3 models...")
        results = applicator.apply_ensemble_predictions(
            df=df_prod,
            top_k=3,
            threshold=0.5,
            verbose=True
        )
        
        
        # Generate timestamp first
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ensemble_model_dir = Path(current_app.config['MODELS_FOLDER']) / f"ensemble_{model_name}_{timestamp}"
        
        # Récupérer les top_models depuis results.attrs
        top_models = results.attrs.get('top_models', [])
        
        model_save_info = applicator.save_ensemble_model(
            output_dir=str(ensemble_model_dir),
            top_models=top_models,
            results=results,
            dataset_name=os.path.basename(filepath)
        )
        current_app.logger.info(f"✅ Ensemble model saved: {model_save_info['model_path']}")
        
        # ÉTAPE 5: Upload vers S3 (stockage cloud par utilisateur)
        s3_uploaded = False
        s3_bucket = get_s3_bucket()
        s3_prefix = f"user_models/{current_user.id}/ensemble_{model_name}_{timestamp}/"
        
        try:
            storage_service = ModelStorageService()
            
            if storage_service.s3_available:
                current_app.logger.info(f"📤 Uploading to S3: s3://{s3_bucket}/{s3_prefix}")
                
                # Upload tous les fichiers du dossier ensemble
                s3_client = boto3.client('s3')
                for file_path in ensemble_model_dir.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(ensemble_model_dir)
                        s3_key = f"{s3_prefix}{str(relative_path).replace(chr(92), '/')}"
                        s3_client.upload_file(str(file_path), s3_bucket, s3_key)
                
                s3_uploaded = True
                
                # ✨ CLOUD-ONLY: Supprimer le dossier local après upload
                import shutil
                shutil.rmtree(ensemble_model_dir)
                current_app.logger.info(f"🗑️  Local ensemble model deleted (cloud-only mode)")
                current_app.logger.info(f"✅ Model uploaded to S3")
            else:
                raise ImportError("boto3 not available - S3 storage required")
        except Exception as e:
            current_app.logger.error(f"❌ S3 upload failed: {e}")
            return jsonify({'error': f'Erreur lors de l\'upload S3: {str(e)}'}), 500
        
        # �💾 ÉTAPE 6: Sauvegarder en base de données (pour affichage dans /models)
        try:
            ensemble_model_record = TrainingHistory(
                user_id=current_user.id,
                model_name=f"ensemble_{model_name}_{timestamp}",
                dataset_name=os.path.basename(filepath),
                dataset_size=len(df_prod),
                dataset_features=len(df_prod.columns),
                model_path=f"s3://{s3_bucket}/{s3_prefix}",
                f1_score=None,  # Pas de F1 pour unlabeled
                roc_auc=None,
                precision=None,
                recall=None,
                accuracy=None,
                training_time_seconds=0,
                status='completed',
                meta_transformer_used=True,
                hyperparameters=json.dumps({
                    'model_type': 'ensemble_unlabeled',
                    'storage_type': 's3',
                    's3_bucket': s3_bucket,
                    's3_prefix': s3_prefix,
                    'top_models': [{'name': name, 'similarity': float(sim)} for name, sim in top_models],
                    'n_models': len(top_models),
                    'best_match': best_model.model_name,
                    'similarity_score': float(similarity_score),
                    'methods': ['ensemble', 'anomaly_detection', 'isotonic_calibration']
                }),
                features_engineered=json.dumps({
                    'dataset_rows': len(df_prod),
                    'dataset_columns': len(df_prod.columns),
                    'top_models_selected': len(top_models)
                })
            )
            db.session.add(ensemble_model_record)
            db.session.commit()
            current_app.logger.info(f"✅ Model saved to database: ID={ensemble_model_record.id}")
        except Exception as e:
            current_app.logger.warning(f"⚠️ Could not save to database: {e}")
            db.session.rollback()
        
        # 🗑️ Clean up temporary downloaded dataset
        if temp_file and temp_file.exists():
            temp_file.unlink()
            current_app.logger.info(f"🗑️  Temporary dataset deleted: {temp_file}")
        
        return jsonify({
            'success': True,
            'message': f'Modèle ensemble créé avec succès',
            'model_id': ensemble_model_record.id,
            'model_path': f"s3://{s3_bucket}/{s3_prefix}",
            'model_name': f"ensemble_{model_name}_{timestamp}",
            'info': {
                'dataset_analyzed': len(df_prod),
                'dataset_rows': len(df_prod),
                'top_models_selected': len(top_models),
                'best_match': best_model.model_name,
                'similarity_score': float(similarity_score),
                'n_base_models': model_save_info['n_models']
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in apply_unlabeled: {str(e)}\n{traceback.format_exc()}")
        
        # Clean up temp files
        if 'temp_file' in locals() and temp_file and temp_file.exists():
            temp_file.unlink()
            current_app.logger.info(f"🗑️  Cleaned up temp file after error")
        
        return jsonify({'error': f'Erreur lors de la création du modèle: {str(e)}'}), 500


def _create_simplified_output_unlabeled(df_original: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """
    Create simplified CSV output for unlabeled predictions
    Format: Customer_ID (optional), Transaction_ID, Timestamp, Fraud_Probability, Risk_Level
    """
    output_columns = []
    
    # Detect Customer ID
    customer_id_col = None
    customer_keywords = ['customer_id', 'cust_id', 'customer_ref', 'cust_ref', 'client_id', 
                        'client_ref', 'custid', 'clientid', 'account_id', 'accountid']
    
    for col in df_original.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in customer_keywords):
            # Exclude unwanted columns
            if not any(exclude in col_lower for exclude in ['age', 'tenure', 'days', 'months', 
                                                            'years', 'amount', 'balance', 'type', 
                                                            'region', 'status', 'date', 'time']):
                # Check cardinality
                uniqueness = df_original[col].nunique() / len(df_original)
                if 0.01 <= uniqueness <= 0.90:
                    customer_id_col = col
                    break
    
    if customer_id_col:
        output_columns.append(('Customer_ID', df_original[customer_id_col]))
    
    # Detect Transaction ID
    transaction_id_col = None
    tx_patterns = ['transaction_id', 'tx_id', 'trans_id', 'transid', 'txid', 
                  'transaction_number', 'tx_number', 'reference', 'ref', 'id']
    
    for col in df_original.columns:
        col_lower = col.lower()
        for pattern in tx_patterns:
            if pattern in col_lower and col != customer_id_col:
                uniqueness = df_original[col].nunique() / len(df_original)
                if uniqueness > 0.85:
                    transaction_id_col = col
                    break
        if transaction_id_col:
            break
    
    if transaction_id_col:
        output_columns.append(('Transaction_ID', df_original[transaction_id_col]))
    
    # Detect Timestamp
    timestamp_col = None
    for col in df_original.columns:
        if pd.api.types.is_datetime64_any_dtype(df_original[col]):
            timestamp_col = col
            break
        if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower():
            try:
                pd.to_datetime(df_original[col])
                timestamp_col = col
                break
            except:
                pass
    
    if timestamp_col:
        output_columns.append(('Timestamp', df_original[timestamp_col]))
    
    # Add predictions columns from apply_automl_production.py
    # ✅ Colonnes générées par apply_ensemble_predictions()
    output_columns.append(('Fraud_Probability', results['fraud_probability']))
    output_columns.append(('Prediction_Variance', results['prediction_variance']))
    output_columns.append(('Prediction_Stability', results['prediction_stability']))
    
    # ✅ Colonnes générées par add_anomaly_detection()
    output_columns.append(('Anomaly_Score', results['anomaly_score']))
    output_columns.append(('Is_Anomaly', results['is_anomaly']))
    output_columns.append(('Combined_Score', results['combined_score']))
    
    # ✅ Colonnes générées par calibrate_probabilities()
    output_columns.append(('Fraud_Probability_Calibrated', results['fraud_probability_calibrated']))
    
    # Add risk level (basé sur combined_score)
    # 🎯 Seuils optimisés pour le mode ensemble (équilibre détection/précision)
    # Compromis entre sensibilité du mode ensemble et contrôle des faux positifs
    risk_levels = []
    fraude_binary = []  # Colonne binaire pour compatibilité avec mode classique
    
    for score in results['combined_score']:
        if score >= 0.50:  # 50% - Très haute confiance
            risk_levels.append('HIGH')
            fraude_binary.append(1)
        elif score >= 0.40:  # 40% - Confiance moyenne (seuil de détection)
            risk_levels.append('MEDIUM')
            fraude_binary.append(1)  # MEDIUM et HIGH = Fraude
        else:  # < 40% - Probablement légitime
            risk_levels.append('LOW')
            fraude_binary.append(0)
    
    output_columns.append(('Risk_Level', risk_levels))
    output_columns.append(('Fraude', fraude_binary))  # Ajout colonne binaire
    
    # Add anomaly score if available
    if 'anomaly_score' in results.columns:
        output_columns.append(('Anomaly_Score', results['anomaly_score']))
    
    # Add prediction stability if available
    if 'prediction_stability' in results.columns:
        output_columns.append(('Prediction_Stability', results['prediction_stability']))
    
    # Create output dataframe
    output_dict = {name: values for name, values in output_columns}
    output_df = pd.DataFrame(output_dict)
    
    return output_df


@api_bp.route('/download_s3_predictions', methods=['GET'])
@login_required
def download_s3_predictions():
    """
    Download predictions CSV from S3
    
    Expected: ?key=user_data/{user_id}/predictions/filename.csv
    Returns: CSV file download
    """
    try:
        from flask import send_file
        import tempfile
        
        s3_key = request.args.get('key')
        current_app.logger.info(f"📥 Download request - S3 key received: {s3_key}")
        current_app.logger.info(f"📥 Current user ID: {current_user.id}")
        current_app.logger.info(f"📥 Full request URL: {request.url}")
        current_app.logger.info(f"📥 Request args: {request.args}")
        
        if not s3_key:
            current_app.logger.error("❌ S3 key manquante dans la requête")
            return jsonify({'error': 'S3 key manquante'}), 400
        
        # Verify user has access (key must contain their user_id)
        expected_prefix = f"user_data/{current_user.id}/"
        if expected_prefix not in s3_key:
            current_app.logger.error(f"❌ Accès refusé - Expected prefix: {expected_prefix}, Got: {s3_key}")
            return jsonify({'error': 'Accès non autorisé'}), 403
        
        # Download from S3 to temp file
        s3_bucket = get_s3_bucket()
        s3_client = boto3.client('s3')
        
        # Extract filename from S3 key
        filename = s3_key.split('/')[-1]
        current_app.logger.info(f"📥 Downloading from S3: s3://{s3_bucket}/{s3_key}")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        temp_path = temp_file.name
        temp_file.close()
        
        # Download from S3
        current_app.logger.info(f"📥 Downloading from S3: s3://{s3_bucket}/{s3_key}")
        s3_client.download_file(s3_bucket, s3_key, temp_path)
        
        # Send file to user (file will be deleted after sending)
        return send_file(
            temp_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        current_app.logger.error(f"Error downloading from S3: {str(e)}")
        return jsonify({'error': f'Erreur lors du téléchargement: {str(e)}'}), 500
