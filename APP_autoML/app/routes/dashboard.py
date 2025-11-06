"""
Dashboard routes for main application interface
"""
from flask import Blueprint, render_template, request, jsonify, send_file, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from app.models.history import TrainingHistory
from app.services.automl_service import AutoMLService
import pandas as pd
import os
import shutil
from pathlib import Path
from datetime import datetime

dashboard_bp = Blueprint('dashboard', __name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'parquet'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@dashboard_bp.route('/')
@login_required
def index():
    """Main dashboard page"""
    # Get user statistics
    total_models = TrainingHistory.query.filter_by(
        user_id=current_user.id,
        status='completed'
    ).count()
    
    # Total predictions = nombre total d'entraînements
    total_predictions = TrainingHistory.query.filter_by(
        user_id=current_user.id
    ).count()
    
    # Get recent trainings
    recent_trainings = TrainingHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(
        TrainingHistory.created_at.desc()
    ).limit(10).all()
    
    # Calculate success rate
    total_trainings = TrainingHistory.query.filter_by(user_id=current_user.id).count()
    successful_trainings = TrainingHistory.query.filter_by(
        user_id=current_user.id,
        status='completed'
    ).count()
    success_rate = (successful_trainings / total_trainings * 100) if total_trainings > 0 else 0
    
    # Récupérer la licence active
    active_license = current_user.get_active_license()
    
    stats = {
        'total_models': total_models,
        'total_predictions': total_predictions,
        'success_rate': round(success_rate, 2),
        'license_type': active_license.license_type if active_license else 'None',
        'license_days_left': active_license.days_remaining() if active_license else 0,
        'has_license': active_license is not None
    }
    
    return render_template('dashboard/index.html', stats=stats, recent_trainings=recent_trainings)


@dashboard_bp.route('/upload')
@login_required
def upload_page():
    """Dataset upload page"""
    return render_template('dashboard/upload.html')


@dashboard_bp.route('/models')
@login_required
def models_page():
    """User models listing page"""
    # Get all completed trainings (saved models)
    models = TrainingHistory.query.filter_by(
        user_id=current_user.id,
        status='completed'
    ).order_by(
        TrainingHistory.created_at.desc()
    ).all()
    
    return render_template('dashboard/models.html', models=models)


@dashboard_bp.route('/model/<int:model_id>')
@login_required
def model_detail(model_id):
    """Model detail page"""
    model = TrainingHistory.query.filter_by(
        id=model_id,
        user_id=current_user.id
    ).first_or_404()
    
    return render_template('dashboard/model_detail.html', model=model)


@dashboard_bp.route('/predict')
@login_required
def predict_page():
    """Prediction page"""
    # Get user's trained models
    models = TrainingHistory.query.filter_by(
        user_id=current_user.id,
        status='completed'
    ).order_by(
        TrainingHistory.created_at.desc()
    ).all()
    
    return render_template('dashboard/predict.html', models=models)


@dashboard_bp.route('/history')
@login_required
def history_page():
    """Training history page"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    pagination = TrainingHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(
        TrainingHistory.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('dashboard/history.html', pagination=pagination)


@dashboard_bp.route('/download/<int:model_id>/predictions')
@login_required
def download_predictions(model_id):
    """Download predictions CSV"""
    model = TrainingHistory.query.filter_by(
        id=model_id,
        user_id=current_user.id
    ).first_or_404()
    
    if not model.result_file:
        return jsonify({'error': 'Aucun fichier de résultats disponible'}), 404
    
    result_path = Path(model.result_file)
    if not result_path.exists():
        return jsonify({'error': 'Fichier de résultats introuvable'}), 404
    
    return send_file(
        result_path,
        as_attachment=True,
        download_name=f'predictions_{model.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )


@dashboard_bp.route('/download/<int:model_id>/model')
@login_required
def download_model(model_id):
    """Download trained model (ZIP)"""
    model = TrainingHistory.query.filter_by(
        id=model_id,
        user_id=current_user.id
    ).first_or_404()
    
    if not model.model_path:
        return jsonify({'error': 'Aucun modèle disponible'}), 404
    
    model_dir = Path(model.model_path)
    if not model_dir.exists():
        return jsonify({'error': 'Modèle introuvable'}), 404
    
    # Create ZIP file
    import shutil
    zip_path = model_dir.parent / f"{model.model_name}.zip"
    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', model_dir)
    
    return send_file(
        zip_path,
        as_attachment=True,
        download_name=f'{model.model_name}.zip'
    )


@dashboard_bp.route('/delete/<int:model_id>', methods=['POST'])
@login_required
def delete_model(model_id):
    """Delete a trained model"""
    try:
        model = TrainingHistory.query.filter_by(
            id=model_id,
            user_id=current_user.id
        ).first_or_404()
        
        # Delete model files
        if model.model_path:
            model_dir = Path(model.model_path)
            if model_dir.exists():
                shutil.rmtree(model_dir)
        
        # Delete database record
        db.session.delete(model)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Modèle supprimé avec succès'})
    except Exception as e:
        db.session.rollback()
        print(f"Erreur lors de la suppression du modèle {model_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


@dashboard_bp.route('/download/<filename>')
@login_required
def download_file(filename):
    """Download prediction results file"""
    try:
        upload_folder = Path(current_app.config['UPLOAD_FOLDER'])
        file_path = upload_folder / filename
        
        if not file_path.exists():
            return jsonify({'error': 'Fichier introuvable'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@dashboard_bp.route('/settings')
@login_required
def settings_page():
    """User settings page"""
    return render_template('dashboard/settings.html')
