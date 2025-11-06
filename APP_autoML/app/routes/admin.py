"""
Routes d'administration - Gestion des utilisateurs et licences

Accessible uniquement aux administrateurs (is_admin=True)
"""
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user
from functools import wraps
from sqlalchemy import func
from app import db
from app.models.user import User
from app.models.license import License
from app.models.history import TrainingHistory
from datetime import datetime, timedelta
import secrets

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


def admin_required(f):
    """Décorateur pour restreindre l'accès aux administrateurs"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Vous devez être connecté pour accéder à cette page', 'error')
            return redirect(url_for('auth.login'))
        
        if not current_user.is_admin:
            flash('Accès refusé. Cette page est réservée aux administrateurs.', 'error')
            return redirect(url_for('dashboard.index'))
        
        return f(*args, **kwargs)
    return decorated_function


@admin_bp.route('/')
@login_required
@admin_required
def index():
    """Page principale d'administration - Dashboard"""
    
    # Statistiques générales
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    total_licenses = License.query.count()
    active_licenses = License.query.filter_by(is_active=True).count()
    
    # Statistiques par type de licence
    license_stats = db.session.query(
        License.license_type,
        func.count(License.id)
    ).group_by(License.license_type).all()
    
    # Utilisateurs récents (7 derniers jours)
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_users = User.query.filter(User.created_at >= week_ago).count()
    
    # Historique d'entraînement (30 derniers jours)
    month_ago = datetime.utcnow() - timedelta(days=30)
    recent_trainings = TrainingHistory.query.filter(
        TrainingHistory.created_at >= month_ago
    ).count()
    
    # Licences expirées
    expired_licenses = License.query.filter(
        License.expires_at < datetime.utcnow(),
        License.is_active == True
    ).count()
    
    return render_template('admin/index.html',
                         total_users=total_users,
                         active_users=active_users,
                         total_licenses=total_licenses,
                         active_licenses=active_licenses,
                         license_stats=license_stats,
                         recent_users=recent_users,
                         recent_trainings=recent_trainings,
                         expired_licenses=expired_licenses)


@admin_bp.route('/users')
@login_required
@admin_required
def users():
    """Gestion des utilisateurs"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # Filtres
    search = request.args.get('search', '')
    status = request.args.get('status', 'all')  # all, active, inactive
    
    query = User.query
    
    # Appliquer les filtres
    if search:
        query = query.filter(
            (User.email.ilike(f'%{search}%')) |
            (User.username.ilike(f'%{search}%')) |
            (User.first_name.ilike(f'%{search}%')) |
            (User.last_name.ilike(f'%{search}%'))
        )
    
    if status == 'active':
        query = query.filter_by(is_active=True)
    elif status == 'inactive':
        query = query.filter_by(is_active=False)
    
    # Pagination
    users = query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('admin/users.html', users=users, search=search, status=status)


@admin_bp.route('/users/<int:user_id>')
@login_required
@admin_required
def user_detail(user_id):
    """Détails d'un utilisateur"""
    user = User.query.get_or_404(user_id)
    
    # Récupérer les licences de l'utilisateur
    licenses = License.query.filter_by(user_id=user_id).order_by(License.created_at.desc()).all()
    
    # Récupérer l'historique d'entraînement
    trainings = TrainingHistory.query.filter_by(user_id=user_id).order_by(
        TrainingHistory.created_at.desc()
    ).limit(10).all()
    
    return render_template('admin/user_detail.html', user=user, licenses=licenses, trainings=trainings)


@admin_bp.route('/users/<int:user_id>/toggle-status', methods=['POST'])
@login_required
@admin_required
def toggle_user_status(user_id):
    """Activer/Désactiver un utilisateur"""
    user = User.query.get_or_404(user_id)
    
    # Empêcher de se désactiver soi-même
    if user.id == current_user.id:
        flash('Vous ne pouvez pas désactiver votre propre compte', 'error')
        return redirect(url_for('admin.user_detail', user_id=user_id))
    
    user.is_active = not user.is_active
    db.session.commit()
    
    status = 'activé' if user.is_active else 'désactivé'
    flash(f'Utilisateur {user.email} {status} avec succès', 'success')
    
    return redirect(url_for('admin.user_detail', user_id=user_id))


@admin_bp.route('/users/<int:user_id>/toggle-admin', methods=['POST'])
@login_required
@admin_required
def toggle_admin_status(user_id):
    """Promouvoir/Rétrograder un administrateur"""
    user = User.query.get_or_404(user_id)
    
    # Empêcher de se rétrograder soi-même
    if user.id == current_user.id:
        flash('Vous ne pouvez pas modifier votre propre statut administrateur', 'error')
        return redirect(url_for('admin.user_detail', user_id=user_id))
    
    user.is_admin = not user.is_admin
    db.session.commit()
    
    status = 'administrateur' if user.is_admin else 'utilisateur standard'
    flash(f'{user.email} est maintenant {status}', 'success')
    
    return redirect(url_for('admin.user_detail', user_id=user_id))


@admin_bp.route('/users/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    """Supprimer un utilisateur"""
    user = User.query.get_or_404(user_id)
    
    # Empêcher de se supprimer soi-même
    if user.id == current_user.id:
        flash('Vous ne pouvez pas supprimer votre propre compte', 'error')
        return redirect(url_for('admin.user_detail', user_id=user_id))
    
    email = user.email
    db.session.delete(user)
    db.session.commit()
    
    flash(f'Utilisateur {email} supprimé avec succès', 'success')
    return redirect(url_for('admin.users'))


@admin_bp.route('/licenses')
@login_required
@admin_required
def licenses():
    """Gestion des licences"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # Filtres
    license_type = request.args.get('type', 'all')  # all, trial, basic, premium, enterprise
    status = request.args.get('status', 'all')  # all, active, expired
    
    query = License.query
    
    # Appliquer les filtres
    if license_type != 'all':
        query = query.filter_by(license_type=license_type)
    
    if status == 'active':
        query = query.filter_by(is_active=True)
    elif status == 'expired':
        query = query.filter(
            License.expires_at < datetime.utcnow(),
            License.is_active == True
        )
    
    # Pagination
    licenses = query.order_by(License.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('admin/licenses.html', licenses=licenses, license_type=license_type, status=status)


@admin_bp.route('/licenses/<int:license_id>/toggle-status', methods=['POST'])
@login_required
@admin_required
def toggle_license_status(license_id):
    """Activer/Désactiver une licence"""
    license = License.query.get_or_404(license_id)
    
    license.is_active = not license.is_active
    db.session.commit()
    
    status = 'activée' if license.is_active else 'désactivée'
    flash(f'Licence {license.license_key} {status} avec succès', 'success')
    
    return redirect(url_for('admin.licenses'))


@admin_bp.route('/licenses/<int:license_id>/extend', methods=['POST'])
@login_required
@admin_required
def extend_license(license_id):
    """Prolonger une licence"""
    license = License.query.get_or_404(license_id)
    days = request.form.get('days', type=int, default=30)
    
    if license.expires_at:
        # Si la licence est déjà expirée, partir d'aujourd'hui
        if license.expires_at < datetime.utcnow():
            license.expires_at = datetime.utcnow() + timedelta(days=days)
        else:
            # Sinon ajouter au délai existant
            license.expires_at += timedelta(days=days)
    else:
        # Licence illimitée -> ajouter une expiration
        license.expires_at = datetime.utcnow() + timedelta(days=days)
    
    license.is_active = True
    db.session.commit()
    
    flash(f'Licence prolongée de {days} jours', 'success')
    return redirect(url_for('admin.licenses'))


@admin_bp.route('/licenses/create', methods=['GET', 'POST'])
@login_required
@admin_required
def create_license():
    """Créer une nouvelle licence"""
    if request.method == 'POST':
        user_id = request.form.get('user_id', type=int)
        license_type = request.form.get('license_type')
        days = request.form.get('days', type=int)
        
        # Générer une clé de licence
        prefix = license_type.upper()
        license_key = f"{prefix}-{secrets.token_hex(8).upper()}"
        
        # Définir les limites selon le type
        limits = {
            'trial': {'max_models': 3, 'max_datasets_size_mb': 100},
            'basic': {'max_models': 10, 'max_datasets_size_mb': 500},
            'premium': {'max_models': 50, 'max_datasets_size_mb': 2000},
            'enterprise': {'max_models': 999, 'max_datasets_size_mb': 10000}
        }
        
        new_license = License(
            user_id=user_id if user_id else None,
            license_type=license_type,
            license_key=license_key,
            max_models=limits[license_type]['max_models'],
            max_datasets_size_mb=limits[license_type]['max_datasets_size_mb'],
            is_active=True,
            expires_at=datetime.utcnow() + timedelta(days=days) if days else None
        )
        
        db.session.add(new_license)
        db.session.commit()
        
        flash(f'Licence {license_key} créée avec succès', 'success')
        
        # Afficher le formulaire avec la clé générée
        users = User.query.order_by(User.email).all()
        return render_template('admin/create_license.html', users=users, generated_key=license_key)
    
    # GET: Afficher le formulaire
    users = User.query.order_by(User.email).all()
    return render_template('admin/create_license.html', users=users)


@admin_bp.route('/stats')
@login_required
@admin_required
def stats():
    """Statistiques détaillées"""
    
    # Stats utilisateurs
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    
    # Stats licences
    total_licenses = License.query.count()
    active_licenses = License.query.filter_by(is_active=True).count()
    expired_licenses = License.query.filter(
        License.expires_at < datetime.utcnow(),
        License.is_active == False
    ).count()
    
    # Nouveaux utilisateurs (7 derniers jours)
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_users = User.query.filter(User.created_at >= week_ago).count()
    
    # Répartition des licences par type
    license_stats = db.session.query(
        License.license_type,
        func.count(License.id).label('count')
    ).group_by(License.license_type).all()
    
    # Statut des licences par type
    license_status_counts = {}
    for license_type, _ in license_stats:
        active = License.query.filter_by(license_type=license_type, is_active=True).count()
        expired = License.query.filter(
            License.license_type == license_type,
            License.expires_at < datetime.utcnow()
        ).count()
        license_status_counts[license_type] = {'active': active, 'expired': expired}
    
    # Évolution des inscriptions (30 derniers jours)
    registration_labels = []
    registration_counts = []
    for i in range(30, -1, -1):
        date = datetime.utcnow().date() - timedelta(days=i)
        next_date = date + timedelta(days=1)
        count = User.query.filter(
            User.created_at >= date,
            User.created_at < next_date
        ).count()
        registration_labels.append(date.strftime('%d/%m'))
        registration_counts.append(count)
    
    # Données pour le graphique de licences
    license_labels = []
    license_counts = []
    for license_type, count in license_stats:
        license_labels.append(license_type.upper())
        license_counts.append(count)
    
    return render_template('admin/stats.html',
                         total_users=total_users,
                         active_licenses=active_licenses,
                         expired_licenses=expired_licenses,
                         recent_users=recent_users,
                         total_licenses=total_licenses,
                         license_stats=license_stats,
                         license_status_counts=license_status_counts,
                         registration_labels=registration_labels,
                         registration_counts=registration_counts,
                         license_labels=license_labels,
                         license_counts=license_counts)
