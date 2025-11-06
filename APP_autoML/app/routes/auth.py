"""
Authentication routes for user login, registration, and logout
"""
from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash
from app import db
from app.models.user import User
from app.models.license import License
from datetime import datetime, timedelta
import os
import json
import secrets
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from pip._vendor import requests

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


def generate_license_key(prefix='TRIAL'):
    """Générer une clé de licence unique"""
    random_part = secrets.token_hex(8).upper()  # 16 caractères hexadécimaux
    return f"{prefix}-{random_part[:4]}-{random_part[4:8]}-{random_part[8:12]}-{random_part[12:]}"


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember', False)
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            # Connexion réussie (pas de vérification de licence ici)
            login_user(user, remember=remember)
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('dashboard.index'))
        else:
            flash('Email ou mot de passe incorrect', 'error')
    
    return render_template('auth/login.html')


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        company = request.form.get('company', '')
        
        # Validation
        if not all([username, email, password, confirm_password]):
            flash('Tous les champs sont requis', 'error')
            return redirect(url_for('auth.register'))
        
        if password != confirm_password:
            flash('Les mots de passe ne correspondent pas', 'error')
            return redirect(url_for('auth.register'))
        
        if len(password) < 8:
            flash('Le mot de passe doit contenir au moins 8 caractères', 'error')
            return redirect(url_for('auth.register'))
        
        # Check if user exists
        if User.query.filter_by(email=email).first():
            flash('Cet email est déjà utilisé', 'error')
            return redirect(url_for('auth.register'))
        
        if User.query.filter_by(username=username).first():
            flash('Ce nom d\'utilisateur est déjà utilisé', 'error')
            return redirect(url_for('auth.register'))
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            company=company
        )
        new_user.set_password(password)
        
        # Generate unique license key
        import secrets
        license_key = f"TRIAL-{secrets.token_hex(8).upper()}"
        
        # Create trial license (14 days)
        trial_license = License(
            user=new_user,
            license_type='trial',
            expires_at=datetime.utcnow() + timedelta(days=14),
            max_models=3,
            license_key=license_key
        )
        
        db.session.add(new_user)
        db.session.add(trial_license)
        db.session.commit()
        
        flash('Inscription réussie ! Vous avez 14 jours de trial gratuit.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/register.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('Vous avez été déconnecté avec succès', 'success')
    return redirect(url_for('auth.login'))


@auth_bp.route('/profile')
@login_required
def profile():
    """User profile page"""
    return render_template('auth/profile.html', user=current_user)


@auth_bp.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    """Update user profile"""
    username = request.form.get('username')
    company = request.form.get('company', '')
    
    if username and username != current_user.username:
        # Check if username is available
        if User.query.filter_by(username=username).first():
            flash('Ce nom d\'utilisateur est déjà utilisé', 'error')
            return redirect(url_for('auth.profile'))
        current_user.username = username
    
    current_user.company = company
    db.session.commit()
    
    flash('Profil mis à jour avec succès', 'success')
    return redirect(url_for('auth.profile'))


@auth_bp.route('/change-password', methods=['POST'])
@login_required
def change_password():
    """Change user password"""
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    if not current_user.check_password(current_password):
        flash('Mot de passe actuel incorrect', 'error')
        return redirect(url_for('auth.profile'))
    
    if new_password != confirm_password:
        flash('Les nouveaux mots de passe ne correspondent pas', 'error')
        return redirect(url_for('auth.profile'))
    
    if len(new_password) < 8:
        flash('Le mot de passe doit contenir au moins 8 caractères', 'error')
        return redirect(url_for('auth.profile'))
    
    current_user.set_password(new_password)
    db.session.commit()
    
    flash('Mot de passe changé avec succès', 'success')
    return redirect(url_for('auth.profile'))


@auth_bp.route('/license', methods=['GET', 'POST'])
@login_required
def license_activation():
    """Page d'activation de licence"""
    if request.method == 'POST':
        license_key = request.form.get('license_key', '').strip()
        
        if not license_key:
            flash('Veuillez entrer une clé de licence', 'error')
            return redirect(url_for('auth.license_activation'))
        
        # Vérifier si la clé existe et n'est pas déjà utilisée
        license_obj = License.query.filter_by(license_key=license_key).first()
        
        if not license_obj:
            flash('Clé de licence invalide', 'error')
            return redirect(url_for('auth.license_activation'))
        
        if license_obj.user_id and license_obj.user_id != current_user.id:
            flash('Cette clé de licence est déjà utilisée', 'error')
            return redirect(url_for('auth.license_activation'))
        
        # Désactiver toutes les anciennes licences de l'utilisateur
        old_licenses = License.query.filter_by(user_id=current_user.id, is_active=True).all()
        for old_license in old_licenses:
            old_license.is_active = False
        
        # Activer la licence pour cet utilisateur
        license_obj.user_id = current_user.id
        license_obj.is_active = True
        license_obj.created_at = datetime.utcnow()
        
        # Définir la date d'expiration selon le type
        if license_obj.license_type == 'trial':
            license_obj.expires_at = datetime.utcnow() + timedelta(days=14)
        elif license_obj.license_type == 'basic':
            license_obj.expires_at = datetime.utcnow() + timedelta(days=30)
        elif license_obj.license_type == 'premium':
            license_obj.expires_at = datetime.utcnow() + timedelta(days=365)
        else:  # enterprise
            license_obj.expires_at = None  # Illimité
        
        db.session.commit()
        
        flash(f'Licence {license_obj.license_type.upper()} activée avec succès !', 'success')
        return redirect(url_for('dashboard.index'))
    
    # GET: Afficher le formulaire
    active_license = current_user.get_active_license()
    return render_template('auth/license.html', active_license=active_license)


# ========================================
# 🔐 GOOGLE OAUTH2 AUTHENTICATION
# ========================================

@auth_bp.route('/google/login')
def google_login():
    """Redirection vers Google OAuth"""
    google_client_id = os.environ.get('GOOGLE_CLIENT_ID')
    redirect_uri = os.environ.get('GOOGLE_REDIRECT_URI', 'http://127.0.0.1:5000/auth/google/callback')
    
    if not google_client_id:
        flash('Google OAuth n\'est pas configuré', 'error')
        return redirect(url_for('auth.login'))
    
    # URL d'autorisation Google
    google_auth_url = (
        'https://accounts.google.com/o/oauth2/v2/auth?'
        f'client_id={google_client_id}&'
        f'redirect_uri={redirect_uri}&'
        'response_type=code&'
        'scope=openid%20email%20profile&'
        'access_type=offline&'
        'prompt=select_account'
    )
    
    return redirect(google_auth_url)


@auth_bp.route('/google/callback')
def google_callback():
    """Callback Google OAuth - Traitement du code d'autorisation"""
    code = request.args.get('code')
    error = request.args.get('error')
    
    if error:
        flash(f'Erreur Google OAuth: {error}', 'error')
        return redirect(url_for('auth.login'))
    
    if not code:
        flash('Code d\'autorisation manquant', 'error')
        return redirect(url_for('auth.login'))
    
    try:
        # Échanger le code contre un access token
        google_client_id = os.environ.get('GOOGLE_CLIENT_ID')
        google_client_secret = os.environ.get('GOOGLE_CLIENT_SECRET')
        redirect_uri = os.environ.get('GOOGLE_REDIRECT_URI', 'http://127.0.0.1:5000/auth/google/callback')
        
        # Debug: Afficher les credentials (masquer le secret partiellement)
        print(f"🔍 DEBUG - Client ID: {google_client_id}")
        print(f"🔍 DEBUG - Client Secret: {google_client_secret[:10]}...{google_client_secret[-5:] if google_client_secret else 'None'}")
        print(f"🔍 DEBUG - Redirect URI: {redirect_uri}")
        
        token_url = 'https://oauth2.googleapis.com/token'
        token_data = {
            'code': code,
            'client_id': google_client_id,
            'client_secret': google_client_secret,
            'redirect_uri': redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        token_response = requests.post(token_url, data=token_data)
        token_json = token_response.json()
        
        print(f"🔍 DEBUG - Token response: {token_json}")
        
        if 'error' in token_json:
            flash(f'Erreur lors de l\'échange du token: {token_json["error"]}', 'error')
            return redirect(url_for('auth.login'))
        
        # Vérifier et décoder l'ID token
        id_token_jwt = token_json.get('id_token')
        
        try:
            # Vérifier l'ID token avec tolérance de temps (clock_skew_in_seconds)
            idinfo = id_token.verify_oauth2_token(
                id_token_jwt, 
                google_requests.Request(), 
                google_client_id,
                clock_skew_in_seconds=10  # Tolérance de 10 secondes pour l'horloge
            )
            
            # Extraire les informations utilisateur
            google_id = idinfo['sub']
            email = idinfo.get('email')
            name = idinfo.get('name', '')
            given_name = idinfo.get('given_name', '')
            family_name = idinfo.get('family_name', '')
            
            # Vérifier si l'utilisateur existe déjà
            user = User.query.filter_by(email=email).first()
            
            if user:
                # Utilisateur existant - mettre à jour google_id si nécessaire
                if not user.google_id:
                    user.google_id = google_id
                    db.session.commit()
                
                # Connexion
                login_user(user, remember=True)
                user.last_login = datetime.utcnow()
                db.session.commit()
                
                flash(f'Bienvenue {user.first_name or user.username} !', 'success')
                return redirect(url_for('dashboard.index'))
            
            else:
                # Nouvel utilisateur - créer un compte
                username = email.split('@')[0]
                
                # Vérifier si le username existe déjà
                counter = 1
                original_username = username
                while User.query.filter_by(username=username).first():
                    username = f"{original_username}{counter}"
                    counter += 1
                
                new_user = User(
                    email=email,
                    username=username,
                    google_id=google_id,
                    first_name=given_name,
                    last_name=family_name,
                    is_active=True
                )
                
                # Pas de mot de passe pour les comptes Google
                new_user.password_hash = None
                
                # Créer une licence trial par défaut (EXACTEMENT comme l'inscription normale)
                license_key = f"TRIAL-{secrets.token_hex(8).upper()}"
                
                trial_license = License(
                    user=new_user,  # Utiliser la relation user (comme l'inscription)
                    license_type='trial',
                    expires_at=datetime.utcnow() + timedelta(days=14),
                    max_models=3,
                    license_key=license_key
                )
                
                # Ajouter user ET licence en même temps (comme l'inscription normale)
                db.session.add(new_user)
                db.session.add(trial_license)
                db.session.commit()
                
                print(f"✅ User and License created: {new_user.email} - License: {license_key}")
                
                # Connexion automatique
                login_user(new_user, remember=True)
                new_user.last_login = datetime.utcnow()
                db.session.commit()
                
                flash(f'Bienvenue {new_user.first_name} ! Votre compte a été créé avec une licence trial de 14 jours.', 'success')
                return redirect(url_for('dashboard.index'))
        
        except ValueError as e:
            flash(f'Token invalide: {str(e)}', 'error')
            return redirect(url_for('auth.login'))
    
    except Exception as e:
        flash(f'Erreur lors de l\'authentification Google: {str(e)}', 'error')
        return redirect(url_for('auth.login'))
