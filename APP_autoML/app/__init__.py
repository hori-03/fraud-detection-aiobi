"""
Application Factory pour Flask

Initialise l'application Flask avec toutes ses extensions et configurations.
"""

import os
from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from pathlib import Path
import json

# Extensions Flask
db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate()


def create_app(config_name=None):
    """
    Factory pour créer l'application Flask
    
    Args:
        config_name: Nom de la configuration ('development', 'production', 'testing')
    
    Returns:
        Application Flask configurée
    """
    app = Flask(__name__)
    
    # Charger la configuration
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    from config.config import config
    app.config.from_object(config[config_name])
    
    # Optimisations de performance
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # Cache statique 1 an
    app.config['TEMPLATES_AUTO_RELOAD'] = False if config_name == 'production' else True
    
    # Créer les dossiers nécessaires
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    Path(app.config['MODELS_FOLDER']).mkdir(parents=True, exist_ok=True)
    Path(app.config['LOG_FOLDER']).mkdir(parents=True, exist_ok=True)
    
    # Initialiser les extensions
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    
    # Ajouter un filtre Jinja2 pour parser JSON
    @app.template_filter('from_json')
    def from_json_filter(value):
        """Parse une chaîne JSON en objet Python"""
        if not value:
            return []
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    
    # Configuration Login Manager
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Veuillez vous connecter pour accéder à cette page.'
    login_manager.login_message_category = 'warning'
    
    # User loader pour Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        from app.models.user import User
        return User.query.get(int(user_id))
    
    # Enregistrer les blueprints (routes)
    from app.routes.auth import auth_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.api import api_bp
    from app.routes.admin import admin_bp
    
    app.register_blueprint(auth_bp)  # auth_bp a déjà url_prefix='/auth'
    app.register_blueprint(dashboard_bp)  # dashboard_bp n'a pas de prefix (sera à la racine)
    app.register_blueprint(api_bp)  # api_bp a déjà url_prefix='/api'
    app.register_blueprint(admin_bp)  # admin_bp a url_prefix='/admin'
    
    # Optimisation: Headers de cache pour améliorer les performances
    @app.after_request
    def add_header(response):
        """Ajoute des headers de cache pour optimiser le chargement"""
        # Cache pour les fichiers statiques (CSS, JS, images)
        if request.path.startswith('/static/'):
            response.cache_control.max_age = 31536000  # 1 an
            response.cache_control.public = True
        return response
    
    # Route racine
    @app.route('/')
    def index():
        from flask import redirect, url_for
        return redirect(url_for('dashboard.index'))
    
    # Importer les modèles AVANT de créer les tables
    from app.models import user, license, history, reference_model
    
    # Note: db.create_all() est désactivé car on utilise Flask-Migrate
    # Les migrations gèrent maintenant la création/modification des tables
    # with app.app_context():
    #     db.create_all()
    
    # Log de démarrage
    app.logger.info(f"Application démarrée en mode {config_name}")
    app.logger.info(f"AutoML models directory: {app.config['AUTOML_MODELS_DIR']}")
    
    return app
