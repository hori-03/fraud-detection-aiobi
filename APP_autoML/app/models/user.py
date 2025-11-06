"""
Modèle User - Gestion des utilisateurs

Représente un utilisateur de l'application AML.
"""

from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db


class User(UserMixin, db.Model):
    """Modèle utilisateur pour l'authentification"""
    
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255))
    
    # OAuth Google (optionnel)
    google_id = db.Column(db.String(120), unique=True, nullable=True)
    
    # Informations utilisateur
    first_name = db.Column(db.String(80))
    last_name = db.Column(db.String(80))
    company = db.Column(db.String(120))
    
    # Statut
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    
    # Dates
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime)
    
    # Relations
    licenses = db.relationship('License', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    history = db.relationship('TrainingHistory', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hasher le mot de passe"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Vérifier le mot de passe"""
        return check_password_hash(self.password_hash, password)
    
    def get_active_license(self):
        """Récupérer la licence active (ou None)"""
        return self.licenses.filter_by(is_active=True).first()
    
    def has_valid_license(self):
        """Vérifier si l'utilisateur a une licence valide"""
        active_license = self.get_active_license()
        if not active_license:
            return False
        
        # Vérifier la date d'expiration
        if active_license.expires_at and active_license.expires_at < datetime.utcnow():
            active_license.is_active = False
            db.session.commit()
            return False
        
        return True
    
    def __repr__(self):
        return f'<User {self.username}>'
