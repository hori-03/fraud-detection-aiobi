"""
Modèle License - Gestion des licences utilisateurs

Gère les licences d'accès à l'application AML.
"""

from datetime import datetime
from app import db


class License(db.Model):
    """Modèle de licence utilisateur"""
    
    __tablename__ = 'licenses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)  # NULL = non assignée
    
    # Type de licence
    license_type = db.Column(db.String(50), nullable=False)  # 'trial', 'basic', 'premium', 'enterprise'
    
    # Limites
    max_models = db.Column(db.Integer, default=10)  # Nombre max de modèles entraînables
    max_datasets_size_mb = db.Column(db.Integer, default=100)  # Taille max des datasets en MB
    
    # Statut
    is_active = db.Column(db.Boolean, default=True, index=True)
    
    # Dates
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=True)  # None = illimité
    
    # Clé de licence
    license_key = db.Column(db.String(100), unique=True, nullable=False)
    
    def is_valid(self):
        """Vérifier si la licence est valide"""
        if not self.is_active:
            return False
        
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        
        return True
    
    def days_remaining(self):
        """Nombre de jours restants avant expiration"""
        if not self.expires_at:
            return None  # Illimité
        
        delta = self.expires_at - datetime.utcnow()
        return max(0, delta.days)
    
    def __repr__(self):
        return f'<License {self.license_type} - User {self.user_id}>'
