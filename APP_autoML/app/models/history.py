"""
Modèle TrainingHistory - Historique des entraînements

Enregistre tous les entraînements de modèles effectués par les utilisateurs.
"""

from datetime import datetime
from app import db


class TrainingHistory(db.Model):
    """Historique des entraînements de modèles"""
    
    __tablename__ = 'training_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Informations dataset
    dataset_name = db.Column(db.String(200), nullable=False)
    dataset_size = db.Column(db.Integer)  # Nombre de lignes
    dataset_features = db.Column(db.Integer)  # Nombre de features
    fraud_rate = db.Column(db.Float)  # Taux de fraude (%)
    
    # Informations modèle
    model_name = db.Column(db.String(200), nullable=False)
    model_path = db.Column(db.String(500))  # Chemin vers le modèle sauvegardé
    
    # Métriques de performance
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    roc_auc = db.Column(db.Float)
    
    # Hyperparamètres utilisés (JSON string)
    hyperparameters = db.Column(db.Text)  # JSON des hyperparams
    
    # Feature engineering appliqué
    features_engineered = db.Column(db.Text)  # Liste des features créées
    
    # Temps d'exécution
    training_time_seconds = db.Column(db.Float)
    
    # Métadonnées
    automl_version = db.Column(db.String(50))  # Version de l'AutoML utilisée
    meta_transformer_used = db.Column(db.Boolean, default=True)  # Meta-Transformer ou règles
    
    # Statut
    status = db.Column(db.String(50), default='completed')  # 'completed', 'failed', 'running'
    error_message = db.Column(db.Text, nullable=True)
    
    # Dates
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def to_dict(self):
        """Convertir en dictionnaire pour JSON"""
        return {
            'id': self.id,
            'dataset_name': self.dataset_name,
            'dataset_size': self.dataset_size,
            'dataset_features': self.dataset_features,
            'fraud_rate': self.fraud_rate,
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'training_time_seconds': self.training_time_seconds,
            'meta_transformer_used': self.meta_transformer_used,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<TrainingHistory {self.model_name} - F1={self.f1_score:.4f}>'
