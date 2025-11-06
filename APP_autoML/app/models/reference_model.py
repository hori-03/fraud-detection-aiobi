"""
Modèle ReferenceModel - Modèles de référence (BACKOFFICE)

Table contenant les ~40 modèles pré-entraînés (Dataset1-40) utilisés pour:
- Auto-match sur datasets non étiquetés
- Ensemble predictions (top-3 modèles similaires)
- Transfert learning

⚠️ IMPORTANT: Cette table est INTERNE (backoffice uniquement)
Les utilisateurs ne voient PAS ces modèles.
Ils obtiennent leur propre modèle XGBoost après training.
"""

from datetime import datetime
from app import db


class ReferenceModel(db.Model):
    """Modèles de référence pré-entraînés (BACKOFFICE)"""
    
    __tablename__ = 'reference_models'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Identification du modèle
    model_name = db.Column(db.String(100), nullable=False, unique=True, index=True)  # Ex: "Dataset27"
    model_path = db.Column(db.String(500), nullable=False)  # Chemin local: data/automl_models/Dataset27/
    
    # 🚀 PRODUCTION: Stockage cloud (S3/GCS)
    s3_bucket = db.Column(db.String(200))  # Ex: "fraud-detection-models"
    s3_prefix = db.Column(db.String(500))  # Ex: "automl_models/dataset27/"
    storage_type = db.Column(db.String(20), default='local')  # 'local', 's3', 'gcs', 'azure'
    
    # Informations dataset d'entraînement
    dataset_size = db.Column(db.Integer)  # Nombre de lignes
    num_features = db.Column(db.Integer)  # Nombre de features originales
    num_engineered_features = db.Column(db.Integer)  # Features après engineering
    fraud_rate = db.Column(db.Float)  # Taux de fraude (%)
    
    # Colonnes du dataset (pour matching sémantique)
    column_names = db.Column(db.Text)  # JSON array: ["tx_id", "amount", "merchant", ...]
    column_types = db.Column(db.Text)  # JSON dict: {"amount": "float", "merchant": "str", ...}
    
    # === MÉTADONNÉES COMPLÈTES DE dataset_metadata.json ===
    # Structure de base
    numerical_cols = db.Column(db.Text)  # JSON array: colonnes numériques
    categorical_cols = db.Column(db.Text)  # JSON array: colonnes catégorielles
    n_numerical = db.Column(db.Integer)  # Nombre de colonnes numériques
    n_categorical = db.Column(db.Integer)  # Nombre de colonnes catégorielles
    
    # Features disponibles (pour matching sémantique)
    has_amount = db.Column(db.Boolean)
    has_timestamp = db.Column(db.Boolean)
    has_merchant = db.Column(db.Boolean)
    has_card = db.Column(db.Boolean)
    has_currency = db.Column(db.Boolean)
    has_country = db.Column(db.Boolean)
    has_balance = db.Column(db.Boolean)
    has_customer = db.Column(db.Boolean)
    has_account = db.Column(db.Boolean)
    
    # Features temporelles
    temporal_features = db.Column(db.Text)  # JSON: {"has_date": true, "has_time": false, ...}
    
    # Patterns des montants
    amount_patterns = db.Column(db.Text)  # JSON: statistiques des colonnes de montants
    
    # Patterns catégoriels
    categorical_patterns = db.Column(db.Text)  # JSON: cardinalité, valeurs fréquentes
    
    # Signature du dataset (pour comparaison rapide)
    signature = db.Column(db.Text)  # JSON: {"col_hash": 123456, "domain": "banking", ...}
    
    # Métriques de performance
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    roc_auc = db.Column(db.Float)
    
    # Hyperparamètres XGBoost (pour similarité)
    hyperparameters = db.Column(db.Text)  # JSON des hyperparams
    
    # Feature importance (pour matching avancé)
    feature_importance = db.Column(db.Text)  # JSON: {"amount": 0.35, "merchant": 0.18, ...}
    
    # Engineering flags appliqués
    engineering_methods = db.Column(db.Text)  # JSON: {"polynomial": true, "interaction": false, ...}
    
    # Métadonnées pour matching
    domain = db.Column(db.String(100))  # "banking", "e-commerce", "insurance", "telecom"
    data_quality = db.Column(db.String(50))  # "high", "medium", "low"
    imbalance_ratio = db.Column(db.Float)  # Ratio fraud/normal (ex: 0.02 = 2% fraude)
    
    # Statut et gestion
    is_active = db.Column(db.Boolean, default=True, index=True)  # Modèle actif/désactivé
    version = db.Column(db.String(50))  # Version du modèle (ex: "1.0", "2.1")
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_used_at = db.Column(db.DateTime)  # Dernière utilisation pour stats
    
    # Statistiques d'utilisation
    usage_count = db.Column(db.Integer, default=0)  # Nombre d'utilisations
    avg_similarity_score = db.Column(db.Float)  # Score moyen de similarité quand utilisé
    
    # Notes administrateur
    description = db.Column(db.Text)  # Description du modèle
    tags = db.Column(db.String(500))  # Tags séparés par virgules: "fraud,banking,africa"
    
    def __repr__(self):
        return f'<ReferenceModel {self.model_name} - {self.domain}>'
    
    @property
    def metrics(self):
        """Retourne les métriques sous forme de dict"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc
        }
    
    def increment_usage(self, similarity_score: float = None):
        """Incrémente le compteur d'utilisation et met à jour les stats"""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()
        
        if similarity_score is not None:
            # Mise à jour moyenne glissante du score de similarité
            if self.avg_similarity_score is None:
                self.avg_similarity_score = similarity_score
            else:
                # Moyenne pondérée (70% ancien, 30% nouveau)
                self.avg_similarity_score = 0.7 * self.avg_similarity_score + 0.3 * similarity_score
        
        db.session.commit()
    
    def to_dict(self):
        """Conversion en dictionnaire (pour API)"""
        import json
        
        return {
            'id': self.id,
            'model_name': self.model_name,
            'model_path': self.model_path,
            'dataset_size': self.dataset_size,
            'num_features': self.num_features,
            'fraud_rate': self.fraud_rate,
            'column_names': json.loads(self.column_names) if self.column_names else [],
            'metrics': self.metrics,
            'domain': self.domain,
            'is_active': self.is_active,
            'usage_count': self.usage_count,
            'avg_similarity_score': self.avg_similarity_score
        }
    
    @staticmethod
    def get_active_models(domain: str = None, min_roc_auc: float = 0.90):
        """
        Récupère les modèles actifs, optionnellement filtrés par domaine
        
        Args:
            domain: Filtre par domaine (banking, e-commerce, etc.)
            min_roc_auc: ROC-AUC minimum requis
        
        Returns:
            Liste de ReferenceModel
        """
        query = ReferenceModel.query.filter_by(is_active=True)
        
        if domain:
            query = query.filter_by(domain=domain)
        
        if min_roc_auc:
            query = query.filter(ReferenceModel.roc_auc >= min_roc_auc)
        
        return query.order_by(ReferenceModel.roc_auc.desc()).all()
    
    @staticmethod
    def find_best_match(column_names: list, dataset_size: int = None, fraud_rate: float = None):
        """
        Trouve le meilleur modèle de référence basé sur la similarité des colonnes
        
        Args:
            column_names: Liste des noms de colonnes du dataset utilisateur
            dataset_size: Taille du dataset (optionnel)
            fraud_rate: Taux de fraude (optionnel)
        
        Returns:
            ReferenceModel le plus similaire
        """
        import json
        from utils.column_matcher import ColumnMatcher
        
        active_models = ReferenceModel.get_active_models()
        matcher = ColumnMatcher()
        
        best_match = None
        best_score = 0
        
        for model in active_models:
            if not model.column_names:
                continue
            
            model_columns = json.loads(model.column_names)
            
            # Calculer similarité sémantique
            result = matcher.calculate_semantic_similarity(column_names, model_columns)
            similarity = result['similarity']
            
            # Bonus si taille similaire
            if dataset_size and model.dataset_size:
                size_ratio = min(dataset_size, model.dataset_size) / max(dataset_size, model.dataset_size)
                similarity *= (0.9 + 0.1 * size_ratio)  # Bonus jusqu'à +10%
            
            # Bonus si fraud_rate similaire
            if fraud_rate and model.fraud_rate:
                rate_diff = abs(fraud_rate - model.fraud_rate)
                if rate_diff < 0.05:  # Différence < 5%
                    similarity *= 1.05  # +5% bonus
            
            if similarity > best_score:
                best_score = similarity
                best_match = model
        
        return best_match, best_score
