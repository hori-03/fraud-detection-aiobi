"""Script pour initialiser la base de données et créer un utilisateur de démonstration."""

import sys
from pathlib import Path

# Ajouter le dossier racine au path pour les imports
sys.path.insert(0, str(Path(__file__).parent))

from app import create_app, db
from app.models.user import User
from app.models.license import License
from datetime import datetime, timedelta

def init_database():
    """Initialise la base de données et crée les tables."""
    app = create_app('development')
    
    with app.app_context():
        # Créer toutes les tables
        print("Création des tables de base de données...")
        db.create_all()
        print("✓ Tables créées avec succès")
        
        # Vérifier si un utilisateur existe déjà
        existing_user = User.query.filter_by(email='demo@example.com').first()
        if existing_user:
            print("✓ L'utilisateur de démonstration existe déjà")
            return
        
        # Créer un utilisateur de démonstration
        print("Création de l'utilisateur de démonstration...")
        demo_user = User(
            email='demo@example.com',
            username='demo',
            first_name='Demo',
            last_name='User',
            company='Demo Company'
        )
        demo_user.set_password('demo123')
        
        # Créer une licence d'essai NON ASSIGNÉE
        trial_license = License(
            user_id=None,  # Pas encore assignée
            license_type='trial',
            license_key='TRIAL-DEMO-2024',
            max_models=3,
            max_datasets_size_mb=50,
            is_active=False,  # Sera activée lors de l'entrée de la clé
            expires_at=None  # Sera définie lors de l'activation
        )
        
        db.session.add(demo_user)
        db.session.add(trial_license)
        db.session.commit()
        
        print("✓ Utilisateur de démonstration créé")
        print("\nInformations de connexion:")
        print("  Email: demo@example.com")
        print("  Mot de passe: demo123")
        print("\nActivation de licence:")
        print("  Clé de licence: TRIAL-DEMO-2024")
        print("  Type: trial (14 jours après activation)")
        print("\n✅ Base de données initialisée avec succès!")

if __name__ == '__main__':
    init_database()
