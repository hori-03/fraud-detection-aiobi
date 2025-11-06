"""
Point d'entrée de l'application Flask AML AutoML

Ce fichier initialise et lance l'application Flask pour la détection de fraude
en utilisant le métamodèle AutoML existant (automl_transformer).
"""

import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour accéder à automl_transformer
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app

# Créer l'application Flask
app = create_app()

if __name__ == '__main__':
    # Récupérer le port depuis les variables d'environnement (Railway)
    port = int(os.environ.get('PORT', 5000))
    
    # En développement: debug=True, en production: debug=False
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    print(f"🚀 Démarrage de l'application AML AutoML sur le port {port}")
    print(f"📊 Mode: {'Développement' if debug else 'Production'}")
    print(f"🤖 AutoML: Utilise automl_transformer/full_automl.py")
    print(f"🔗 URL: http://localhost:{port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
