"""
Script de diagnostic pour les modèles ensemble
"""
import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from app import create_app, db
from app.models.history import TrainingHistory
import json

app = create_app()

with app.app_context():
    print("=" * 60)
    print("DIAGNOSTIC DES MODÈLES ENSEMBLE")
    print("=" * 60)
    
    # Trouver tous les modèles (vérifier ceux qui contiennent "ensemble" dans le nom)
    all_models = TrainingHistory.query.all()
    ensemble_models = [m for m in all_models if 'ensemble' in m.model_name.lower()]
    
    print(f"\n📊 Total de modèles dans la DB: {len(all_models)}")
    print(f"📊 Nombre de modèles ensemble trouvés: {len(ensemble_models)}\n")
    
    if not ensemble_models:
        print("❌ PROBLÈME: Aucun modèle ensemble dans la base de données!")
        print("\n💡 Solutions possibles:")
        print("1. Créer un modèle ensemble via l'interface web")
        print("2. Vérifier que le training a bien enregistré le modèle avec model_type='ensemble'")
    else:
        for i, model in enumerate(ensemble_models, 1):
            print(f"\n{'='*60}")
            print(f"Modèle Ensemble #{i}")
            print(f"{'='*60}")
            print(f"ID: {model.id}")
            print(f"Dataset: {model.dataset_name}")
            print(f"Date: {model.created_at}")
            print(f"Model Name: {model.model_name}")
            print(f"Model Path: {model.model_path}")
            
            # Vérifier hyperparameters
            if model.hyperparameters:
                try:
                    params = json.loads(model.hyperparameters) if isinstance(model.hyperparameters, str) else model.hyperparameters
                    print(f"\n📋 Hyperparameters:")
                    print(f"  - Clés disponibles: {list(params.keys())}")
                    
                    if 'ensemble_models' in params:
                        ensemble_info = params['ensemble_models']
                        print(f"\n✅ Ensemble models trouvés: {len(ensemble_info)}")
                        for model_name, model_data in ensemble_info.items():
                            print(f"  - {model_name}:")
                            print(f"    Type: {model_data.get('type', 'N/A')}")
                            print(f"    Path: {model_data.get('path', 'N/A')}")
                            
                            # Vérifier si le fichier existe
                            model_file = model_data.get('path', '')
                            if model_file:
                                full_path = os.path.join(app.config.get('AUTOML_MODELS_DIR', 'data/automl_models'), model_file)
                                exists = os.path.exists(full_path)
                                print(f"    Fichier existe: {'✅' if exists else '❌'} ({full_path})")
                    else:
                        print("\n❌ PROBLÈME: Clé 'ensemble_models' manquante dans hyperparameters!")
                        print(f"   Contenu actuel: {params}")
                        
                except Exception as e:
                    print(f"\n❌ Erreur lors du parsing des hyperparameters: {e}")
                    print(f"   Valeur brute: {model.hyperparameters[:200]}...")
            else:
                print("\n❌ PROBLÈME: Aucun hyperparameter enregistré!")
    
    print("\n" + "=" * 60)
    print("FIN DU DIAGNOSTIC")
    print("=" * 60)
