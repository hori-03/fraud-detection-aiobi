"""
retrain_all_models.py

Réentraîne TOUS les modèles AutoML (40 datasets) avec la version actuelle du code.
Cela résout les problèmes de compatibilité avec ColumnMatcher et autres composants.

Usage:
    python retrain_all_models.py
    
Options:
    - Exécution séquentielle pour éviter les problèmes de mémoire
    - Sauvegarde dans data/automl_models/ (écrase les anciens)
    - Affichage de la progression
"""

import subprocess
import time
from pathlib import Path
import sys

# Déterminer le répertoire racine du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration
DATASETS_DIR = PROJECT_ROOT / "data/datasets"
AUTOML_SCRIPT = PROJECT_ROOT / "automl_transformer/full_automl.py"
START_DATASET = 1  # Commencer à Dataset1
END_DATASET = 40   # Finir à Dataset40

def retrain_all_models():
    """Réentraîne tous les modèles AutoML"""
    
    print("="*80)
    print("🚀 RÉENTRAÎNEMENT COMPLET DE TOUS LES MODÈLES AUTOML")
    print("="*80)
    print(f"\n📂 Datasets: {START_DATASET} à {END_DATASET}")
    print(f"📁 Dossier: {DATASETS_DIR}")
    print(f"🔧 Script: {AUTOML_SCRIPT}\n")
    
    # Vérifier que le script existe
    if not Path(AUTOML_SCRIPT).exists():
        print(f"❌ Erreur: {AUTOML_SCRIPT} introuvable")
        return
    
    successful = []
    failed = []
    skipped = []
    
    start_time = time.time()
    
    for i in range(START_DATASET, END_DATASET + 1):
        dataset_file = DATASETS_DIR / f"Dataset{i}.csv"
        
        print(f"\n{'='*80}")
        print(f"📊 Dataset {i}/{END_DATASET}: {dataset_file.name}")
        print(f"{'='*80}")
        
        # Vérifier que le dataset existe
        if not dataset_file.exists():
            print(f"⚠️  Fichier introuvable: {dataset_file}")
            skipped.append(i)
            continue
        
        # Construire la commande
        cmd = [
            sys.executable,  # python.exe
            str(AUTOML_SCRIPT),
            str(dataset_file)  # Chemin direct du dataset
        ]
        
        print(f"🔄 Lancement: {' '.join(cmd)}")
        
        dataset_start = time.time()
        
        try:
            # Exécuter le script
            result = subprocess.run(
                cmd,
                capture_output=False,  # Afficher la sortie en temps réel
                text=True,
                check=True,
                cwd=str(PROJECT_ROOT)  # Exécuter depuis la racine du projet
            )
            
            dataset_time = time.time() - dataset_start
            print(f"\n✅ Dataset{i} réentraîné avec succès en {dataset_time:.1f}s")
            successful.append(i)
            
        except subprocess.CalledProcessError as e:
            dataset_time = time.time() - dataset_start
            print(f"\n❌ Erreur lors du réentraînement de Dataset{i} (temps: {dataset_time:.1f}s)")
            print(f"   Code de retour: {e.returncode}")
            failed.append(i)
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interruption par l'utilisateur")
            print(f"   Datasets traités: {len(successful)}/{END_DATASET}")
            break
        
        except Exception as e:
            print(f"\n❌ Erreur inattendue: {e}")
            failed.append(i)
    
    # Résumé final
    total_time = time.time() - start_time
    
    print(f"\n\n{'='*80}")
    print(f"📊 RÉSUMÉ DU RÉENTRAÎNEMENT")
    print(f"{'='*80}")
    print(f"\n⏱️  Temps total: {total_time/60:.1f} minutes ({total_time:.0f}s)")
    print(f"\n✅ Succès: {len(successful)}/{END_DATASET}")
    if successful:
        print(f"   Datasets: {', '.join(f'Dataset{i}' for i in successful)}")
    
    if failed:
        print(f"\n❌ Échecs: {len(failed)}")
        print(f"   Datasets: {', '.join(f'Dataset{i}' for i in failed)}")
    
    if skipped:
        print(f"\n⚠️  Ignorés: {len(skipped)}")
        print(f"   Datasets: {', '.join(f'Dataset{i}' for i in skipped)}")
    
    print(f"\n{'='*80}")
    
    if len(successful) == END_DATASET:
        print("🎉 TOUS LES MODÈLES ONT ÉTÉ RÉENTRAÎNÉS AVEC SUCCÈS!")
    elif len(successful) > 0:
        print(f"⚠️  {len(successful)}/{END_DATASET} modèles réentraînés")
    else:
        print("❌ AUCUN MODÈLE RÉENTRAÎNÉ")
    
    print(f"{'='*80}\n")
    
    return successful, failed, skipped


if __name__ == "__main__":
    print("\n🚀 Démarrage du réentraînement complet des modèles AutoML...")
    print("⏱️  Temps estimé: ~2-3 heures pour 40 datasets\n")
    
    # Demander confirmation
    response = input("Continuer? (oui/non): ").strip().lower()
    
    if response in ['oui', 'o', 'y', 'yes']:
        print("\n🔥 C'est parti!\n")
        successful, failed, skipped = retrain_all_models()
        
        # Code de sortie
        if len(failed) > 0:
            sys.exit(1)  # Erreur
        else:
            sys.exit(0)  # Succès
    else:
        print("\n❌ Réentraînement annulé\n")
        sys.exit(0)
