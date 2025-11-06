# -*- coding: utf-8 -*-
"""
Script de régénération des structures de datasets - VERSION v2.0 (18 features)

Ce script:
1. Régénère les fichiers structure pour Dataset1-Dataset30 (18 features)
2. Régénère les fichiers metamodel_training_examples avec 18 features
3. Affiche la progression

RETOUR À v2.0: Les 7 features Option A ont été supprimées

Usage:
    python regenerate_structures_with_new_features.py
"""

import subprocess
import sys
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Déterminer le répertoire racine du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent

def regenerate_structure(dataset_num):
    """Régénérer la structure d'un dataset"""
    dataset_name = f"Dataset{dataset_num}"
    dataset_path = PROJECT_ROOT / f"data/datasets/{dataset_name}.csv"
    
    if not Path(dataset_path).exists():
        print(f"⚠️  {dataset_name}.csv n'existe pas, skip")
        return False
    
    print(f"\n{'='*60}")
    print(f"📊 Régénération de la structure pour {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Lancer extract_structure.py avec encoding UTF-8
        extract_script = PROJECT_ROOT / 'base' / 'extract_structure.py'
        result = subprocess.run(
            [sys.executable, str(extract_script), str(dataset_path)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Remplace les caractères invalides
            timeout=300,  # 5 minutes max par dataset
            cwd=str(PROJECT_ROOT)  # Exécuter depuis la racine du projet
        )
        
        if result.returncode == 0:
            print(f"✅ Structure régénérée pour {dataset_name}")
            return True
        else:
            print(f"❌ Erreur pour {dataset_name}:")
            print(result.stderr)
            return False
    
    except subprocess.TimeoutExpired:
        print(f"⏱️  Timeout pour {dataset_name}")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue pour {dataset_name}: {e}")
        return False

def regenerate_metamodel_examples(dataset_num):
    """Régénérer les exemples metamodel d'un dataset"""
    dataset_name = f"Dataset{dataset_num}"
    
    print(f"\n🔄 Régénération des exemples metamodel pour {dataset_name}")
    
    try:
        # Lancer create_metamodel_examples.py avec encoding UTF-8
        metamodel_script = PROJECT_ROOT / 'base' / 'create_metamodel_examples.py'
        result = subprocess.run(
            [sys.executable, str(metamodel_script), dataset_name],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Remplace les caractères invalides
            timeout=300,
            cwd=str(PROJECT_ROOT)  # Exécuter depuis la racine du projet
        )
        
        if result.returncode == 0:
            print(f"✅ Exemples metamodel régénérés pour {dataset_name}")
            return True
        else:
            print(f"❌ Erreur exemples metamodel pour {dataset_name}:")
            print(result.stderr)
            return False
    
    except subprocess.TimeoutExpired:
        print(f"⏱️  Timeout exemples metamodel pour {dataset_name}")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue exemples metamodel pour {dataset_name}: {e}")
        return False

def main():
    print("="*70)
    print("🚀 RÉGÉNÉRATION DES STRUCTURES - RETOUR À v2.0 (18 FEATURES)")
    print("="*70)
    print()
    print("� Retour à la version v2.0:")
    print("   • Suppression des 7 features Option A (class_separation, silhouette, etc.)")
    print("   • 18 structure features originales uniquement")
    print("   • Architecture: input_dim = 38 (18 structure + 20 importance)")
    print()
    print("🎯 Raison: Les 7 features Option A dégradaient la performance")
    print("   Impact attendu: Val Loss 0.005 → <0.003 (40%+ amélioration)")
    print()
    
    # Datasets à régénérer (1-30)
    datasets = list(range(1, 31))
    
    success_count = 0
    failed_datasets = []
    
    for dataset_num in datasets:
        print(f"\n{'#'*70}")
        print(f"# DATASET {dataset_num}/30")
        print(f"{'#'*70}")
        
        # Étape 1: Régénérer structure
        structure_ok = regenerate_structure(dataset_num)
        
        if structure_ok:
            # Étape 2: Régénérer exemples metamodel
            examples_ok = regenerate_metamodel_examples(dataset_num)
            
            if examples_ok:
                success_count += 1
                print(f"✅ Dataset{dataset_num} régénéré avec succès!")
            else:
                failed_datasets.append(f"Dataset{dataset_num} (exemples metamodel)")
        else:
            failed_datasets.append(f"Dataset{dataset_num} (structure)")
    
    # Résumé final
    print("\n" + "="*70)
    print("📊 RÉSUMÉ DE LA RÉGÉNÉRATION")
    print("="*70)
    print(f"✅ Succès: {success_count}/{len(datasets)} datasets")
    print(f"❌ Échecs: {len(failed_datasets)}/{len(datasets)} datasets")
    
    if failed_datasets:
        print("\n⚠️  Datasets avec erreurs:")
    
    print("\n" + "="*70)
    print("🎯 PROCHAINES ÉTAPES")
    print("="*70)
    print("1. ✅ Structures régénérées avec 18 features (v2.0)")
    print("2. ✅ Exemples metamodel mis à jour")
    print("3. ⏭️  Lancer l'entraînement du modèle v2.0:")
    print("   python automl_transformer/train_automl_metatransformer.py")
    print("4. ⏭️  Valider performances (Val Loss HP ~0.006-0.008)")
    print("="*70)

if __name__ == "__main__":
    main()
