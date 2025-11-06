"""
🔄 RÉENTRAÎNEMENT DES 40 MODÈLES AVEC META-TRANSFORMER
======================================================

Réentraîne tous les modèles existants pour ajouter:
- engineering_flags (pour feature engineering dynamique)
- meta_transformer_used (flag pour mode Meta-Transformer)
- feature_names (colonnes finales du modèle)

Les anciens modèles seront écrasés avec la nouvelle version.
"""

import sys
from pathlib import Path
import time
from datetime import datetime

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from automl_transformer.full_automl import FullAutoML
import pandas as pd
import json


def retrain_model(dataset_name: str, dataset_path: Path, output_dir: Path) -> dict:
    """
    Réentraîne un modèle avec Meta-Transformer
    
    Args:
        dataset_name: Nom du dataset (ex: 'dataset1')
        dataset_path: Chemin vers le CSV du dataset
        output_dir: Dossier de sortie du modèle
    
    Returns:
        dict avec résultats
    """
    print(f"\n{'='*80}")
    print(f"🔄 RÉENTRAÎNEMENT: {dataset_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Créer instance AutoML avec Meta-Transformer
        automl = FullAutoML(
            use_meta_transformer=True,
            use_feature_selector=True,
            feature_selector_mode='direct'
        )
        
        # Entraîner
        print(f"📊 Dataset: {dataset_path}")
        performance = automl.fit(str(dataset_path), target_col='is_fraud')
        
        # Sauvegarder le modèle
        automl.save_model(str(output_dir))
        
        training_time = time.time() - start_time
        
        return {
            'success': True,
            'dataset_name': dataset_name,
            'training_time': training_time,
            'test_f1': performance.get('test_f1', 0),
            'test_auc': performance.get('test_auc', 0),
            'n_features': performance.get('n_features', 0),
            'engineering_flags': performance.get('engineering_flags'),
            'meta_transformer_used': True
        }
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'dataset_name': dataset_name,
            'error': str(e)
        }


def main():
    """Réentraîne tous les 40 modèles"""
    
    print(f"\n{'='*80}")
    print(f"🚀 RÉENTRAÎNEMENT GLOBAL AVEC META-TRANSFORMER")
    print(f"{'='*80}")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Chemins
    datasets_dir = Path("data/datasets")
    models_output_dir = Path("data/automl_models")
    
    # Découvrir tous les datasets
    dataset_files = sorted(datasets_dir.glob("Dataset*.csv"))
    
    print(f"\n📂 Datasets trouvés: {len(dataset_files)}")
    print(f"📁 Dossier de sortie: {models_output_dir}")
    
    print(f"\n⚠️  Les modèles existants seront ÉCRASÉS")
    print(f"🚀 Démarrage automatique dans 3 secondes...")
    time.sleep(3)
    
    # Réentraînement
    results = []
    total_start = time.time()
    
    for i, dataset_file in enumerate(dataset_files, 1):
        # Extraire le numéro du dataset
        dataset_num = int(dataset_file.stem.replace('Dataset', ''))
        dataset_name = f"dataset{dataset_num}"
        output_dir = models_output_dir / dataset_name
        
        print(f"\n[{i}/{len(dataset_files)}] {dataset_name}")
        
        result = retrain_model(dataset_name, dataset_file, output_dir)
        results.append(result)
        
        # Afficher progression
        if result['success']:
            print(f"   ✅ Success: F1={result['test_f1']:.4f}, AUC={result['test_auc']:.4f}, Time={result['training_time']:.1f}s")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Résumé final
    total_time = time.time() - total_start
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n{'='*80}")
    print(f"📊 RÉSUMÉ DU RÉENTRAÎNEMENT")
    print(f"{'='*80}")
    print(f"✅ Réussis: {len(successful)}/{len(results)}")
    print(f"❌ Échoués: {len(failed)}/{len(results)}")
    print(f"⏱️  Temps total: {total_time/60:.1f} minutes")
    print(f"⏱️  Temps moyen: {total_time/len(results):.1f} secondes/modèle")
    
    if successful:
        avg_f1 = sum(r['test_f1'] for r in successful) / len(successful)
        avg_auc = sum(r['test_auc'] for r in successful) / len(successful)
        print(f"\n📈 Performances moyennes:")
        print(f"   F1-Score: {avg_f1:.4f}")
        print(f"   ROC-AUC:  {avg_auc:.4f}")
    
    if failed:
        print(f"\n❌ Modèles échoués:")
        for r in failed:
            print(f"   - {r['dataset_name']}: {r.get('error', 'Unknown')}")
    
    # Sauvegarder le rapport
    report_file = Path("scripts/retraining/retrain_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_models': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'total_time_seconds': total_time,
            'results': results
        }, f, indent=2)
    
    print(f"\n📄 Rapport sauvegardé: {report_file}")
    print(f"\n✅ RÉENTRAÎNEMENT TERMINÉ!")


if __name__ == "__main__":
    main()
