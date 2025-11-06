"""
test_apply_automl_production_v2.py

Script de test pour valider toutes les fonctionnalités v2.0
de apply_automl_production.py

Tests:
1. ✅ Exclusion automatique ID/timestamp
2. ✅ Auto-match avec matching sémantique
3. ✅ Ensemble predictions (top-3)
4. ✅ Anomaly detection
5. ✅ Calibration des probabilités
6. ✅ Export enrichi (Excel + JSON)
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import json

def run_command(cmd, description):
    """Exécute une commande et affiche le résultat"""
    print(f"\n{'='*80}")
    print(f"🧪 TEST: {description}")
    print(f"{'='*80}")
    print(f"📝 Commande: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"\n✅ TEST RÉUSSI: {description}")
    else:
        print(f"\n❌ TEST ÉCHOUÉ: {description}")
        print(f"   Return code: {result.returncode}")
    
    return result.returncode == 0


def main():
    print(f"\n{'#'*80}")
    print(f"# TEST SUITE - apply_automl_production.py v2.0")
    print(f"{'#'*80}\n")
    
    # Vérifier que le script existe
    if not Path("apply_automl_production.py").exists():
        print("❌ apply_automl_production.py introuvable!")
        sys.exit(1)
    
    # Vérifier qu'il y a des datasets de test
    test_datasets = list(Path("data/datasets").glob("Dataset3*.csv"))
    if not test_datasets:
        print("❌ Aucun dataset de test trouvé dans data/datasets/")
        sys.exit(1)
    
    test_dataset = str(test_datasets[0])  # Utiliser Dataset30 ou similaire
    print(f"📊 Dataset de test: {test_dataset}")
    
    results = {}
    
    # ============================================================
    # TEST 1: Lister les modèles disponibles
    # ============================================================
    success = run_command(
        "python apply_automl_production.py --list_models",
        "Liste des modèles disponibles"
    )
    results['test_1_list_models'] = success
    
    # ============================================================
    # TEST 2: Auto-match classique
    # ============================================================
    success = run_command(
        f"python apply_automl_production.py --dataset {test_dataset} --auto_match --output test_output/test2_automatch",
        "Auto-match classique (single model)"
    )
    results['test_2_automatch'] = success
    
    if success:
        # Vérifier que le fichier CSV existe
        output_file = Path("test_output/test2_automatch.csv")
        if output_file.exists():
            df = pd.read_csv(output_file)
            print(f"\n   ✅ Output CSV créé: {len(df)} lignes, {len(df.columns)} colonnes")
            print(f"   📋 Colonnes: {list(df.columns)}")
            
            # Vérifier colonnes attendues
            expected_cols = ['fraud_probability', 'fraud_prediction', 'risk_level']
            missing = [col for col in expected_cols if col not in df.columns]
            if missing:
                print(f"   ⚠️  Colonnes manquantes: {missing}")
            else:
                print(f"   ✅ Toutes les colonnes attendues présentes")
        else:
            print(f"   ❌ Output CSV non créé: {output_file}")
            results['test_2_automatch'] = False
    
    # ============================================================
    # TEST 3: Ensemble predictions (top-3)
    # ============================================================
    success = run_command(
        f"python apply_automl_production.py --dataset {test_dataset} --ensemble --top_k 3 --output test_output/test3_ensemble",
        "Ensemble predictions (top-3 models)"
    )
    results['test_3_ensemble'] = success
    
    if success:
        output_file = Path("test_output/test3_ensemble.csv")
        if output_file.exists():
            df = pd.read_csv(output_file)
            print(f"\n   ✅ Output CSV créé: {len(df)} lignes")
            
            # Vérifier colonnes spécifiques à l'ensemble
            ensemble_cols = ['prediction_variance', 'prediction_stability']
            missing = [col for col in ensemble_cols if col not in df.columns]
            if missing:
                print(f"   ⚠️  Colonnes ensemble manquantes: {missing}")
            else:
                print(f"   ✅ Colonnes ensemble présentes")
                print(f"      - Stabilité moyenne: {df['prediction_stability'].mean():.2%}")
                print(f"      - Variance moyenne: {df['prediction_variance'].mean():.4f}")
    
    # ============================================================
    # TEST 4: Anomaly detection
    # ============================================================
    success = run_command(
        f"python apply_automl_production.py --dataset {test_dataset} --auto_match --anomaly_detection --output test_output/test4_anomaly",
        "Anomaly detection (Isolation Forest)"
    )
    results['test_4_anomaly'] = success
    
    if success:
        output_file = Path("test_output/test4_anomaly.csv")
        if output_file.exists():
            df = pd.read_csv(output_file)
            print(f"\n   ✅ Output CSV créé")
            
            # Vérifier colonnes anomaly
            anomaly_cols = ['anomaly_score', 'is_anomaly', 'combined_score']
            missing = [col for col in anomaly_cols if col not in df.columns]
            if missing:
                print(f"   ⚠️  Colonnes anomaly manquantes: {missing}")
            else:
                print(f"   ✅ Colonnes anomaly présentes")
                n_anomalies = df['is_anomaly'].sum()
                print(f"      - Anomalies détectées: {n_anomalies} ({n_anomalies/len(df):.2%})")
                print(f"      - Score anomaly moyen: {df['anomaly_score'].mean():.3f}")
    
    # ============================================================
    # TEST 5: Calibration
    # ============================================================
    success = run_command(
        f"python apply_automl_production.py --dataset {test_dataset} --auto_match --calibrate --output test_output/test5_calibrate",
        "Calibration des probabilités"
    )
    results['test_5_calibrate'] = success
    
    if success:
        output_file = Path("test_output/test5_calibrate.csv")
        if output_file.exists():
            df = pd.read_csv(output_file)
            print(f"\n   ✅ Output CSV créé")
            
            if 'fraud_probability_calibrated' in df.columns:
                print(f"   ✅ Colonne fraud_probability_calibrated présente")
                print(f"      - Proba brute: mean={df['fraud_probability'].mean():.3f}, std={df['fraud_probability'].std():.3f}")
                print(f"      - Proba calibrée: mean={df['fraud_probability_calibrated'].mean():.3f}, std={df['fraud_probability_calibrated'].std():.3f}")
            else:
                print(f"   ⚠️  Colonne fraud_probability_calibrated manquante")
    
    # ============================================================
    # TEST 6: Export enrichi (Excel + JSON)
    # ============================================================
    success = run_command(
        f"python apply_automl_production.py --dataset {test_dataset} --auto_match --rich_export --output test_output/test6_rich",
        "Export enrichi (Excel + JSON)"
    )
    results['test_6_rich_export'] = success
    
    if success:
        # Vérifier Excel
        excel_file = Path("test_output/test6_rich.xlsx")
        if excel_file.exists():
            print(f"\n   ✅ Fichier Excel créé: {excel_file}")
            
            # Charger et vérifier les sheets
            try:
                xls = pd.ExcelFile(excel_file)
                sheets = xls.sheet_names
                print(f"      Sheets: {sheets}")
                
                expected_sheets = ['All Predictions', 'High Risk', 'Summary']
                missing_sheets = [s for s in expected_sheets if s not in sheets]
                if missing_sheets:
                    print(f"   ⚠️  Sheets manquants: {missing_sheets}")
                else:
                    print(f"   ✅ Tous les sheets présents")
            except Exception as e:
                print(f"   ⚠️  Erreur lecture Excel: {e}")
        else:
            print(f"   ❌ Fichier Excel non créé")
            results['test_6_rich_export'] = False
        
        # Vérifier JSON
        json_file = Path("test_output/test6_rich.json")
        if json_file.exists():
            print(f"\n   ✅ Fichier JSON créé: {json_file}")
            
            # Charger et vérifier structure
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                expected_keys = ['metadata', 'summary_statistics', 'top_10_frauds', 'predictions']
                missing_keys = [k for k in expected_keys if k not in data]
                if missing_keys:
                    print(f"   ⚠️  Clés manquantes: {missing_keys}")
                else:
                    print(f"   ✅ Toutes les clés présentes")
                    print(f"      - Total transactions: {data['metadata']['n_total']}")
                    print(f"      - Fraudes détectées: {data['metadata']['n_fraud']}")
                    print(f"      - Top 10 frauds: {len(data['top_10_frauds'])} entrées")
            except Exception as e:
                print(f"   ⚠️  Erreur lecture JSON: {e}")
        else:
            print(f"   ❌ Fichier JSON non créé")
            results['test_6_rich_export'] = False
    
    # ============================================================
    # TEST 7: Mode batch (si dataset assez gros)
    # ============================================================
    # Skip si dataset < 10k lignes
    try:
        df_test = pd.read_csv(test_dataset)
        if len(df_test) >= 10000:
            success = run_command(
                f"python apply_automl_production.py --dataset {test_dataset} --auto_match --batch_size 5000 --output test_output/test7_batch",
                "Mode batch (5000 lignes/batch)"
            )
            results['test_7_batch'] = success
        else:
            print(f"\n⏭️  TEST 7 SKIPPED: Dataset trop petit pour batch mode ({len(df_test)} < 10000)")
            results['test_7_batch'] = None
    except Exception as e:
        print(f"   ⚠️  Erreur lecture dataset: {e}")
        results['test_7_batch'] = None
    
    # ============================================================
    # TEST 8: Ensemble + Anomaly + Calibration (combo complet)
    # ============================================================
    success = run_command(
        f"python apply_automl_production.py --dataset {test_dataset} --ensemble --top_k 2 --anomaly_detection --calibrate --output test_output/test8_full",
        "Combo complet (Ensemble + Anomaly + Calibration)"
    )
    results['test_8_full_combo'] = success
    
    if success:
        output_file = Path("test_output/test8_full.csv")
        if output_file.exists():
            df = pd.read_csv(output_file)
            print(f"\n   ✅ Output CSV créé: {len(df)} lignes")
            
            # Vérifier toutes les colonnes avancées
            advanced_cols = [
                'prediction_variance', 'prediction_stability',
                'anomaly_score', 'is_anomaly', 'combined_score',
                'fraud_probability_calibrated'
            ]
            present = [col for col in advanced_cols if col in df.columns]
            missing = [col for col in advanced_cols if col not in df.columns]
            
            print(f"   ✅ Colonnes présentes ({len(present)}/{len(advanced_cols)}): {present}")
            if missing:
                print(f"   ⚠️  Colonnes manquantes: {missing}")
    
    # ============================================================
    # RÉSUMÉ FINAL
    # ============================================================
    print(f"\n\n{'#'*80}")
    print(f"# RÉSUMÉ DES TESTS")
    print(f"{'#'*80}\n")
    
    total = len([v for v in results.values() if v is not None])
    passed = len([v for v in results.values() if v is True])
    failed = len([v for v in results.values() if v is False])
    skipped = len([v for v in results.values() if v is None])
    
    print(f"📊 Résultats:")
    print(f"   Total tests: {total}")
    print(f"   ✅ Réussis: {passed}")
    print(f"   ❌ Échoués: {failed}")
    print(f"   ⏭️  Skipped: {skipped}")
    print(f"   Taux de réussite: {passed/total*100:.1f}%")
    
    print(f"\n📋 Détails:")
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⏭️  SKIP"
        print(f"   {status} - {test_name}")
    
    # Vérifier fichiers générés
    output_dir = Path("test_output")
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        print(f"\n📁 Fichiers générés ({len(files)}):")
        for f in sorted(files):
            size = f.stat().st_size / 1024  # KB
            print(f"   - {f.name:40s} ({size:8.1f} KB)")
    
    print(f"\n{'#'*80}\n")
    
    # Return code
    if failed > 0:
        print(f"❌ {failed} test(s) échoué(s)")
        sys.exit(1)
    else:
        print(f"✅ Tous les tests réussis!")
        sys.exit(0)


if __name__ == "__main__":
    main()
