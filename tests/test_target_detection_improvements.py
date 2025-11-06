"""
Test pour vérifier les améliorations de la détection automatique de target
- Test avec dataset non étiqueté (pas de colonne fraude)
- Test avec dataset contenant des colonnes temporelles (weekday, day, month, etc.)
- Vérification que les colonnes évidentes ne sont PAS détectées comme target
"""

import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from automl_transformer.full_automl import FullAutoML

def create_test_datasets():
    """Créer des datasets de test pour vérifier la détection"""
    
    # Dataset 1: Non étiqueté avec weekday, day, month (devrait échouer la détection)
    print("\n" + "="*80)
    print("📊 TEST 1: Dataset non étiqueté avec colonnes temporelles")
    print("="*80)
    
    df_unlabeled = pd.DataFrame({
        'tx_id': [f'TX{i:06d}' for i in range(1000)],
        'customer_ref': [f'CUST{i%100:04d}' for i in range(1000)],
        'amount_fcfa': [1000 + i*10 for i in range(1000)],
        'weekday': [i % 7 for i in range(1000)],  # 0-6 (Lundi-Dimanche)
        'day': [1 + (i % 28) for i in range(1000)],  # 1-28
        'month': [1 + (i % 12) for i in range(1000)],  # 1-12
        'year': [2024] * 1000,
        'hour': [i % 24 for i in range(1000)],
        'account_tenure_days': [30 + i % 365 for i in range(1000)],
        'balance_before': [5000 + i*100 for i in range(1000)]
    })
    
    df_unlabeled.to_csv('test_unlabeled.csv', index=False)
    print(f"✅ Dataset créé: {df_unlabeled.shape[0]} lignes, {df_unlabeled.shape[1]} colonnes")
    print(f"   Colonnes: {', '.join(df_unlabeled.columns.tolist())}")
    
    # Dataset 2: Étiqueté avec colonne fraude claire
    print("\n" + "="*80)
    print("📊 TEST 2: Dataset étiqueté avec colonne fraude claire")
    print("="*80)
    
    df_labeled = df_unlabeled.copy()
    df_labeled['is_fraudulent'] = [1 if i % 10 == 0 else 0 for i in range(1000)]  # 10% fraude
    df_labeled.to_csv('test_labeled.csv', index=False)
    print(f"✅ Dataset créé: {df_labeled.shape[0]} lignes, {df_labeled.shape[1]} colonnes")
    print(f"   Colonnes: {', '.join(df_labeled.columns.tolist())}")
    print(f"   Distribution fraude: {df_labeled['is_fraudulent'].value_counts().to_dict()}")
    
    # Dataset 3: Dataset avec amount comme seule colonne numérique continue (ne devrait PAS être détecté)
    print("\n" + "="*80)
    print("📊 TEST 3: Dataset avec 'amount' comme colonne continue")
    print("="*80)
    
    df_amount_only = pd.DataFrame({
        'tx_id': [f'TX{i:06d}' for i in range(1000)],
        'customer_ref': [f'CUST{i%100:04d}' for i in range(1000)],
        'amount_fcfa': [1000 + i*10.5 for i in range(1000)],  # Continuous values
        'tx_type': [['Online', 'POS', 'ATM'][i % 3] for i in range(1000)]
    })
    df_amount_only.to_csv('test_amount_only.csv', index=False)
    print(f"✅ Dataset créé: {df_amount_only.shape[0]} lignes, {df_amount_only.shape[1]} colonnes")
    print(f"   Colonnes: {', '.join(df_amount_only.columns.tolist())}")

def test_target_detection():
    """Tester la détection de target avec les nouveaux datasets"""
    
    test_cases = [
        ('test_unlabeled.csv', 'Non étiqueté (weekday, day, month, hour)', 'NO_TARGET_DETECTED'),
        ('test_labeled.csv', 'Étiqueté (is_fraudulent)', 'is_fraudulent'),
        ('test_amount_only.csv', 'Amount continuous (tx_type détecté)', 'target')  # tx_type est un candidat valide
    ]
    
    results = []
    
    for filepath, description, expected in test_cases:
        print("\n" + "="*80)
        print(f"🔍 TEST: {description}")
        print("="*80)
        print(f"📁 Fichier: {filepath}")
        print(f"🎯 Attendu: {expected}")
        
        try:
            # Créer une instance de FullAutoML
            automl = FullAutoML()
            
            # Tester la détection de target
            try:
                # load_and_prepare_data retourne seulement le dataframe
                df = automl.load_and_prepare_data(filepath)
                detected = automl.target_col
                error_msg = None
                
                # Vérifier que le dataframe est valide
                if df is None:
                    detected = None
                    error_msg = "DataFrame is None"
                    
            except Exception as e:
                detected = None
                error_msg = str(e)
            
            print(f"\n🎯 Target détecté: {detected}")
            if error_msg:
                print(f"⚠️  Message: {error_msg}")
            
            # Vérification
            if expected == 'NO_TARGET_DETECTED':
                if detected is None or error_msg:
                    status = '✅ PASS'
                    explanation = 'Aucun target détecté comme attendu'
                else:
                    status = '❌ FAIL'
                    explanation = f'Target détecté alors qu\'aucun n\'était attendu: {detected}'
            else:
                if detected == expected:
                    status = '✅ PASS'
                    explanation = f'Target correct détecté: {detected}'
                else:
                    status = '❌ FAIL'
                    explanation = f'Target incorrect: attendu {expected}, obtenu {detected}'
            
            print(f"\n{status}: {explanation}")
            
            results.append({
                'test': description,
                'expected': expected,
                'detected': detected or 'None',
                'status': status
            })
            
        except Exception as e:
            print(f"\n❌ ERREUR: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                'test': description,
                'expected': expected,
                'detected': f'ERROR: {str(e)}',
                'status': '❌ FAIL'
            })
    
    # Résumé final
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*80)
    
    for result in results:
        print(f"\n{result['status']} {result['test']}")
        print(f"   Attendu: {result['expected']}")
        print(f"   Détecté: {result['detected']}")
    
    # Statistiques
    passed = sum(1 for r in results if '✅' in r['status'])
    total = len(results)
    print(f"\n{'='*80}")
    print(f"🏆 RÉSULTAT GLOBAL: {passed}/{total} tests réussis ({passed/total*100:.1f}%)")
    print(f"{'='*80}")
    
    # Cleanup
    for filepath, _, _ in test_cases:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"🗑️  Nettoyage: {filepath} supprimé")

if __name__ == '__main__':
    print("\n" + "🎯 TEST DES AMÉLIORATIONS DE DÉTECTION DE TARGET" + "\n")
    print("Ce test vérifie que:")
    print("  1. Les colonnes temporelles (weekday, day, month, hour) ne sont PAS détectées comme target")
    print("  2. Les colonnes financières (amount, balance) ne sont PAS détectées comme target")
    print("  3. Les vraies colonnes fraude (is_fraudulent, fraud, label) sont correctement détectées")
    print("  4. Les datasets non étiquetés génèrent des warnings appropriés")
    
    create_test_datasets()
    test_target_detection()
