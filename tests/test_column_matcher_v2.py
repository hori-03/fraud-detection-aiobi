"""
Test de compatibilité du module column_matcher.py v2.0
Vérifie que les autres scripts peuvent l'utiliser sans problème
"""

import sys
sys.path.insert(0, 'C:\\Users\\HP\\Desktop\\fraud-project')

from utils.column_matcher import ColumnMatcher, compare_datasets

def test_basic_usage():
    """Test usage de base (compatibilité v1.0)"""
    print("\n" + "="*70)
    print("TEST 1: Usage de base (compatibilité v1.0)")
    print("="*70)
    
    matcher = ColumnMatcher(fuzzy_threshold=0.7)
    
    cols1 = ['tx_amount', 'customer_id', 'timestamp']
    cols2 = ['amount', 'client_id', 'tx_time']
    
    result = matcher.calculate_semantic_similarity(cols1, cols2, verbose=False)
    
    print(f"✅ Similarity: {result['similarity']:.1%}")
    print(f"✅ Exact matches: {result['exact_matches']}")
    print(f"✅ Semantic matches: {result['semantic_matches']}")
    print(f"✅ Total matches: {result['total_matches']}")
    
    assert result['similarity'] > 0.7, "Similarity trop basse"
    assert result['total_matches'] >= 2, "Pas assez de matches"
    print("✅ Test 1 PASSED")
    
    return True


def test_new_features():
    """Test nouvelles fonctionnalités v2.0"""
    print("\n" + "="*70)
    print("TEST 2: Nouvelles fonctionnalités v2.0")
    print("="*70)
    
    matcher = ColumnMatcher(fuzzy_threshold=0.7, use_cache=True)
    
    # Test 2.1: Abréviations
    print("\n  2.1 - Support abréviations")
    cols1 = ['tx_amt', 'cust_id']
    cols2 = ['transaction_amount', 'customer_id']
    result = matcher.calculate_semantic_similarity(cols1, cols2, verbose=False)
    print(f"    ✅ Similarity: {result['similarity']:.1%} (attendu >80%)")
    assert result['similarity'] > 0.8, f"Abréviations non matchées: {result['similarity']}"
    
    # Test 2.2: Métriques enrichies
    print("\n  2.2 - Nouvelles métriques")
    assert 'precision' in result, "Métrique precision manquante"
    assert 'recall' in result, "Métrique recall manquante"
    assert 'f1_score' in result, "Métrique f1_score manquante"
    assert 'confidence' in result, "Métrique confidence manquante"
    print(f"    ✅ Precision: {result['precision']:.1%}")
    print(f"    ✅ Recall: {result['recall']:.1%}")
    print(f"    ✅ F1-Score: {result['f1_score']:.1%}")
    print(f"    ✅ Confidence: {result['confidence']:.1%}")
    
    # Test 2.3: Qualité du match
    print("\n  2.3 - Évaluation qualité")
    quality = matcher.get_match_quality(cols1, cols2)
    print(f"    ✅ Match quality: {quality}")
    assert quality in ['excellent', 'good', 'fair', 'poor', 'incompatible'], "Qualité invalide"
    
    # Test 2.4: Suggestions de mapping
    print("\n  2.4 - Suggestions de mapping")
    suggestions = matcher.suggest_column_mapping(['montant'], ['amount', 'value'], top_n=2)
    print(f"    ✅ Suggestions pour 'montant': {list(suggestions['montant'])}")
    assert 'montant' in suggestions, "Suggestions manquantes"
    
    # Test 2.5: Analyse de groupes
    print("\n  2.5 - Analyse de groupes")
    groups = matcher.analyze_column_groups(['tx_id', 'tx_amount', 'customer_id'], verbose=False)
    print(f"    ✅ Groupes détectés: {list(groups.keys())}")
    assert 'transaction_id' in groups, "Groupe transaction_id manquant"
    assert 'amount' in groups, "Groupe amount manquant"
    
    print("\n✅ Test 2 PASSED")
    return True


def test_backward_compatibility():
    """Test compatibilité avec code existant"""
    print("\n" + "="*70)
    print("TEST 3: Compatibilité backward (apply_automl_production.py)")
    print("="*70)
    
    # Simuler usage dans apply_automl_production.py
    matcher = ColumnMatcher(fuzzy_threshold=0.7)
    
    # Pattern utilisé dans apply_automl_production.py ligne 250
    cols1 = ['amount', 'customer_id', 'timestamp', 'merchant', 'country']
    cols2 = ['tx_amount', 'cust_id', 'tx_time', 'vendor', 'pays']
    
    semantic_result = matcher.calculate_semantic_similarity(cols1, cols2, verbose=False)
    
    # Vérifier que les clés attendues existent
    required_keys = ['similarity', 'overlap_ratio', 'exact_matches', 'semantic_matches', 
                     'fuzzy_matches', 'total_matches', 'details']
    
    for key in required_keys:
        assert key in semantic_result, f"Clé {key} manquante (requis par apply_automl_production.py)"
        print(f"  ✅ {key}: présent")
    
    print(f"\n✅ Similarity: {semantic_result['similarity']:.1%}")
    print(f"✅ Total matches: {semantic_result['total_matches']}/{len(cols1)}")
    
    print("\n✅ Test 3 PASSED - apply_automl_production.py compatible")
    return True


def test_compare_datasets_function():
    """Test fonction compare_datasets"""
    print("\n" + "="*70)
    print("TEST 4: Fonction compare_datasets")
    print("="*70)
    
    cols1 = ['amount', 'time', 'customer']
    cols2 = ['tx_amt', 'timestamp', 'client_id']
    
    # Test avec use_cache paramètre (nouveau en v2.0)
    result = compare_datasets(cols1, cols2, fuzzy_threshold=0.7, verbose=False, use_cache=True)
    
    print(f"✅ Similarity: {result['similarity']:.1%}")
    print(f"✅ Match quality: {result['match_quality']}")
    
    assert 'match_quality' in result, "match_quality manquant"
    assert result['similarity'] > 0.5, "Similarity trop basse"
    
    print("\n✅ Test 4 PASSED")
    return True


def test_performance_cache():
    """Test performance avec cache"""
    print("\n" + "="*70)
    print("TEST 5: Performance avec cache")
    print("="*70)
    
    import time
    
    cols1 = ['tx_amount'] * 10 + ['customer_id'] * 10
    cols2 = ['amount'] * 10 + ['client_id'] * 10
    
    # Avec cache
    matcher_cache = ColumnMatcher(fuzzy_threshold=0.7, use_cache=True)
    start = time.time()
    for _ in range(5):
        matcher_cache.calculate_semantic_similarity(cols1, cols2, verbose=False)
    time_with_cache = time.time() - start
    
    # Sans cache
    matcher_no_cache = ColumnMatcher(fuzzy_threshold=0.7, use_cache=False)
    start = time.time()
    for _ in range(5):
        matcher_no_cache.calculate_semantic_similarity(cols1, cols2, verbose=False)
    time_without_cache = time.time() - start
    
    print(f"  Avec cache:    {time_with_cache:.4f}s")
    print(f"  Sans cache:    {time_without_cache:.4f}s")
    print(f"  Speedup:       {time_without_cache/time_with_cache:.2f}x")
    
    print("\n✅ Test 5 PASSED")
    return True


def main():
    """Exécution de tous les tests"""
    print("\n" + "="*70)
    print("🧪 TESTS DE COMPATIBILITÉ - column_matcher.py v2.0")
    print("="*70)
    
    tests = [
        ("Usage de base (v1.0 compatible)", test_basic_usage),
        ("Nouvelles fonctionnalités v2.0", test_new_features),
        ("Compatibilité backward", test_backward_compatibility),
        ("Fonction compare_datasets", test_compare_datasets_function),
        ("Performance cache", test_performance_cache)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "✅ PASSED"))
        except Exception as e:
            results.append((test_name, f"❌ FAILED: {str(e)}"))
            print(f"\n❌ FAILED: {str(e)}")
    
    # Résumé
    print("\n" + "="*70)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*70)
    
    for test_name, result in results:
        print(f"  {result:15s} - {test_name}")
    
    passed = sum(1 for _, r in results if "PASSED" in r)
    total = len(results)
    
    print(f"\n{'='*70}")
    print(f"Résultat: {passed}/{total} tests passés")
    
    if passed == total:
        print("✅ TOUS LES TESTS PASSÉS - column_matcher.py v2.0 est compatible!")
    else:
        print("⚠️ Certains tests ont échoué")
    
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
