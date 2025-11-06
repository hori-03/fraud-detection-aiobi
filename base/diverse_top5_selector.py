"""
Sélecteur de configurations XGBoost diversifiées
SCRIPT ESSENTIEL pour créer des configurations variées au lieu du top-5 similaire
"""

import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

def load_grid_results(filepath):
    """Charger les résultats de GridSearchCV"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_param_vectors(results):
    """Extraire les vecteurs de paramètres pour calcul de distance"""
    param_vectors = []
    configs = []
    
    for result in results:
        # Extraire les paramètres disponibles (compatible avec grille fraud optimisée)
        params = result['params']
        vector = [
            params['max_depth'],
            params['learning_rate'], 
            params['subsample'],
            params['colsample_bytree'],
            params['gamma'],
            params['min_child_weight'],
            params.get('reg_alpha', 0),  # Nouveau paramètre avec valeur par défaut
            params.get('reg_lambda', 1.0),  # Nouveau paramètre avec valeur par défaut
            params.get('scale_pos_weight', 1.0),  # Nouveau paramètre avec valeur par défaut
            params['n_estimators']
        ]
        param_vectors.append(vector)
        configs.append(result)
    
    return np.array(param_vectors), configs

def diverse_top15_selection(results, score_threshold=0.95):
    """
    Sélection diversifiée du top-15
    
    Stratégie:
    1. Filtrer les configs avec score >= threshold
    2. Sélectionner la meilleure
    3. Pour les suivantes, choisir celles les plus différentes
    """
    
    # Détecter la colonne de score principale (compatible avec multiple scoring)
    score_column = None
    possible_score_columns = ['mean_test_score', 'mean_test_f1', 'mean_test_roc_auc', 'mean_test_precision']
    
    for col in possible_score_columns:
        if col in results[0]:
            score_column = col
            break
    
    if score_column is None:
        available_keys = [k for k in results[0].keys() if 'mean_test' in k]
        raise KeyError(f"Aucune colonne de score trouvée. Colonnes mean_test disponibles: {available_keys}")
    
    print(f"📊 Utilisation du score: {score_column}")
    
    # Filtrer par score minimum
    good_results = [r for r in results if r[score_column] >= score_threshold]
    print(f"📊 {len(good_results)} configurations avec {score_column} >= {score_threshold}")
    
    if len(good_results) < 15:
        print("⚠️ Pas assez de bonnes configurations, réduction du threshold")
        good_results = sorted(results, key=lambda x: x[score_column], reverse=True)[:50]
    
    # Extraire vecteurs de paramètres
    param_vectors, configs = extract_param_vectors(good_results)
    
    # Normaliser pour égaliser l'importance des paramètres
    scaler = StandardScaler()
    param_vectors_scaled = scaler.fit_transform(param_vectors)
    
    # Sélection diversifiée
    selected_indices = []
    
    # 1. Sélectionner la meilleure configuration
    best_idx = 0  # Premier dans la liste triée
    selected_indices.append(best_idx)
    
    # 2. Sélectionner les 14 suivantes par diversité maximale
    for _ in range(14):
        max_min_distance = -1
        best_candidate = -1
        
        for i, candidate_vector in enumerate(param_vectors_scaled):
            if i in selected_indices:
                continue
                
            # Calculer distance minimale aux déjà sélectionnés
            min_distance = float('inf')
            for selected_idx in selected_indices:
                distance = np.linalg.norm(candidate_vector - param_vectors_scaled[selected_idx])
                min_distance = min(min_distance, distance)
            
            # Garder le candidat avec la plus grande distance minimale
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_candidate = i
        
        if best_candidate != -1:
            selected_indices.append(best_candidate)
    
    # Retourner les configurations sélectionnées
    diverse_top5 = [configs[i] for i in selected_indices]
    
    print("\n🎯 Top-5 Diversifié sélectionné:")
    for i, config in enumerate(diverse_top5, 1):
        params = config['params']
        score = config[score_column]  # Utiliser la colonne de score détectée
        print(f"{i}. Score: {score:.4f} | depth:{params['max_depth']} lr:{params['learning_rate']:.3f} "
              f"sub:{params['subsample']} reg_a:{params.get('reg_alpha', 0):.2f} "
              f"scale:{params.get('scale_pos_weight', 1.0):.1f}")
    
    return diverse_top5

def analyze_diversity(configs):
    """Analyser la diversité des configurations sélectionnées"""
    print("\n📊 Analyse de diversité:")
    
    param_vectors, _ = extract_param_vectors(configs)
    
    # Calculer matrice de distances
    distances = euclidean_distances(param_vectors)
    
    # Distance moyenne entre configurations
    n = len(configs)
    total_distance = 0
    count = 0
    
    for i in range(n):
        for j in range(i+1, n):
            total_distance += distances[i][j]
            count += 1
    
    avg_distance = total_distance / count if count > 0 else 0
    print(f"Distance moyenne entre configs: {avg_distance:.3f}")
    
    # Analyser variabilité par paramètre (mis à jour pour grille fraud optimisée)
    param_names = ['max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 
                   'gamma', 'min_child_weight', 'reg_alpha', 'reg_lambda', 'scale_pos_weight', 'n_estimators']
    
    for i, param_name in enumerate(param_names):
        values = param_vectors[:, i]
        diversity = np.std(values)
        print(f"{param_name}: std={diversity:.3f} | range=[{values.min():.3f}, {values.max():.3f}]")

if __name__ == "__main__":
    import sys
    
    # Utilisation avec les résultats de baseline_xgboost.py
    print("🎯 SÉLECTEUR DE CONFIGURATIONS DIVERSIFIÉES")
    print("=" * 50)
    print("Ce script transforme un top-5 similaire en configurations diversifiées")
    print("Usage: python diverse_top5_selector.py [DATASET_NAME]")
    
    # Déterminer le nom du dataset
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "Dataset19"
    print(f"📊 Dataset: {dataset_name}")
    
    try:
        # Tenter de charger TOUS les résultats (depuis baseline_xgboost.py) - Nouvelle organisation
        results = load_grid_results(f'data/results/{dataset_name}_grid_search_results.json')
        
        # Sélection diversifiée
        diverse_top15 = diverse_top15_selection(results, score_threshold=0.98)
        
        # Analyser la diversité
        analyze_diversity(diverse_top15)
        
        # Sauvegarder la version diversifiée - Nouvelle organisation
        output_file = f'data/top5/{dataset_name}_diverse_top15_selection.json'
        with open(output_file, 'w') as f:
            json.dump(diverse_top15, f, indent=2)
        
        print(f"\n✅ Top-15 diversifié sauvegardé dans '{output_file}'")
        
    except FileNotFoundError:
        print(f"❌ Fichier de résultats non trouvé: data/results/{dataset_name}_grid_search_results.json")
        print("Lancez d'abord baseline_xgboost.py")