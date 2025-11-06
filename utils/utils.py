"""
utils.py

Fonctions utilitaires optimisées pour le projet de détection de fraude et méta-modèle.
"""

import pandas as pd
import os
import time
from tqdm import tqdm

def load_dataset_optimized(csv_path, parquet_path=None):
    """
    Charge un dataset de manière optimisée (Parquet si disponible).
    
    Args:
        csv_path (str): Chemin vers le fichier CSV
        parquet_path (str): Chemin vers le fichier Parquet (optionnel)
    
    Returns:
        pd.DataFrame: Dataset chargé
    """
    if parquet_path is None:
        parquet_path = csv_path.replace('.csv', '.parquet')
    
    print("Chargement du dataset...")
    start_time = time.time()
    
    if os.path.exists(parquet_path):
        print("  → Chargement depuis Parquet (optimisé)")
        df = pd.read_parquet(parquet_path)
    else:
        print("  → Chargement depuis CSV et conversion Parquet")
        df = pd.read_csv(csv_path)
        # Sauvegarder en Parquet pour les prochaines fois
        df.to_parquet(parquet_path, index=False)
        print(f"  → Fichier Parquet sauvegardé : {parquet_path}")
    
    print(f"Dataset chargé en {time.time() - start_time:.2f}s")
    return df

def encode_categorical_columns(df, exclude_cols=None, show_progress=True):
    """
    Encode les colonnes catégorielles avec barre de progression.
    
    Args:
        df (pd.DataFrame): Dataset à encoder
        exclude_cols (list): Colonnes à exclure de l'encodage
        show_progress (bool): Afficher la barre de progression
    
    Returns:
        pd.DataFrame: Dataset encodé
    """
    from sklearn.preprocessing import LabelEncoder
    
    if exclude_cols is None:
        exclude_cols = []
    
    categorical_cols = [col for col in df.select_dtypes(include=['object']).columns 
                       if col not in exclude_cols]
    
    if show_progress:
        print("Encodage des variables catégorielles...")
        categorical_cols = tqdm(categorical_cols, desc="Encodage")
    
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    return df

def get_optimal_n_jobs(max_jobs=4):
    """
    Détermine le nombre optimal de jobs parallèles.
    
    Args:
        max_jobs (int): Nombre maximum de jobs
    
    Returns:
        int: Nombre optimal de jobs
    """
    return min(max_jobs, os.cpu_count() or 1)

def save_with_timing(data, filepath, format='json'):
    """
    Sauvegarde des données avec mesure du temps.
    
    Args:
        data: Données à sauvegarder
        filepath (str): Chemin de sauvegarde
        format (str): Format ('json', 'parquet', 'csv')
    """
    import json
    
    start_time = time.time()
    
    if format == 'json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif format == 'parquet' and isinstance(data, pd.DataFrame):
        data.to_parquet(filepath, index=False)
    elif format == 'csv' and isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    
    save_time = time.time() - start_time
    file_size = os.path.getsize(filepath) / 1024 / 1024  # MB
    
    print(f"Fichier sauvegardé en {save_time:.2f}s ({file_size:.1f} MB) : {filepath}")
