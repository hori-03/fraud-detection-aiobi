# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import time
import os
import sys
from tqdm import tqdm
from scipy.stats import skew, kurtosis

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Chemin du dataset à analyser - Nouvelle organisation
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else 'data/datasets/Dataset19.csv'
DATASET_NAME = os.path.splitext(os.path.basename(DATA_PATH))[0]  # Extrait le nom du dataset
PARQUET_PATH = f'data/parquet/{DATASET_NAME}.parquet'

print("Chargement du dataset pour extraction de structure...")
start_time = time.time()

# Utiliser Parquet si disponible (plus rapide)
if os.path.exists(PARQUET_PATH):
    print("Chargement depuis Parquet (optimisé)")
    df = pd.read_parquet(PARQUET_PATH)
else:
    print("Chargement depuis CSV")
    df = pd.read_csv(DATA_PATH)

print(f"Dataset chargé en {time.time() - start_time:.2f}s")

# 🎯 ANALYSE SPÉCIALISÉE FRAUD DETECTION
print("🔍 Analyse spécialisée pour détection de fraude...")

# Détecter la colonne target (fraude) - Support pour tous les datasets
target_candidates = ['fraud_flag', 'is_fraud', 'fraud', 'label', 'target', 'class', 'is_fraudulent', 
                     'label_suspect', 'suspicious_flag', 'label_fraude', 'flag_fraude',
                     'flagged_suspicious', 'flagged_anomaly', 'skimming_detected', 'aml_flagged', 
                     'suspicious_activity', 'payment_irregularity', 'market_manipulation_flag']
target_col = None
for col in df.columns:
    col_lower = col.lower()
    if (col_lower in target_candidates or 
        'fraud' in col_lower or 'suspect' in col_lower or 'fraude' in col_lower or
        'flagged' in col_lower or 'skimming' in col_lower or 'aml' in col_lower or
        'anomaly' in col_lower or 'suspicious' in col_lower or 'irregularity' in col_lower or
        'manipulation' in col_lower):
        target_col = col
        break

# Analyse de déséquilibre des classes
class_balance = None
if target_col and target_col in df.columns:
    # Convertir 'yes'/'no' ou 'oui'/'non' en 1/0 si nécessaire (pour anciens datasets)
    if df[target_col].dtype == 'object':
        target_values = df[target_col].str.lower().map({'yes': 1, 'no': 0, 'oui': 1, 'non': 0})
    else:
        target_values = df[target_col]
    
    class_counts = target_values.value_counts()
    total = len(df)
    class_balance = {
        'target_column': target_col,
        'class_distribution': class_counts.to_dict(),
        'fraud_ratio': float(class_counts.get(1, 0) / total),
        'normal_ratio': float(class_counts.get(0, 0) / total),
        'imbalance_ratio': float(class_counts.max() / class_counts.min()) if class_counts.min() > 0 else float('inf')
    }
    print(f"📊 Classe cible détectée: {target_col}")
    print(f"   • Fraude: {class_balance['fraud_ratio']:.3%}")
    print(f"   • Normal: {class_balance['normal_ratio']:.3%}")
    print(f"   • Déséquilibre: {class_balance['imbalance_ratio']:.1f}:1")

# Extraire la structure avec analyses avancées
structure = {
    'dataset_name': DATASET_NAME,
    'n_rows': len(df),
    'n_cols': len(df.columns),
    'class_balance': class_balance,
    'feature_summary': {
        'numerical_features': 0,
        'categorical_features': 0,
        'binary_features': 0,
        'high_cardinality_features': 0,
        'features_with_missing': 0,
        'total_missing': 0,
        'avg_unique_values': 0
    },
    'fraud_analysis': {
        'potential_id_columns': [],
        'potential_amount_columns': [],
        'potential_time_columns': [],
        'high_correlation_features': []
    },
    'columns': []
}

print("Extraction des informations par colonne...")
total_unique = 0
for col in tqdm(df.columns, desc="Traitement des colonnes"):
    col_info = {
        'name': col,
        'dtype': str(df[col].dtype),
        'n_unique': df[col].nunique(),
        'sample_values': df[col].dropna().unique()[:5].tolist(),
        'missing': int(df[col].isna().sum()),
        'missing_pct': float(df[col].isna().sum() / len(df) * 100)
    }
    
    # Compteurs pour résumé
    total_unique += col_info['n_unique']
    if col_info['missing'] > 0:
        structure['feature_summary']['features_with_missing'] += 1
    structure['feature_summary']['total_missing'] += col_info['missing']
    
    # Classification des types de colonnes
    if pd.api.types.is_numeric_dtype(df[col]):
        structure['feature_summary']['numerical_features'] += 1
        col_info['mean'] = float(df[col].mean()) if not df[col].isna().all() else 0.0
        col_info['std'] = float(df[col].std()) if not df[col].isna().all() else 0.0
        col_info['min'] = float(df[col].min()) if not df[col].isna().all() else 0.0
        col_info['max'] = float(df[col].max()) if not df[col].isna().all() else 0.0
        
        # 🎯 Détection de colonnes spéciales pour fraud
        if 'amount' in col.lower() or 'value' in col.lower() or 'sum' in col.lower():
            structure['fraud_analysis']['potential_amount_columns'].append(col)
        if col_info['n_unique'] == len(df):  # Potentiel ID
            structure['fraud_analysis']['potential_id_columns'].append(col)
            
    else:
        # Analyse colonnes catégorielles
        if col_info['n_unique'] == 2:
            structure['feature_summary']['binary_features'] += 1
        elif col_info['n_unique'] > 50:
            structure['feature_summary']['high_cardinality_features'] += 1
        else:
            structure['feature_summary']['categorical_features'] += 1
            
        # Détection colonnes temporelles
        if 'time' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower():
            structure['fraud_analysis']['potential_time_columns'].append(col)
    
    structure['columns'].append(col_info)

# Finaliser les statistiques résumées
structure['feature_summary']['avg_unique_values'] = total_unique / len(df.columns)

# ============= CALCUL DES 18 FEATURES POUR META-TRANSFORMER =============
print("🧠 Calcul des features pour Meta-Transformer...")

# 1. Missing rate global
total_values = df.shape[0] * df.shape[1]
total_missing = df.isnull().sum().sum()
structure['missing_rate'] = float(total_missing / total_values)

# 2. Duplicate rate
n_duplicates = df.duplicated().sum()
structure['duplicate_rate'] = float(n_duplicates / len(df))

# 3. Corrélations avec target (déjà partiellement fait, on complète)
numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != target_col]
correlations = []

if target_col and target_col in df.columns:
    for col in numeric_cols:
        try:
            corr = df[col].corr(df[target_col])
            if not pd.isna(corr):
                correlations.append(abs(corr))
        except:
            continue

structure['correlation_max'] = float(max(correlations)) if correlations else 0.0
structure['correlation_min'] = float(min(correlations)) if correlations else 0.0
structure['correlation_mean'] = float(np.mean(correlations)) if correlations else 0.0

# 4. Variance, Skewness, Kurtosis des features numériques
variances = []
skewnesses = []
kurtoses = []

for col in numeric_cols:
    values = df[col].dropna()
    if len(values) > 0:
        # Variance
        var = values.var()
        if not pd.isna(var):
            variances.append(var)
        
        # Skewness
        try:
            sk = skew(values)
            if not pd.isna(sk):
                skewnesses.append(sk)
        except:
            pass
        
        # Kurtosis
        try:
            kurt = kurtosis(values)
            if not pd.isna(kurt):
                kurtoses.append(kurt)
        except:
            pass

structure['variance_max'] = float(max(variances)) if variances else 0.0
structure['variance_min'] = float(min(variances)) if variances else 0.0
structure['variance_mean'] = float(np.mean(variances)) if variances else 0.0

structure['skewness_max'] = float(max(skewnesses)) if skewnesses else 0.0
structure['skewness_min'] = float(min(skewnesses)) if skewnesses else 0.0
structure['skewness_mean'] = float(np.mean(skewnesses)) if skewnesses else 0.0

structure['kurtosis_max'] = float(max(kurtoses)) if kurtoses else 0.0
structure['kurtosis_mean'] = float(np.mean(kurtoses)) if kurtoses else 0.0

# ============= FIN CALCUL META-TRANSFORMER (v2.0 - 18 features) =============

print(f"✅ Features Meta-Transformer calculées (v2.0 - 18 features):")
print(f"   • Missing rate: {structure['missing_rate']:.6f}")
print(f"   • Duplicate rate: {structure['duplicate_rate']:.6f}")
print(f"   • Correlation (max/min/mean): {structure['correlation_max']:.4f} / {structure['correlation_min']:.4f} / {structure['correlation_mean']:.4f}")
print(f"   • Variance (max/min/mean): {structure['variance_max']:.2e} / {structure['variance_min']:.2e} / {structure['variance_mean']:.2e}")
print(f"   • Skewness (max/min/mean): {structure['skewness_max']:.4f} / {structure['skewness_min']:.4f} / {structure['skewness_mean']:.4f}")
print(f"   • Kurtosis (max/mean): {structure['kurtosis_max']:.4f} / {structure['kurtosis_mean']:.4f}")

# ============= FIN CALCUL META-TRANSFORMER =============

# Analyse de corrélation si colonne target trouvée
if target_col and target_col in df.columns:
    print("🔗 Analyse de corrélation avec target...")
    numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != target_col]
    if numerical_cols:
        correlations = []
        for col in numerical_cols[:10]:  # Limiter pour performance
            try:
                corr = abs(df[col].corr(df[target_col]))
                if not pd.isna(corr):
                    correlations.append({'feature': col, 'correlation': float(corr)})
            except:
                continue
        
        # Trier par corrélation décroissante
        correlations.sort(key=lambda x: x['correlation'], reverse=True)
        structure['fraud_analysis']['high_correlation_features'] = correlations[:5]

# Ajout de recommandations pour XGBoost
structure['xgboost_recommendations'] = {
    'scale_pos_weight_needed': class_balance is not None and class_balance['imbalance_ratio'] > 10,
    'feature_selection_needed': structure['n_cols'] > 50,
    'missing_imputation_needed': structure['feature_summary']['features_with_missing'] > 0,
    'encoding_needed': structure['feature_summary']['categorical_features'] > 0,
    'outlier_detection_needed': len(structure['fraud_analysis']['potential_amount_columns']) > 0
}

# Sauvegarder la structure dans un fichier JSON - Nouvelle organisation
output_file = f'data/structure/{DATASET_NAME}_dataset_structure.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(structure, f, indent=2, ensure_ascii=False)

print(f'\n✅ Structure du dataset extraite et sauvegardée dans {output_file}')
print(f"\n📊 RÉSUMÉ FRAUD DETECTION:")
print(f"   • Features numériques: {structure['feature_summary']['numerical_features']}")
print(f"   • Features catégorielles: {structure['feature_summary']['categorical_features']}")
print(f"   • Features binaires: {structure['feature_summary']['binary_features']}")
print(f"   • Colonnes montants potentielles: {len(structure['fraud_analysis']['potential_amount_columns'])}")
print(f"   • Colonnes temporelles potentielles: {len(structure['fraud_analysis']['potential_time_columns'])}")

if class_balance:
    print(f"   • Déséquilibre des classes: {class_balance['imbalance_ratio']:.1f}:1")
    print(f"   • Scale_pos_weight recommandé: {class_balance['imbalance_ratio']:.1f}")

print(f"\n🎯 Recommandations XGBoost générées dans le fichier JSON")
