"""
Analyse rapide des prédictions pour comprendre pourquoi 0 fraudes
"""
import pandas as pd
import sys

# Charger le dataset
df = pd.read_csv(r"c:\Users\HP\Downloads\9_20251107_122049_aml_test_dataset.csv")

print("=" * 80)
print("ANALYSE DU DATASET NON ÉTIQUETÉ")
print("=" * 80)

print(f"\n📊 Dimensions: {df.shape[0]} lignes x {df.shape[1]} colonnes")

print(f"\n📋 Colonnes du dataset:")
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    nulls = df[col].isnull().sum()
    print(f"  {i:2d}. {col:30s} | Type: {str(dtype):10s} | Nulls: {nulls:5d}")

print(f"\n🔍 Aperçu des premières lignes:")
print(df.head())

print(f"\n📈 Statistiques descriptives:")
print(df.describe())

# Vérifier s'il y a une colonne de probabilité ou de prédiction
pred_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['fraud', 'pred', 'proba', 'risk', 'score'])]
if pred_cols:
    print(f"\n🎯 Colonnes de prédiction trouvées: {pred_cols}")
    for col in pred_cols:
        print(f"\n  {col}:")
        print(f"    Min: {df[col].min()}")
        print(f"    Max: {df[col].max()}")
        print(f"    Mean: {df[col].mean():.4f}")
        print(f"    Valeurs uniques: {df[col].nunique()}")
        print(f"    Distribution:\n{df[col].value_counts()}")
else:
    print(f"\n⚠️  Aucune colonne de prédiction trouvée!")
    print(f"   Le fichier téléchargé est-il bien le fichier de résultats?")

# Vérifier les montants si disponibles
amount_cols = [col for col in df.columns if 'amount' in col.lower() or 'montant' in col.lower()]
if amount_cols:
    print(f"\n💰 Colonnes de montant trouvées: {amount_cols}")
    for col in amount_cols:
        print(f"\n  {col}:")
        print(f"    Min: {df[col].min()}")
        print(f"    Max: {df[col].max()}")
        print(f"    Mean: {df[col].mean():.2f}")
        print(f"    Median: {df[col].median():.2f}")

print("\n" + "=" * 80)
