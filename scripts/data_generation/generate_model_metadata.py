"""
generate_model_metadata.py

Génère les fichiers dataset_metadata.json enrichis pour tous les modèles AutoML existants
en analysant en profondeur les datasets sources.

Version améliorée avec métadonnées complètes pour matching précis.

Usage:
    python generate_model_metadata.py [--force]
    
Options:
    --force : Régénère même si les métadonnées existent déjà
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
import argparse
from collections import Counter


def detect_domain(df: pd.DataFrame, column_names: list) -> str:
    """Détecte le domaine du dataset (card, mobile, wire, etc.)"""
    col_str = ' '.join([col.lower() for col in column_names])
    
    # Scores par domaine
    scores = {
        'card_fraud': 0,
        'mobile_money': 0,
        'wire_transfer': 0,
        'atm': 0,
        'corporate_banking': 0,
        'mobile_banking': 0,
        'crypto': 0,
        'mortgage': 0,
        'investment': 0,
        'insurance': 0,
        'pos_retail': 0,
        'p2p_lending': 0,
    }
    
    # Card fraud indicators
    if any(kw in col_str for kw in ['card', 'merchant', 'authorization', 'cvv', 'pin']):
        scores['card_fraud'] += 3
    
    # Mobile money indicators
    if any(kw in col_str for kw in ['mobile', 'wallet', 'airtime', 'momo', 'orange_money']):
        scores['mobile_money'] += 3
    
    # Wire transfer indicators
    if any(kw in col_str for kw in ['wire', 'transfer', 'swift', 'iban', 'beneficiary']):
        scores['wire_transfer'] += 3
    
    # ATM indicators
    if any(kw in col_str for kw in ['atm', 'withdrawal', 'cash', 'dispenser']):
        scores['atm'] += 3
    
    # Corporate banking
    if any(kw in col_str for kw in ['corporate', 'company', 'business', 'headquarters']):
        scores['corporate_banking'] += 3
    
    # Mobile banking app
    if any(kw in col_str for kw in ['app', 'mobile_banking', 'login', 'device']):
        scores['mobile_banking'] += 2
    
    # Crypto
    if any(kw in col_str for kw in ['crypto', 'bitcoin', 'ethereum', 'wallet', 'blockchain']):
        scores['crypto'] += 3
    
    # Mortgage
    if any(kw in col_str for kw in ['mortgage', 'loan', 'property', 'interest_rate']):
        scores['mortgage'] += 3
    
    # Investment/Trading
    if any(kw in col_str for kw in ['trade', 'investment', 'portfolio', 'stock', 'broker']):
        scores['investment'] += 3
    
    # Insurance
    if any(kw in col_str for kw in ['insurance', 'claim', 'policy', 'premium']):
        scores['insurance'] += 3
    
    # POS Retail
    if any(kw in col_str for kw in ['pos', 'retail', 'merchant', 'terminal']):
        scores['pos_retail'] += 2
    
    # P2P Lending
    if any(kw in col_str for kw in ['p2p', 'lending', 'borrower', 'lender', 'credit_grade']):
        scores['p2p_lending'] += 3
    
    # Retourner le domaine avec le score le plus élevé
    max_score = max(scores.values())
    if max_score == 0:
        return 'unknown'
    
    return max(scores, key=scores.get)


def analyze_amount_patterns(df: pd.DataFrame, amount_cols: list) -> dict:
    """Analyse les patterns des colonnes de montants"""
    patterns = {}
    
    for col in amount_cols[:3]:  # Analyser max 3 colonnes de montant
        if col not in df.columns:
            continue
        
        amounts = df[col].dropna()
        if len(amounts) == 0:
            continue
        
        patterns[col] = {
            'mean': float(amounts.mean()),
            'median': float(amounts.median()),
            'std': float(amounts.std()),
            'min': float(amounts.min()),
            'max': float(amounts.max()),
            'q25': float(amounts.quantile(0.25)),
            'q75': float(amounts.quantile(0.75)),
            'skewness': float(amounts.skew()) if len(amounts) > 1 else 0,
            'has_zeros': int((amounts == 0).sum()),
            'has_negatives': int((amounts < 0).sum()),
        }
    
    return patterns


def analyze_categorical_patterns(df: pd.DataFrame, cat_cols: list) -> dict:
    """Analyse les patterns des colonnes catégorielles"""
    patterns = {}
    
    for col in cat_cols[:5]:  # Analyser max 5 colonnes catégorielles
        if col not in df.columns:
            continue
        
        values = df[col].dropna()
        if len(values) == 0:
            continue
        
        value_counts = values.value_counts()
        
        patterns[col] = {
            'n_unique': int(df[col].nunique()),
            'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
            'most_common_pct': float(value_counts.iloc[0] / len(values)) if len(value_counts) > 0 else 0,
            'top_5_values': list(value_counts.head(5).index.astype(str)),
            'is_high_cardinality': df[col].nunique() > len(df) * 0.5,
        }
    
    return patterns


def detect_temporal_features(column_names: list) -> dict:
    """Détecte les features temporelles"""
    temporal = {
        'has_timestamp': False,
        'has_date': False,
        'has_time': False,
        'has_hour': False,
        'has_weekday': False,
        'has_month': False,
        'timestamp_cols': [],
        'hour_cols': [],
        'weekday_cols': [],
    }
    
    for col in column_names:
        col_lower = col.lower()
        
        if any(kw in col_lower for kw in ['timestamp', 'datetime', 'tx_time']):
            temporal['has_timestamp'] = True
            temporal['timestamp_cols'].append(col)
        
        if 'date' in col_lower and 'update' not in col_lower:
            temporal['has_date'] = True
        
        if any(kw in col_lower for kw in ['time', 'heure']) and 'timestamp' not in col_lower:
            temporal['has_time'] = True
        
        if any(kw in col_lower for kw in ['hour', 'heure']):
            temporal['has_hour'] = True
            temporal['hour_cols'].append(col)
        
        if any(kw in col_lower for kw in ['weekday', 'day_of_week', 'dayofweek']):
            temporal['has_weekday'] = True
            temporal['weekday_cols'].append(col)
        
        if any(kw in col_lower for kw in ['month', 'mois']):
            temporal['has_month'] = True
    
    return temporal


def detect_fraud_indicators(df: pd.DataFrame, column_names: list) -> dict:
    """Détecte la colonne de fraude et analyse la distribution"""
    fraud_info = {
        'fraud_col_name': None,
        'fraud_rate': 0.0,
        'fraud_count': 0,
        'n_samples': len(df),
    }
    
    # Liste complète des mots-clés pour détecter les colonnes de fraude/anomalie/suspicion
    fraud_keywords = [
        # Fraud variations
        'fraud', 'is_fraud', 'fraud_flag', 'fraudulent', 'is_fraudulent', 'fraude', 
        'flag_fraude', 'label_fraude', 'fraud_alert', 'fraud_indicator',
        # Anomaly variations
        'anomaly', 'is_anomaly', 'anomalous',
        # Suspicious variations
        'suspicious', 'is_suspicious', 'suspicious_flag', 'suspicious_activity', 
        'label_suspect', 'flagged_suspicious',
        # Specific fraud types
        'skimming', 'skimming_detected', 'skimming_flag',
        'chargeback', 'chargeback_fraud',
        'payment_irregularity', 'irregularity',
        'aml_flagged', 'aml_flag',  # Anti-Money Laundering
        'market_manipulation', 'market_manipulation_flag',
        'default_fraud', 'default_fraud_flag'
    ]
    fraud_col = None
    
    # Chercher la colonne avec le meilleur match
    for col in column_names:
        col_lower = col.lower()
        for kw in fraud_keywords:
            if kw in col_lower:
                fraud_col = col
                break
        if fraud_col:
            break
    
    if fraud_col and fraud_col in df.columns:
        fraud_info['fraud_col_name'] = fraud_col
        
        # Gérer les colonnes textuelles (yes/no, oui/non)
        if df[fraud_col].dtype == 'object':
            val_map = {'yes': 1, 'oui': 1, 'y': 1, '1': 1, 'true': 1,
                       'no': 0, 'non': 0, 'n': 0, '0': 0, 'false': 0}
            mapped = df[fraud_col].str.lower().map(val_map)
            if not mapped.isna().all():
                fraud_info['fraud_count'] = int(mapped.sum())
                fraud_info['fraud_rate'] = float(mapped.mean())
        else:
            fraud_info['fraud_count'] = int(df[fraud_col].sum())
            fraud_info['fraud_rate'] = float(df[fraud_col].mean())
        
        if fraud_info['fraud_rate'] > 0:
            fraud_info['class_imbalance'] = float((1 - fraud_info['fraud_rate']) / fraud_info['fraud_rate'])
    
    return fraud_info


def extract_dataset_metadata(df: pd.DataFrame, include_detailed: bool = True) -> dict:
    """Extrait les métadonnées enrichies d'un dataset"""
    
    column_names = list(df.columns)
    numerical_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    
    # Métadonnées de base
    meta = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'column_names': column_names,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'n_numerical': len(numerical_cols),
        'n_categorical': len(categorical_cols),
    }
    
    # Détection de colonnes spécifiques
    col_str = ' '.join([col.lower() for col in column_names])
    
    meta['has_amount'] = any('amount' in col.lower() or 'montant' in col.lower() for col in column_names)
    meta['has_timestamp'] = any('time' in col.lower() or 'date' in col.lower() for col in column_names)
    meta['has_merchant'] = any('merchant' in col.lower() or 'commercant' in col.lower() for col in column_names)
    meta['has_card'] = any('card' in col.lower() or 'carte' in col.lower() for col in column_names)
    meta['has_currency'] = any('currency' in col.lower() or 'devise' in col.lower() for col in column_names)
    meta['has_country'] = any('country' in col.lower() or 'pays' in col.lower() for col in column_names)
    meta['has_balance'] = any('balance' in col.lower() or 'solde' in col.lower() for col in column_names)
    meta['has_customer'] = any('customer' in col.lower() or 'client' in col.lower() for col in column_names)
    meta['has_account'] = any('account' in col.lower() or 'compte' in col.lower() for col in column_names)
    
    # Détection de domaine
    meta['domain'] = detect_domain(df, column_names)
    
    # Features temporelles
    meta['temporal_features'] = detect_temporal_features(column_names)
    
    # Informations sur la fraude
    meta['fraud_info'] = detect_fraud_indicators(df, column_names)
    
    if include_detailed:
        # Identifier les colonnes de montant
        amount_cols = [col for col in column_names if any(kw in col.lower() for kw in 
                      ['amount', 'montant', 'value', 'valeur', 'balance', 'solde', 'price', 'prix'])]
        
        # Patterns de montants
        if amount_cols:
            meta['amount_patterns'] = analyze_amount_patterns(df, amount_cols)
        
        # Patterns catégoriels
        if categorical_cols:
            meta['categorical_patterns'] = analyze_categorical_patterns(df, categorical_cols)
    
    # Signature unique du dataset (pour détection de duplicates)
    meta['signature'] = {
        'n_cols': len(column_names),
        'col_hash': hash(tuple(sorted(column_names))) % (10**8),  # Hash des noms de colonnes
        'domain': meta['domain'],
        'fraud_rate_bucket': round(meta['fraud_info']['fraud_rate'], 2) if meta['fraud_info']['fraud_rate'] else 0,
    }
    
    return meta


def main():
    parser = argparse.ArgumentParser(description='Génère les métadonnées enrichies pour tous les modèles AutoML')
    parser.add_argument('--force', action='store_true', help='Régénère même si les métadonnées existent')
    parser.add_argument('--dataset', type=str, help='Générer seulement pour un dataset spécifique (ex: dataset20)')
    args = parser.parse_args()
    
    automl_dir = Path("data/automl_models")
    datasets_dir = Path("data/datasets")
    
    if not automl_dir.exists():
        print("❌ Dossier data/automl_models introuvable!")
        return
    
    if not datasets_dir.exists():
        print("❌ Dossier data/datasets introuvable!")
        return
    
    print("🔍 Génération des métadonnées enrichies pour tous les modèles AutoML...")
    if args.force:
        print("   Mode FORCE activé - Régénération de toutes les métadonnées\n")
    else:
        print("   Mode normal - Skip si métadonnées existent déjà\n")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    updated_count = 0
    
    model_dirs = sorted(automl_dir.iterdir())
    
    # Filtrer si un dataset spécifique est demandé
    if args.dataset:
        model_dirs = [d for d in model_dirs if d.name == args.dataset]
        if not model_dirs:
            print(f"❌ Modèle {args.dataset} introuvable!")
            return
    
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Trouver le dataset source
        dataset_num = model_name.replace('dataset', '')
        dataset_file = datasets_dir / f"Dataset{dataset_num}.csv"
        
        if not dataset_file.exists():
            dataset_file = datasets_dir / f"dataset{dataset_num}.csv"
        
        if not dataset_file.exists():
            print(f"⚠️  {model_name:15s} - Dataset source introuvable")
            skip_count += 1
            continue
        
        metadata_file = model_dir / "dataset_metadata.json"
        
        # Si le fichier existe et pas force, skip
        if metadata_file.exists() and not args.force:
            print(f"ℹ️  {model_name:15s} - Métadonnées existantes (skip)")
            skip_count += 1
            continue
        
        try:
            # Charger le dataset
            df = pd.read_csv(dataset_file)
            
            # Extraire les métadonnées enrichies
            metadata = extract_dataset_metadata(df, include_detailed=True)
            
            # Sauvegarder
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            domain_str = metadata['domain'].replace('_', ' ').title()
            fraud_rate = metadata['fraud_info']['fraud_rate']
            
            if args.force and metadata_file.exists():
                print(f"🔄 {model_name:15s} - Métadonnées mises à jour ({metadata['n_cols']} cols, {domain_str}, fraud={fraud_rate:.1%})")
                updated_count += 1
            else:
                print(f"✅ {model_name:15s} - Métadonnées générées ({metadata['n_cols']} cols, {domain_str}, fraud={fraud_rate:.1%})")
                success_count += 1
            
        except Exception as e:
            print(f"❌ {model_name:15s} - Erreur: {e}")
            error_count += 1
    
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ:")
    print(f"  ✅ Nouvelles:     {success_count}")
    print(f"  🔄 Mises à jour:  {updated_count}")
    print(f"  ℹ️  Skippées:     {skip_count}")
    print(f"  ❌ Erreurs:       {error_count}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
