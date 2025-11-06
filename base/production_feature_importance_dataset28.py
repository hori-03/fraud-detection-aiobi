#!/usr/bin/env python3

"""Script de production pour extraire l'importance des features du modèle XGBoost Dataset28
Version adaptée pour Dataset28 (25K lignes avec fraud 3.61% - Insurance Claims Fraud)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def load_model_and_data():
    """Charger le modèle XGBoost et les données"""
    print("=== CHARGEMENT DU MODÈLE ET DONNÉES ===")
    
    # Chemins - Dataset28
    model_path = Path("data/models/Dataset28_xgb_model.joblib")
    grid_results_path = Path("data/results/Dataset28_grid_search_results.json")
    data_path = Path("data/datasets/Dataset28.csv")
    
    # Vérifier l'existence des fichiers
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé : {model_path}")
    if not grid_results_path.exists():
        raise FileNotFoundError(f"Résultats GridSearch non trouvés : {grid_results_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset non trouvé : {data_path}")
    
    # Charger le modèle
    model = joblib.load(model_path)
    print(f"✓ Modèle chargé depuis {model_path}")
    
    # Charger les résultats GridSearch pour récupérer les paramètres complets
    with open(grid_results_path, 'r', encoding='utf-8') as f:
        grid_results = json.load(f)
    
    # Gérer le format des résultats (liste ou dictionnaire)
    if isinstance(grid_results, list) and len(grid_results) > 0:
        # Format liste : prendre le premier élément (meilleur résultat)
        best_result = grid_results[0]
        best_params = best_result.get('params', {})
        best_score = best_result.get('score', 'N/A')
        grid_results_dict = {'best_params': best_params, 'best_score': best_score, 'results': grid_results}
    else:
        # Format dictionnaire
        best_params = grid_results.get('best_params', {})
        best_score = grid_results.get('best_score', 'N/A')
        grid_results_dict = grid_results
    
    print(f"✓ Paramètres du meilleur modèle : {best_params}")
    
    # Charger les données originales
    df = pd.read_csv(data_path)
    print(f"✓ Dataset chargé : {df.shape}")
    
    return model, best_params, best_score, df

def prepare_features_for_analysis(df):
    """Préparer les features comme dans l'entraînement (Dataset28 - 25K lignes, fraud 3.61%, Insurance Claims Fraud)"""
    print("\n=== PRÉPARATION DES FEATURES ===")
    
    # Feature engineering identique à baseline_xgboost.py Dataset28
    df = df.copy()
    
    # Exclure les colonnes non-features (Dataset28 - target: fraud_indicator, IDs: claim_number, policy_holder_id)
    columns_to_exclude = [
        'target',                      # Target encodé
        'fraud_indicator',             # Target original (Dataset28)
        'market_manipulation_flag',    # Target original (Dataset27)
        'payment_irregularity',        # Target original (Dataset26)
        'suspicious_activity',         # Target original (Dataset25)
        'fraud_alert',                 # Target original (Dataset24)
        'aml_flagged',                 # Target original (Dataset23)
        'skimming_detected',           # Target original (Dataset22)
        'flagged_suspicious',          # Target original (Dataset21)
        'is_fraudulent_transaction',   # Target original (Dataset20, autres)
        'is_fraudulent',               # Target original (Dataset19, autres)
        'claim_number',                # ID de réclamation (Dataset28)
        'policy_holder_id',            # ID détenteur de police (Dataset28)
        'trade_order_id',              # ID de transaction (Dataset27)
        'brokerage_account_id',        # ID compte courtier (Dataset27)
        'payment_reference',           # ID de transaction (Dataset26)
        'loan_account_id',             # ID compte de prêt (Dataset26)
        'crypto_tx_hash',              # ID de transaction (Dataset25)
        'wallet_address_hash',         # Hash de wallet (Dataset25)
        'session_transaction_id',      # ID de transaction (Dataset24)
        'user_id_hash',                # Hash d'utilisateur (Dataset24)
        'payment_order_id',            # ID de transaction (Dataset23)
        'corporate_client_id',         # ID client corporatif (Dataset23)
        'atm_transaction_ref',         # ID de transaction (Dataset22)
        'card_hash',                   # Hash de carte (Dataset22)
        'wire_reference',              # ID de transaction (Dataset21)
        'account_iban_hash',           # Hash de compte (Dataset21)
        'card_transaction_id',         # ID de transaction (Dataset20)
        'card_number_hash',            # Hash de carte (Dataset20)
        'tx_id',                       # ID de transaction (Dataset19, autres)
        'customer_ref',                # ID client (Dataset19, autres)
        'tx_timestamp',                # Timestamp original (Dataset26)
        'date_transaction',            # Date originale (Dataset24)
        'heure_transaction',           # Time original (Dataset24)
        'datetime_tx',                 # Datetime original (autres datasets)
        'timestamp'                    # Datetime parsé
    ]
    
    # Features temporelles depuis tx_timestamp (Dataset28 a tx_timestamp)
    if 'tx_timestamp' in df.columns and df['tx_timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['tx_timestamp'], errors='coerce')
        df['transaction_hour'] = df['timestamp'].dt.hour
        df['transaction_day'] = df['timestamp'].dt.day
        df['transaction_month'] = df['timestamp'].dt.month
        df['transaction_weekday'] = df['timestamp'].dt.weekday
        df['transaction_is_weekend'] = (df['transaction_weekday'] >= 5).astype(int)
        df['is_business_hours'] = ((df['transaction_hour'] >= 8) & (df['transaction_hour'] <= 18)).astype(int)
        df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 6)).astype(int)
        df['is_early_morning'] = ((df['transaction_hour'] >= 0) & (df['transaction_hour'] <= 6)).astype(int)
        print("Features temporelles créées depuis tx_timestamp")
    
    # Dataset28 A des features balance (INSURANCE CLAIMS - balance_before, balance_after)
    if 'balance_before' in df.columns and 'balance_after' in df.columns:
        df['balance_change'] = df['balance_after'] - df['balance_before']
        df['balance_change_pct'] = (df['balance_change'] / (df['balance_before'] + 1)) * 100
        print("Features balance créées (balance_before + balance_after)")
    
    # Features balance + claim_amount (Dataset28 A claim_amount_fcfa)
    if 'balance_after' in df.columns and 'claim_amount_fcfa' in df.columns:
        df['balance_after_claim_ratio'] = df['balance_after'] / (df['claim_amount_fcfa'] + 1)
        print("Features balance_after_claim_ratio créées")
    
    # Features claim_to_balance (Dataset28 A claim_amount_fcfa + balance_before)
    if 'claim_amount_fcfa' in df.columns and 'balance_before' in df.columns:
        df['claim_to_balance_ratio'] = df['claim_amount_fcfa'] / (df['balance_before'] + 1)
        print("Features claim_to_balance_ratio créées")
    
    # Features policy_age (Dataset28 A policy_age_months)
    if 'policy_age_months' in df.columns:
        df['policy_age_log'] = np.log1p(df['policy_age_months'])
        df['is_new_policy'] = (df['policy_age_months'] < 12).astype(int)
        df['is_old_policy'] = (df['policy_age_months'] > 60).astype(int)
        print("Features policy_age créées")
    
    # Features previous_claims (Dataset28 A previous_claims_count)
    if 'previous_claims_count' in df.columns:
        df['has_previous_claims'] = (df['previous_claims_count'] > 0).astype(int)
        df['is_frequent_claimant'] = (df['previous_claims_count'] >= 3).astype(int)
        print("Features previous_claims créées")
    
    # Features processing_time (Dataset28 A processing_time_ms)
    if 'processing_time_ms' in df.columns:
        df['processing_time_log'] = np.log1p(df['processing_time_ms'])
        df['is_slow_processing'] = (df['processing_time_ms'] > df['processing_time_ms'].quantile(0.90)).astype(int)
        print("Features processing_time créées")
    
    # Encodage des variables catégorielles restantes (Dataset28 - INSURANCE CLAIMS FRAUD)
    print("Encodage des variables catégorielles...")
    # Dataset28 utilise 'fraud_indicator' comme target, exclure aussi: claim_number, policy_holder_id, tx_timestamp
    categorical_cols = [col for col in df.select_dtypes(include=['object']).columns 
                        if col not in ['fraud_indicator', 'claim_number', 'policy_holder_id', 'tx_timestamp', 'timestamp']]
    
    for col in categorical_cols:
        print(f"  Encodage de {col}...")
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    # Features amount - SÉPARÉ en deux étapes comme baseline_xgboost.py
    # Étape 1: amount_log, amount_squared, amount_sqrt (Dataset28 utilise 'claim_amount_fcfa')
    if 'claim_amount_fcfa' in df.columns:
        df['amount_log'] = np.log1p(df['claim_amount_fcfa'])
        df['amount_squared'] = df['claim_amount_fcfa'] ** 2
        df['amount_sqrt'] = np.sqrt(df['claim_amount_fcfa'])
    elif 'trade_value_fcfa' in df.columns:
        df['amount_log'] = np.log1p(df['trade_value_fcfa'])
        df['amount_squared'] = df['trade_value_fcfa'] ** 2
        df['amount_sqrt'] = np.sqrt(df['trade_value_fcfa'])
    elif 'monthly_payment_fcfa' in df.columns:
        df['amount_log'] = np.log1p(df['monthly_payment_fcfa'])
        df['amount_squared'] = df['monthly_payment_fcfa'] ** 2
        df['amount_sqrt'] = np.sqrt(df['monthly_payment_fcfa'])
    elif 'amount_fcfa' in df.columns:
        df['amount_log'] = np.log1p(df['amount_fcfa'])
        df['amount_squared'] = df['amount_fcfa'] ** 2
        df['amount_sqrt'] = np.sqrt(df['amount_fcfa'])
    elif 'withdrawal_amount_fcfa' in df.columns:
        df['amount_log'] = np.log1p(df['withdrawal_amount_fcfa'])
        df['amount_squared'] = df['withdrawal_amount_fcfa'] ** 2
        df['amount_sqrt'] = np.sqrt(df['withdrawal_amount_fcfa'])
    elif 'wire_amount_fcfa' in df.columns:
        df['amount_log'] = np.log1p(df['wire_amount_fcfa'])
        df['amount_squared'] = df['wire_amount_fcfa'] ** 2
        df['amount_sqrt'] = np.sqrt(df['wire_amount_fcfa'])
    elif 'transaction_amount_fcfa' in df.columns:
        df['amount_log'] = np.log1p(df['transaction_amount_fcfa'])
        df['amount_squared'] = df['transaction_amount_fcfa'] ** 2
        df['amount_sqrt'] = np.sqrt(df['transaction_amount_fcfa'])
    elif 'tx_amount_xof' in df.columns:
        df['amount_log'] = np.log1p(df['tx_amount_xof'])
        df['amount_squared'] = df['tx_amount_xof'] ** 2
        df['amount_sqrt'] = np.sqrt(df['tx_amount_xof'])
    
    # Étape 2: Binning amount (Dataset28 utilise 'claim_amount_fcfa')
    if 'claim_amount_fcfa' in df.columns:
        df['amount_zscore'] = (df['claim_amount_fcfa'] - df['claim_amount_fcfa'].mean()) / df['claim_amount_fcfa'].std()
        df['amount_bin'] = pd.cut(df['claim_amount_fcfa'], bins=10, labels=False)
        df['is_high_amount'] = (df['claim_amount_fcfa'] > df['claim_amount_fcfa'].quantile(0.95)).astype(int)
        df['is_low_amount'] = (df['claim_amount_fcfa'] < df['claim_amount_fcfa'].quantile(0.05)).astype(int)
        df['is_round_amount'] = (df['claim_amount_fcfa'] % 10000 == 0).astype(int)
        print("Features claim_amount_fcfa créées")
    elif 'trade_value_fcfa' in df.columns:
        df['amount_zscore'] = (df['trade_value_fcfa'] - df['trade_value_fcfa'].mean()) / df['trade_value_fcfa'].std()
        df['amount_bin'] = pd.cut(df['trade_value_fcfa'], bins=10, labels=False)
        df['is_high_amount'] = (df['trade_value_fcfa'] > df['trade_value_fcfa'].quantile(0.95)).astype(int)
        df['is_low_amount'] = (df['trade_value_fcfa'] < df['trade_value_fcfa'].quantile(0.05)).astype(int)
        df['is_round_amount'] = (df['trade_value_fcfa'] % 10000 == 0).astype(int)
        print("Features trade_value_fcfa créées")
    elif 'monthly_payment_fcfa' in df.columns:
        df['amount_zscore'] = (df['monthly_payment_fcfa'] - df['monthly_payment_fcfa'].mean()) / df['monthly_payment_fcfa'].std()
        df['amount_bin'] = pd.cut(df['monthly_payment_fcfa'], bins=10, labels=False)
        df['is_high_amount'] = (df['monthly_payment_fcfa'] > df['monthly_payment_fcfa'].quantile(0.95)).astype(int)
        df['is_low_amount'] = (df['monthly_payment_fcfa'] < df['monthly_payment_fcfa'].quantile(0.05)).astype(int)
        df['is_round_amount'] = (df['monthly_payment_fcfa'] % 10000 == 0).astype(int)
        print("Features monthly_payment_fcfa créées")
    elif 'amount_fcfa' in df.columns:
        df['amount_bin'] = pd.cut(df['amount_fcfa'], bins=10, labels=False)
        # is_high_amount existe déjà dans Dataset24, mais recréons pour cohérence
        if 'is_high_amount' not in df.columns:
            df['is_high_amount'] = (df['amount_fcfa'] > df['amount_fcfa'].quantile(0.95)).astype(int)
        df['is_low_amount'] = (df['amount_fcfa'] < df['amount_fcfa'].quantile(0.05)).astype(int)
        df['is_round_amount'] = (df['amount_fcfa'] % 10000 == 0).astype(int)
        print("Features amount_fcfa créées")
    elif 'withdrawal_amount_fcfa' in df.columns:
        df['amount_bin'] = pd.cut(df['withdrawal_amount_fcfa'], bins=10, labels=False)
        # is_high_amount existe déjà dans Dataset22, mais recréons pour cohérence
        if 'is_high_amount' not in df.columns:
            df['is_high_amount'] = (df['withdrawal_amount_fcfa'] > df['withdrawal_amount_fcfa'].quantile(0.95)).astype(int)
        df['is_low_amount'] = (df['withdrawal_amount_fcfa'] < df['withdrawal_amount_fcfa'].quantile(0.05)).astype(int)
        df['is_round_amount'] = (df['withdrawal_amount_fcfa'] % 10000 == 0).astype(int)
        print("Features withdrawal_amount_fcfa créées")
    elif 'wire_amount_fcfa' in df.columns:
        df['amount_bin'] = pd.cut(df['wire_amount_fcfa'], bins=10, labels=False)
        # is_high_amount existe déjà dans Dataset21, mais recréons pour cohérence
        if 'is_high_amount' not in df.columns:
            df['is_high_amount'] = (df['wire_amount_fcfa'] > df['wire_amount_fcfa'].quantile(0.95)).astype(int)
        df['is_low_amount'] = (df['wire_amount_fcfa'] < df['wire_amount_fcfa'].quantile(0.05)).astype(int)
        df['is_round_amount'] = (df['wire_amount_fcfa'] % 10000 == 0).astype(int)
        print("Features wire_amount_fcfa créées")
    elif 'transaction_amount_fcfa' in df.columns:
        df['amount_bin'] = pd.cut(df['transaction_amount_fcfa'], bins=10, labels=False)
        if 'is_high_amount' not in df.columns:
            df['is_high_amount'] = (df['transaction_amount_fcfa'] > df['transaction_amount_fcfa'].quantile(0.95)).astype(int)
        df['is_low_amount'] = (df['transaction_amount_fcfa'] < df['transaction_amount_fcfa'].quantile(0.05)).astype(int)
        df['is_round_amount'] = (df['transaction_amount_fcfa'] % 10000 == 0).astype(int)
        print("Features transaction_amount_fcfa créées")
    elif 'amount_fcfa' in df.columns:
        df['amount_log'] = np.log1p(df['amount_fcfa'])
        df['amount_squared'] = df['amount_fcfa'] ** 2
        df['amount_sqrt'] = np.sqrt(df['amount_fcfa'])
        df['amount_bin'] = pd.cut(df['amount_fcfa'], bins=10, labels=False)
        df['is_high_amount'] = (df['amount_fcfa'] > df['amount_fcfa'].quantile(0.95)).astype(int)
        df['is_low_amount'] = (df['amount_fcfa'] < df['amount_fcfa'].quantile(0.05)).astype(int)
        df['is_round_amount'] = (df['amount_fcfa'] % 10000 == 0).astype(int)
        print("Features amount_fcfa créées")
    elif 'tx_amount_xof' in df.columns:
        df['amount_log'] = np.log1p(df['tx_amount_xof'])
        df['amount_squared'] = df['tx_amount_xof'] ** 2
        df['amount_sqrt'] = np.sqrt(df['tx_amount_xof'])
        df['amount_bin'] = pd.cut(df['tx_amount_xof'], bins=10, labels=False)
        df['is_high_amount'] = (df['tx_amount_xof'] > df['tx_amount_xof'].quantile(0.95)).astype(int)
        df['is_low_amount'] = (df['tx_amount_xof'] < df['tx_amount_xof'].quantile(0.05)).astype(int)
        df['is_round_amount'] = (df['tx_amount_xof'] % 10000 == 0).astype(int)
        print("Features tx_amount_xof créées")
    
    # Features âge client et ancienneté compte (Dataset21 N'A PAS cust_age ni account_tenure_days)
    if 'cust_age' in df.columns:
        df['age_log'] = np.log1p(df['cust_age'])
        df['is_young_customer'] = (df['cust_age'] < 25).astype(int)
        df['is_senior_customer'] = (df['cust_age'] > 60).astype(int)
    
    if 'account_tenure_days' in df.columns:
        df['tenure_log'] = np.log1p(df['account_tenure_days'])
        df['is_new_account'] = (df['account_tenure_days'] < 30).astype(int)
        df['is_old_account'] = (df['account_tenure_days'] > 365).astype(int)
    
    # Features spécifiques carte bancaire (Dataset20)
    if 'transaction_velocity_24h' in df.columns:
        df['velocity_log'] = np.log1p(df['transaction_velocity_24h'])
        df['is_high_velocity'] = (df['transaction_velocity_24h'] > df['transaction_velocity_24h'].quantile(0.90)).astype(int)
    
    if 'distance_from_home' in df.columns:
        df['distance_log'] = np.log1p(df['distance_from_home'])
        df['is_far_from_home'] = (df['distance_from_home'] > df['distance_from_home'].quantile(0.90)).astype(int)
        df['is_very_far'] = (df['distance_from_home'] > df['distance_from_home'].quantile(0.95)).astype(int)
    
    # Interactions entre features (EXACTEMENT COMME baseline_xgboost.py)
    if 'transaction_hour' in df.columns and 'amount_log' in df.columns:
        df['hour_amount_interaction'] = df['transaction_hour'] * df['amount_log']
    
    if 'merchant_category' in df.columns and 'amount_log' in df.columns:
        df['merchant_amount_interaction'] = df['merchant_category'] * df['amount_log']
    elif 'cust_region' in df.columns and 'amount_log' in df.columns:
        df['region_amount_interaction'] = df['cust_region'] * df['amount_log']
    elif 'province' in df.columns and 'amount_log' in df.columns:
        df['province_amount_interaction'] = df['province'] * df['amount_log']
    
    # Interactions spécifiques carte bancaire (Dataset20)
    if 'is_foreign_currency' in df.columns and 'is_international' in df.columns:
        df['foreign_and_international'] = df['is_foreign_currency'] * df['is_international']
    
    if 'is_night' in df.columns and 'is_high_amount' in df.columns:
        df['night_and_high_amount'] = df['is_night'] * df['is_high_amount']
    elif 'is_night_tx' in df.columns and 'is_high_amount' in df.columns:
        df['night_and_high_amount'] = df['is_night_tx'] * df['is_high_amount']
    
    if 'transaction_velocity_24h' in df.columns and 'withdrawal_amount_fcfa' in df.columns:
        df['velocity_x_amount'] = df['transaction_velocity_24h'] * df['withdrawal_amount_fcfa'] / 1000000
    elif 'transaction_velocity_24h' in df.columns and 'wire_amount_fcfa' in df.columns:
        df['velocity_x_amount'] = df['transaction_velocity_24h'] * df['wire_amount_fcfa'] / 1000000
    elif 'transaction_velocity_24h' in df.columns and 'transaction_amount_fcfa' in df.columns:
        df['velocity_x_amount'] = df['transaction_velocity_24h'] * df['transaction_amount_fcfa'] / 1000000
    elif 'transaction_velocity_24h' in df.columns and 'amount_log' in df.columns:
        df['velocity_x_amount_log'] = df['transaction_velocity_24h'] * df['amount_log']
    
    if 'tx_method' in df.columns and 'amount_log' in df.columns:
        df['method_amount_interaction'] = df['tx_method'] * df['amount_log']
    
    # Bruitage pour features quantitatives (Dataset27 utilise trade_value_fcfa)
    np.random.seed(42)
    if 'trade_value_fcfa' in df.columns:
        df['amount_noisy'] = df['trade_value_fcfa'] + np.random.normal(0, 0.01 * df['trade_value_fcfa'].std(), len(df))
    elif 'monthly_payment_fcfa' in df.columns:
        df['amount_noisy'] = df['monthly_payment_fcfa'] + np.random.normal(0, 0.01 * df['monthly_payment_fcfa'].std(), len(df))
    elif 'amount_fcfa' in df.columns:
        df['amount_noisy'] = df['amount_fcfa'] + np.random.normal(0, 0.01 * df['amount_fcfa'].std(), len(df))
    elif 'withdrawal_amount_fcfa' in df.columns:
        df['amount_noisy'] = df['withdrawal_amount_fcfa'] + np.random.normal(0, 0.01 * df['withdrawal_amount_fcfa'].std(), len(df))
    elif 'wire_amount_fcfa' in df.columns:
        df['amount_noisy'] = df['wire_amount_fcfa'] + np.random.normal(0, 0.01 * df['wire_amount_fcfa'].std(), len(df))
    elif 'transaction_amount_fcfa' in df.columns:
        df['amount_noisy'] = df['transaction_amount_fcfa'] + np.random.normal(0, 0.01 * df['transaction_amount_fcfa'].std(), len(df))
    elif 'tx_amount_xof' in df.columns:
        df['amount_noisy'] = df['tx_amount_xof'] + np.random.normal(0, 0.01 * df['tx_amount_xof'].std(), len(df))
    
    # Target (Dataset27 utilise 'market_manipulation_flag')
    if 'market_manipulation_flag' in df.columns:
        df['target'] = df['market_manipulation_flag'].astype(int)
        print("Target créé depuis market_manipulation_flag")
    elif 'payment_irregularity' in df.columns:
        df['target'] = df['payment_irregularity'].astype(int)
        print("Target créé depuis payment_irregularity")
    elif 'suspicious_activity' in df.columns:
        df['target'] = df['suspicious_activity'].astype(int)
        print("Target créé depuis suspicious_activity")
    elif 'fraud_alert' in df.columns:
        df['target'] = df['fraud_alert'].astype(int)
        print("Target créé depuis fraud_alert")
    elif 'aml_flagged' in df.columns:
        df['target'] = df['aml_flagged'].astype(int)
        print("Target créé depuis aml_flagged")
    elif 'skimming_detected' in df.columns:
        df['target'] = df['skimming_detected'].astype(int)
        print("Target créé depuis skimming_detected")
    elif 'flagged_suspicious' in df.columns:
        df['target'] = df['flagged_suspicious'].astype(int)
        print("Target créé depuis flagged_suspicious")
    elif 'is_fraudulent' in df.columns:
        df['target'] = df['is_fraudulent'].astype(int)
        print("Target créé depuis is_fraudulent")
    
    # Séparer features et target
    feature_columns = []
    for col in df.columns:
        if col not in columns_to_exclude and col not in ['target', 'timestamp']:
            # Convertir en numérique si nécessaire
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0)
                except:
                    continue
            feature_columns.append(col)
    
    X = df[feature_columns]
    y = df['target'] if 'target' in df.columns else None
    
    print(f"✓ Features préparées : {X.shape[1]} features, {X.shape[0]} samples")
    
    return X, y, feature_columns

def extract_feature_importance(model, feature_names):
    """Extraire l'importance des features du modèle avec structure harmonisée"""
    print("\n=== EXTRACTION IMPORTANCE DES FEATURES ===")
    
    # Récupérer l'importance des features
    importances = getattr(model, 'feature_importances_', None)
    if importances is None:
        raise ValueError("Le modèle n'a pas d'attribut 'feature_importances_'")

    importances = np.array(importances)
    feature_names_list = list(feature_names)

    # Vérifier la cohérence longueur(importances) vs longueur(features)
    if len(importances) != len(feature_names_list):
        print(f"⚠️  Incohérence détectée: importances ({len(importances)}) vs features ({len(feature_names_list)})")
        # Ajuster en prenant le minimum des deux pour éviter ValueError
        min_len = min(len(importances), len(feature_names_list))
        importances = importances[:min_len]
        feature_names_list = feature_names_list[:min_len]
        print(f"   → Ajustement effectué: utilisation des {min_len} premières features pour l'analyse")

    # Créer un DataFrame avec les importances
    feature_importance_df = (
        pd.DataFrame({'feature': feature_names_list, 'importance': importances})
        .sort_values('importance', ascending=False)
        .reset_index(drop=True)
    )

    # Calculs statistiques harmonisés
    total_importance = feature_importance_df['importance'].sum()
    if total_importance == 0:
        feature_importance_df['importance_normalized'] = 0.0
        feature_importance_df['importance_percentage'] = 0.0
    else:
        feature_importance_df['importance_normalized'] = feature_importance_df['importance'] / total_importance
        feature_importance_df['importance_percentage'] = feature_importance_df['importance_normalized'] * 100

    print(f"✓ {len(feature_importance_df)} features analysées")
    print(f"\nTop 10 features les plus importantes:")
    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance_percentage']:.2f}%")

    return feature_importance_df

def save_feature_importance(feature_importance_df, best_params, best_score, dataset_name="Dataset27"):
    """Sauvegarder l'importance des features en JSON avec structure harmonisée Dataset27"""
    print("\n=== SAUVEGARDE DES RÉSULTATS ===")
    
    output_path = Path(f"data/Feature_importance/{dataset_name}_production_feature_importance.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Structure harmonisée du JSON
    output_data = {
        "metadata": {
            "dataset": dataset_name,
            "generation_date": datetime.now().isoformat(),
            "total_features": len(feature_importance_df),
            "best_model_score": float(best_score) if isinstance(best_score, (int, float)) else best_score
        },
        "best_params": best_params,
        "feature_importance": []
    }
    
    # Ajouter les features avec toutes les métriques
    for _, row in feature_importance_df.iterrows():
        feature_info = {
            "feature": row['feature'],
            "importance": float(row['importance']),
            "importance_normalized": float(row['importance_normalized']),
            "importance_percentage": float(row['importance_percentage'])
        }
        output_data["feature_importance"].append(feature_info)
    
    # Calculer le cumul pour identifier le seuil 80%
    cumulative = 0
    for feature_info in output_data["feature_importance"]:
        cumulative += feature_info["importance_percentage"]
        feature_info["importance_cumulative"] = cumulative
    
    # Sauvegarder en JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Importance des features sauvegardée : {output_path}")
    
    # Afficher le nombre de features nécessaires pour 80% d'importance
    threshold = 80.0
    features_for_80 = sum(1 for f in output_data["feature_importance"] if f["importance_cumulative"] <= threshold) + 1
    print(f"\n📊 Analyse:")
    print(f"   • {features_for_80} features suffisent pour expliquer {threshold}% de l'importance totale")
    print(f"   • Top 3 features: {', '.join([f['feature'] for f in output_data['feature_importance'][:3]])}")
    
    return output_path

def main():
    """Fonction principale"""
    print("=" * 80)
    print(" EXTRACTION IMPORTANCE DES FEATURES - Dataset28 (25K lignes, 3.61% fraud) ")
    print("=" * 80)
    
    try:
        # 1. Charger modèle et données
        model, best_params, best_score, df = load_model_and_data()
        
        # 2. Préparer les features (identique à l'entraînement)
        X, y, feature_names = prepare_features_for_analysis(df)
        
        # 3. Extraire l'importance des features
        feature_importance_df = extract_feature_importance(model, feature_names)
        
        # 4. Sauvegarder les résultats
        output_path = save_feature_importance(feature_importance_df, best_params, best_score, "Dataset28")
        
        print("\n" + "=" * 80)
        print(" ✓ EXTRACTION TERMINÉE AVEC SUCCÈS ")
        print("=" * 80)
        print(f"\nFichier généré : {output_path}")
        
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
