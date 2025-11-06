2

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import time
import os


# Charger le dataset avec optimisations - DATASET30 (LOAN DEFAULT FRAUD)
# === CONFIGURATION DATASET ===
DATA_PATH = 'data/datasets/Dataset30.csv'
DATASET_NAME = os.path.splitext(os.path.basename(DATA_PATH))[0]  # Extrait "Dataset30"
PARQUET_PATH = f'data/parquet/{DATASET_NAME}.parquet'

print("Chargement du dataset...")
start_time = time.time()

# Utiliser Parquet si disponible (plus rapide)
if os.path.exists(PARQUET_PATH):
    print("Chargement depuis Parquet (optimisé)")
    df = pd.read_parquet(PARQUET_PATH)
else:
    print("Chargement depuis CSV et conversion Parquet")
    df = pd.read_csv(DATA_PATH)
    # Sauvegarder en Parquet pour les prochaines fois
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Fichier Parquet sauvegardé : {PARQUET_PATH}")

print(f"Dataset chargé en {time.time() - start_time:.2f}s")


# Afficher un aperçu
df.info()
print(df.head())

# Afficher les premières valeurs brutes de fraude (Dataset30 - target: 'default_fraud_flag')
print("Premières valeurs brutes de default_fraud_flag :", df['default_fraud_flag'].head(20).tolist())
print("Colonnes disponibles:", df.columns.tolist())

# Gestion des valeurs manquantes
df = df.fillna(df.median(numeric_only=True))

# FEATURE ENGINEERING D'ABORD (avant encodage) - Dataset30 (LOAN DEFAULT FRAUD)
print("Feature engineering préliminaire (avant encodage) - Dataset30...")

# Features credit risk (Dataset30 a debt_to_income_ratio, number_of_delinquencies, annual_income_k_usd)
if 'debt_to_income_ratio' in df.columns:
    df['debt_income_log'] = np.log1p(df['debt_to_income_ratio'])
    df['is_high_debt'] = (df['debt_to_income_ratio'] > df['debt_to_income_ratio'].quantile(0.75)).astype(int)
    df['is_low_debt'] = (df['debt_to_income_ratio'] < df['debt_to_income_ratio'].quantile(0.25)).astype(int)

if 'number_of_delinquencies' in df.columns:
    df['has_delinquency'] = (df['number_of_delinquencies'] > 0).astype(int)
    df['delinquency_log'] = np.log1p(df['number_of_delinquencies'])

if 'annual_income_k_usd' in df.columns:
    df['income_log'] = np.log1p(df['annual_income_k_usd'])
    df['is_low_income'] = (df['annual_income_k_usd'] < df['annual_income_k_usd'].quantile(0.25)).astype(int)
    if 'loan_amount_fcfa' in df.columns:
        # Ratio loan/income (converti en même unité)
        df['loan_to_income_ratio'] = df['loan_amount_fcfa'] / (df['annual_income_k_usd'] * 1000 + 1)

# Encodage des variables catégorielles avec optimisation (Dataset30 - LOAN DEFAULT FRAUD)
print("Encodage des variables catégorielles...")
# Dataset30 utilise 'default_fraud_flag' comme target, et a: loan_transaction_id, borrower_id, tx_timestamp
categorical_cols = [col for col in df.select_dtypes(include=['object']).columns 
                    if col not in ['default_fraud_flag', 'loan_transaction_id', 'borrower_id', 'tx_timestamp']]

for col in categorical_cols:
    print(f"  Encodage de {col}...")
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Feature engineering avancé Dataset30 (LOAN DEFAULT FRAUD)
print("Feature engineering avancé pour Dataset30...")

# Features temporelles depuis tx_timestamp (Dataset30 a tx_timestamp)
if 'tx_timestamp' in df.columns and df['tx_timestamp'].dtype == 'object':
    # Parser tx_timestamp - Dataset30
    df['timestamp'] = pd.to_datetime(df['tx_timestamp'], errors='coerce')
    
    # Features temporelles Dataset30
    df['transaction_hour'] = df['timestamp'].dt.hour
    df['transaction_day'] = df['timestamp'].dt.day
    df['transaction_month'] = df['timestamp'].dt.month
    df['transaction_weekday'] = df['timestamp'].dt.weekday
    df['transaction_is_weekend'] = (df['transaction_weekday'] >= 5).astype(int)
    df['is_business_hours'] = ((df['transaction_hour'] >= 8) & (df['transaction_hour'] <= 18)).astype(int)
    df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 6)).astype(int)
    df['is_early_morning'] = ((df['transaction_hour'] >= 0) & (df['transaction_hour'] <= 6)).astype(int)
elif 'date_transaction' in df.columns and df['date_transaction'].dtype == 'object':
    # Parser date_transaction - Dataset19
    df['timestamp'] = pd.to_datetime(df['date_transaction'], errors='coerce')
    
    # Features temporelles Dataset19 (hour pré-extrait, PAS de weekday)
    df['transaction_hour'] = df['hour'] if 'hour' in df.columns else df['timestamp'].dt.hour
    df['transaction_day'] = df['timestamp'].dt.day
    df['transaction_month'] = df['timestamp'].dt.month
    df['transaction_weekday'] = df['timestamp'].dt.weekday  # Dataset19 n'a PAS weekday pré-extrait
    df['transaction_is_weekend'] = (df['transaction_weekday'] >= 5).astype(int)
    df['is_business_hours'] = ((df['transaction_hour'] >= 8) & (df['transaction_hour'] <= 18)).astype(int)
    df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 6)).astype(int)
    df['is_early_morning'] = ((df['transaction_hour'] >= 0) & (df['transaction_hour'] <= 6)).astype(int)
elif 'tx_timestamp' in df.columns and df['tx_timestamp'].dtype == 'object':
    # Parser tx_timestamp - Dataset18
    df['timestamp'] = pd.to_datetime(df['tx_timestamp'], errors='coerce')
    
    # Features temporelles Dataset18 (hour et weekday DEJA extraits)
    df['transaction_hour'] = df['hour'] if 'hour' in df.columns else df['timestamp'].dt.hour
    df['transaction_day'] = df['timestamp'].dt.day
    df['transaction_month'] = df['timestamp'].dt.month
    df['transaction_weekday'] = df['weekday'] if 'weekday' in df.columns else df['timestamp'].dt.weekday
    df['transaction_is_weekend'] = (df['transaction_weekday'] >= 5).astype(int)
    df['is_business_hours'] = ((df['transaction_hour'] >= 8) & (df['transaction_hour'] <= 18)).astype(int)
    df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 6)).astype(int)
    df['is_early_morning'] = ((df['transaction_hour'] >= 0) & (df['transaction_hour'] <= 6)).astype(int)
elif 'datetime_tx' in df.columns:
    # Dataset17 a datetime_tx (à parser) + hour ET weekday pré-extraits
    df['timestamp'] = pd.to_datetime(df['datetime_tx'], errors='coerce')
    
    # Features temporelles Dataset17 (hour et weekday DEJA extraits)
    df['transaction_hour'] = df['hour'] if 'hour' in df.columns else df['timestamp'].dt.hour
    df['transaction_day'] = df['timestamp'].dt.day
    df['transaction_month'] = df['timestamp'].dt.month
    df['transaction_weekday'] = df['weekday'] if 'weekday' in df.columns else df['timestamp'].dt.weekday
    df['transaction_is_weekend'] = (df['transaction_weekday'] >= 5).astype(int)
    df['is_business_hours'] = ((df['transaction_hour'] >= 8) & (df['transaction_hour'] <= 18)).astype(int)
    df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 6)).astype(int)
    df['is_early_morning'] = ((df['transaction_hour'] >= 0) & (df['transaction_hour'] <= 6)).astype(int)
elif 'hour' in df.columns and 'weekday' in df.columns:
    # Dataset15 a hour et weekday pré-extraits
    df['transaction_hour'] = df['hour']
    df['transaction_weekday'] = df['weekday']
    df['transaction_is_weekend'] = (df['transaction_weekday'] >= 5).astype(int) if 'is_weekend' not in df.columns else df['is_weekend']
    df['is_business_hours'] = ((df['transaction_hour'] >= 8) & (df['transaction_hour'] <= 18)).astype(int)
    df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 6)).astype(int)
    df['is_early_morning'] = ((df['transaction_hour'] >= 0) & (df['transaction_hour'] <= 6)).astype(int)

# Features amount (Dataset24 utilise 'amount_fcfa')
if 'amount_fcfa' in df.columns:
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
elif 'loan_amount_fcfa' in df.columns:
    df['amount_log'] = np.log1p(df['loan_amount_fcfa'])
    df['amount_squared'] = df['loan_amount_fcfa'] ** 2
    df['amount_sqrt'] = np.sqrt(df['loan_amount_fcfa'])
elif 'tx_amount_xof' in df.columns:
    df['amount_log'] = np.log1p(df['tx_amount_xof'])
    df['amount_squared'] = df['tx_amount_xof'] ** 2
    df['amount_sqrt'] = np.sqrt(df['tx_amount_xof'])

# Binning amount (Dataset28 utilise 'claim_amount_fcfa')
if 'claim_amount_fcfa' in df.columns:
    df['amount_log'] = np.log1p(df['claim_amount_fcfa'])
    df['amount_zscore'] = (df['claim_amount_fcfa'] - df['claim_amount_fcfa'].mean()) / df['claim_amount_fcfa'].std()
    df['amount_bin'] = pd.cut(df['claim_amount_fcfa'], bins=10, labels=False)
    df['is_high_amount'] = (df['claim_amount_fcfa'] > df['claim_amount_fcfa'].quantile(0.95)).astype(int)
    df['is_low_amount'] = (df['claim_amount_fcfa'] < df['claim_amount_fcfa'].quantile(0.05)).astype(int)
    df['is_round_amount'] = (df['claim_amount_fcfa'] % 10000 == 0).astype(int)
elif 'trade_value_fcfa' in df.columns:
    df['amount_log'] = np.log1p(df['trade_value_fcfa'])
    df['amount_zscore'] = (df['trade_value_fcfa'] - df['trade_value_fcfa'].mean()) / df['trade_value_fcfa'].std()
    df['amount_bin'] = pd.cut(df['trade_value_fcfa'], bins=10, labels=False)
    df['is_high_amount'] = (df['trade_value_fcfa'] > df['trade_value_fcfa'].quantile(0.95)).astype(int)
    df['is_low_amount'] = (df['trade_value_fcfa'] < df['trade_value_fcfa'].quantile(0.05)).astype(int)
    df['is_round_amount'] = (df['trade_value_fcfa'] % 10000 == 0).astype(int)
elif 'monthly_payment_fcfa' in df.columns:
    df['amount_log'] = np.log1p(df['monthly_payment_fcfa'])
    df['amount_zscore'] = (df['monthly_payment_fcfa'] - df['monthly_payment_fcfa'].mean()) / df['monthly_payment_fcfa'].std()
    df['amount_bin'] = pd.cut(df['monthly_payment_fcfa'], bins=10, labels=False)
    df['is_high_amount'] = (df['monthly_payment_fcfa'] > df['monthly_payment_fcfa'].quantile(0.95)).astype(int)
    df['is_low_amount'] = (df['monthly_payment_fcfa'] < df['monthly_payment_fcfa'].quantile(0.05)).astype(int)
    df['is_round_amount'] = (df['monthly_payment_fcfa'] % 10000 == 0).astype(int)
elif 'amount_fcfa' in df.columns:
    df['amount_bin'] = pd.cut(df['amount_fcfa'], bins=10, labels=False)
    # is_high_amount peut déjà exister dans Dataset24, recréons si absent
    if 'is_high_amount' not in df.columns:
        df['is_high_amount'] = (df['amount_fcfa'] > df['amount_fcfa'].quantile(0.95)).astype(int)
    df['is_low_amount'] = (df['amount_fcfa'] < df['amount_fcfa'].quantile(0.05)).astype(int)
    df['is_round_amount'] = (df['amount_fcfa'] % 10000 == 0).astype(int)
elif 'withdrawal_amount_fcfa' in df.columns:
    df['amount_bin'] = pd.cut(df['withdrawal_amount_fcfa'], bins=10, labels=False)
    # is_high_amount existe déjà dans Dataset22, mais recréons pour cohérence
    if 'is_high_amount' not in df.columns:
        df['is_high_amount'] = (df['withdrawal_amount_fcfa'] > df['withdrawal_amount_fcfa'].quantile(0.95)).astype(int)
    df['is_low_amount'] = (df['withdrawal_amount_fcfa'] < df['withdrawal_amount_fcfa'].quantile(0.05)).astype(int)
    df['is_round_amount'] = (df['withdrawal_amount_fcfa'] % 10000 == 0).astype(int)
elif 'wire_amount_fcfa' in df.columns:
    df['amount_bin'] = pd.cut(df['wire_amount_fcfa'], bins=10, labels=False)
    if 'is_high_amount' not in df.columns:
        df['is_high_amount'] = (df['wire_amount_fcfa'] > df['wire_amount_fcfa'].quantile(0.95)).astype(int)
    df['is_low_amount'] = (df['wire_amount_fcfa'] < df['wire_amount_fcfa'].quantile(0.05)).astype(int)
    df['is_round_amount'] = (df['wire_amount_fcfa'] % 10000 == 0).astype(int)
elif 'transaction_amount_fcfa' in df.columns:
    df['amount_bin'] = pd.cut(df['transaction_amount_fcfa'], bins=10, labels=False)
    if 'is_high_amount' not in df.columns:
        df['is_high_amount'] = (df['transaction_amount_fcfa'] > df['transaction_amount_fcfa'].quantile(0.95)).astype(int)
    df['is_low_amount'] = (df['transaction_amount_fcfa'] < df['transaction_amount_fcfa'].quantile(0.05)).astype(int)
    df['is_round_amount'] = (df['transaction_amount_fcfa'] % 10000 == 0).astype(int)
elif 'loan_amount_fcfa' in df.columns:
    df['amount_bin'] = pd.cut(df['loan_amount_fcfa'], bins=10, labels=False)
    if 'is_high_amount' not in df.columns:
        df['is_high_amount'] = (df['loan_amount_fcfa'] > df['loan_amount_fcfa'].quantile(0.95)).astype(int)
    df['is_low_amount'] = (df['loan_amount_fcfa'] < df['loan_amount_fcfa'].quantile(0.05)).astype(int)
    df['is_round_amount'] = (df['loan_amount_fcfa'] % 10000 == 0).astype(int)
elif 'tx_amount_xof' in df.columns:
    df['amount_bin'] = pd.cut(df['tx_amount_xof'], bins=10, labels=False)
    df['is_high_amount'] = (df['tx_amount_xof'] > df['tx_amount_xof'].quantile(0.95)).astype(int)
    df['is_low_amount'] = (df['tx_amount_xof'] < df['tx_amount_xof'].quantile(0.05)).astype(int)
    df['is_round_amount'] = (df['tx_amount_xof'] % 10000 == 0).astype(int)

# Features âge client et ancienneté compte (Dataset20 utilise 'cust_age' et 'account_tenure_days')
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



# Interactions entre features (Dataset20 utilise merchant_category, transaction_amount_fcfa)
if 'transaction_hour' in df.columns and 'amount_log' in df.columns:
    df['hour_amount_interaction'] = df['transaction_hour'] * df['amount_log']

if 'merchant_category' in df.columns and 'amount_log' in df.columns:
    # merchant_category est déjà encodé numériquement (Dataset20)
    df['merchant_amount_interaction'] = df['merchant_category'] * df['amount_log']
elif 'cust_region' in df.columns and 'amount_log' in df.columns:
    # cust_region est déjà encodé numériquement (Dataset19)
    df['region_amount_interaction'] = df['cust_region'] * df['amount_log']
elif 'province' in df.columns and 'amount_log' in df.columns:
    # province est déjà encodé numériquement (Dataset18)
    df['province_amount_interaction'] = df['province'] * df['amount_log']

# Interactions spécifiques carte bancaire (Dataset20)
if 'is_foreign_currency' in df.columns and 'is_international' in df.columns:
    df['foreign_and_international'] = df['is_foreign_currency'] * df['is_international']

if 'is_night' in df.columns and 'is_high_amount' in df.columns:
    df['night_and_high_amount'] = df['is_night'] * df['is_high_amount']
elif 'is_night_tx' in df.columns and 'is_high_amount' in df.columns:
    df['night_and_high_amount'] = df['is_night_tx'] * df['is_high_amount']

if 'transaction_velocity_24h' in df.columns and 'transaction_amount_fcfa' in df.columns:
    df['velocity_x_amount'] = df['transaction_velocity_24h'] * df['wire_amount_fcfa'] / 1000000
elif 'transaction_velocity_24h' in df.columns and 'transaction_amount_fcfa' in df.columns:
    df['velocity_x_amount'] = df['transaction_velocity_24h'] * df['transaction_amount_fcfa'] / 1000000
elif 'transaction_velocity_24h' in df.columns and 'amount_log' in df.columns:
    df['velocity_x_amount_log'] = df['transaction_velocity_24h'] * df['amount_log']

if 'tx_method' in df.columns and 'amount_log' in df.columns:
    # tx_method est déjà encodé numériquement
    df['method_amount_interaction'] = df['tx_method'] * df['amount_log']

# Bruitage pour features quantitatives (Dataset24 utilise amount_fcfa)
np.random.seed(42)
if 'amount_fcfa' in df.columns:
    df['amount_noisy'] = df['amount_fcfa'] + np.random.normal(0, 0.01 * df['amount_fcfa'].std(), len(df))
elif 'withdrawal_amount_fcfa' in df.columns:
    df['amount_noisy'] = df['withdrawal_amount_fcfa'] + np.random.normal(0, 0.01 * df['withdrawal_amount_fcfa'].std(), len(df))
elif 'wire_amount_fcfa' in df.columns:
    df['amount_noisy'] = df['wire_amount_fcfa'] + np.random.normal(0, 0.01 * df['wire_amount_fcfa'].std(), len(df))
elif 'transaction_amount_fcfa' in df.columns:
    df['amount_noisy'] = df['transaction_amount_fcfa'] + np.random.normal(0, 0.01 * df['transaction_amount_fcfa'].std(), len(df))
elif 'loan_amount_fcfa' in df.columns:
    df['amount_noisy'] = df['loan_amount_fcfa'] + np.random.normal(0, 0.01 * df['loan_amount_fcfa'].std(), len(df))
elif 'amount_fcfa' in df.columns:
    df['amount_noisy'] = df['amount_fcfa'] + np.random.normal(0, 0.01 * df['amount_fcfa'].std(), len(df))
elif 'tx_amount_xof' in df.columns:
    df['amount_noisy'] = df['tx_amount_xof'] + np.random.normal(0, 0.01 * df['tx_amount_xof'].std(), len(df))

# La colonne cible dans Dataset30 est 'default_fraud_flag' (0/1)
print("Valeurs uniques dans default_fraud_flag :", df['default_fraud_flag'].unique())
print("Distribution default_fraud_flag :", df['default_fraud_flag'].value_counts().to_dict())
print("Nombre de NaN dans default_fraud_flag :", df['default_fraud_flag'].isna().sum())

# default_fraud_flag est déjà numérique (0/1), pas besoin de conversion
df['target'] = df['default_fraud_flag'].astype(int)
print("Après encodage - Distribution target :", df['target'].value_counts().to_dict())

# Supprimer les lignes où la cible est NaN (si applicable)
df = df[df['target'].notna()]

# Exclure les colonnes non-numériques AVANT la séparation X/y (Dataset30 - Loan Default Fraud)
print("Nettoyage des colonnes avant entraînement...")
columns_to_exclude = [
    'target',                      # Target encodé
    'default_fraud_flag',          # Target original (Dataset30)
    'chargeback_fraud',            # Target original (Dataset29)
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
    'loan_transaction_id',         # ID de transaction (Dataset30)
    'borrower_id',                 # ID emprunteur (Dataset30)
    'pos_transaction_id',          # ID de transaction (Dataset29)
    'merchant_id',                 # ID marchand (Dataset29)
    'terminal_id',                 # ID terminal (Dataset29)
    'claim_number',                # ID de réclamation (Dataset28)
    'policy_holder_id',            # ID titulaire police (Dataset28)
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
    'tx_timestamp',                # Timestamp original (Dataset20, 25, 26 - on garde les features dérivées)
    'date_transaction',            # Date originale (Dataset19, 24 - on garde les features dérivées)
    'heure_transaction',           # Time original (Dataset19, 24 - on garde les features dérivées)
    'datetime_tx',                 # Datetime original (autres datasets)
    'timestamp'                    # Datetime parsé (on garde les features dérivées)
]

# Garder toutes les colonnes SAUF celles explicitement exclues
feature_columns = []
for col in df.columns:
    if col not in columns_to_exclude:
        # Convertir en numérique si ce n'est pas déjà fait
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)  # Remplacer NaN par 0
                print(f"Conversion de {col} en numérique")
            except:
                print(f"Exclusion de {col} (non convertible)")
                continue
        
        feature_columns.append(col)
        print(f"Gardé: {col} (type: {df[col].dtype})")

print(f"Features retenues: {len(feature_columns)}")
print(f"Features importantes Dataset10: {[col for col in feature_columns if col in ['payment_method', 'transaction_purpose', 'customer_type', 'zone', 'merchant_category', 'device_type']]}")

# Séparer features et label
X = df[feature_columns]
y = df['target']

print(f"Forme finale de X: {X.shape}")
print(f"Types de données dans X:\n{X.dtypes.value_counts()}")



# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Gestion du déséquilibre des classes
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

# 🔄 RECHERCHE PAR ÉTAPES POUR FRAUD DETECTION
# ÉTAPE 1: Grille grossière pour identifier la meilleure zone
# ÉTAPE 2: Fine-tuning précis autour des meilleurs résultats

# 📊 ADAPTATION SPÉCIFIQUE Dataset25 (15K lignes + Fraud 10.24% - CRYPTOCURRENCY TRANSACTION MONITORING)
print(f"\n🎯 ANALYSE {DATASET_NAME} POUR OPTIMISATION GRILLE:")
print(f"   • Features: ~{len(X_train.columns)} (features engineered + Mortgage/Payment specific)")
print(f"   • Fraud rate: {y_train.mean()*100:.2f}% ({y_train.sum()} cas) → TRÈS FAIBLE (imbalance 70.3:1)")
print(f"   • Scale pos weight optimal: {scale_pos_weight:.1f}:1")
print(f"   • Type: Taille moyenne (42K lignes - MORTGAGE/SUBSCRIPTION PAYMENT FRAUD)")
print(f"   • Stratégie: arbres profonds, learning lent, scale_pos_weight ~{int(scale_pos_weight)}")

# 📊 ÉTAPE 1: GRILLE OPTIMISÉE - Dataset30 (42K lignes, 3.38% fraud)
param_grid_stage1 = {
    # Exploration OPTIMISÉE Dataset30 - Taille moyenne avec Fraud MODÉRÉ
    'max_depth': [6, 8],                      # OPTIMISÉ: 42K lignes + fraud 3.38% → arbres moyennement profonds
    'learning_rate': [0.05, 0.08, 0.11],     # OPTIMISÉ: Apprentissage modéré pour classe modérée
    'subsample': [0.75, 0.85],               # OPTIMISÉ: Subsampling modéré-élevé pour 42K
    'min_child_weight': [3, 5],              # OPTIMISÉ: 1,419 frauds (modéré) → feuilles moyennes
    'gamma': [0.1, 0.2],                     # OPTIMISÉ: 3.38% fraud (modéré) → pruning modéré-élevé
    
    # Paramètres secondaires - OPTIMISÉS Dataset30
    'colsample_bytree': [0.70, 0.80],        # OPTIMISÉ: 21 colonnes source → 70-80% des features
    'reg_alpha': [0.0, 0.1],                 # OPTIMISÉ: L1 faible-modérée (classe modérée)
    'reg_lambda': [0.8, 1.5],                # OPTIMISÉ: L2 modérée-élevée pour 42K lignes
    'n_estimators': [400],                   # ADAPTÉ: Arbres nombreux (taille moyenne + classe modérée)
    
    # Scale pos weight - OPTIMAL pour imbalance ~28.6:1 (MODÉRÉ)
    'scale_pos_weight': [
        scale_pos_weight * 0.85,             # Sous-pénalisation
        scale_pos_weight,                    # Optimal calculé (~29)
        scale_pos_weight * 1.15              # Sur-pénalisation
    ]
}

# Total Étape 1: 2×3×2×2×2×2×2×2×1×3 = 1,152 combinaisons sur 35% données
param_grid = param_grid_stage1

# STRATÉGIE OPTIMISÉE 2-ÉTAPES POUR Dataset30:
# Stage 1: 1,152 combinaisons sur 35% données → Exploration équilibrée (~20-30 min)
# Stage 2: ~2,000-3,000 combinaisons sur 100% données → Fine-tuning précis (~80-120 min)
total_stage1 = 2*3*2*2*2*2*2*2*1*3
print(f"\n🔄 RECHERCHE PAR ÉTAPES - {DATASET_NAME} (TAILLE MOYENNE - 42K LIGNES):")
print(f"   Stage 1 (exploration): {total_stage1} combinaisons sur 35% données (~20-30 min)")
print(f"   Stage 2 (fine-tuning): ~2,000-3,000 combinaisons sur 100% données (~80-120 min)")
print(f"   Temps total estimé: 100-150 minutes (1.7-2.5 heures)")
print(f"   Performance attendue: 98-99% de l'optimal")
print(f"💡 Grille adaptée aux caractéristiques {DATASET_NAME} (Loan Default Fraud + classe MODÉRÉE)")
print("⚡ Stratégie optimisée: exploration rapide puis fine-tuning complet")
print()
print(f"🔍 OPTIMISATIONS SPÉCIFIQUES {DATASET_NAME} (42K LIGNES, {y_train.mean()*100:.2f}% FRAUD MODÉRÉ):")
print(f"   • max_depth [6,8] → Taille 42K + fraud 3.38% → arbres MOYENS-PROFONDS")
print(f"   • learning_rate [0.05-0.11] → Convergence modérée pour classe modérée")
print(f"   • subsample [0.75-0.85] → Sous-échantillonnage MODÉRÉ-ÉLEVÉ (42K lignes)")
print(f"   • min_child_weight [3,5] → {y_train.sum()} frauds (modéré) → feuilles moyennes")
print(f"   • gamma [0.1-0.2] → Pruning modéré-élevé pour classe modérée")
print(f"   • colsample_bytree [0.70-0.80] → Garder 70-80% des ~{len(X_train.columns)} features")
print(f"   • reg_alpha [0.0-0.1] → L1 faible-modérée (classe modérée)")
print(f"   • reg_lambda [0.8-1.5] → L2 modérée-élevée pour généralisation")
print(f"   • scale_pos_weight 3 VALEURS → Imbalance {scale_pos_weight:.2f}:1 (MODÉRÉ)")
print(f"   • n_estimators [400] → Arbres nombreux (taille moyenne + classe modérée)")


# Initialiser le modèle XGBoost directement
model = xgb.XGBClassifier(random_state=42)

# Validation croisée stratifiée avec plus de robustesse
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV avec multiple scoring pour plus de diversité
print(f"Démarrage GridSearchCV avec {len(list(ParameterGrid(param_grid)))} combinaisons...")
print(f"Utilisation de {cv.n_splits} folds de validation croisée")

# Utiliser différentes métriques pour avoir plus de variété dans le classement
scoring_metrics = ['f1', 'precision', 'recall', 'roc_auc']

# 🔄 RECHERCHE PAR ÉTAPES IMPLÉMENTATION
n_jobs = os.cpu_count() or 1
print(f"🚀 Parallélisation sur {n_jobs} CPU")

print(f"\n🎯 === ÉTAPE 1/2: EXPLORATION LARGE (35% données) ===")
print(f"Grille grossière: {len(list(ParameterGrid(param_grid)))} combinaisons")
print(f"Objectif: Identifier les meilleures zones de paramètres RAPIDEMENT")

# STRATÉGIE OPTIMISÉE: Sous-échantillonner pour grille 1 (exploration rapide)
# Grille 1 (35% données) → Trouve la bonne zone rapidement
# Grille 2 (100% données) → Fine-tuning précis avec toutes les données
sample_fraction_stage1 = 0.35
n_samples_stage1 = int(len(X_train) * sample_fraction_stage1)

# Échantillonnage stratifié pour préserver la distribution de fraude
from sklearn.model_selection import train_test_split
X_train_stage1, _, y_train_stage1, _ = train_test_split(
    X_train, y_train, 
    train_size=sample_fraction_stage1, 
    random_state=42, 
    stratify=y_train
)

print(f"📊 Données Stage 1: {len(X_train_stage1):,} lignes ({sample_fraction_stage1*100:.0f}% de {len(X_train):,})")
print(f"   Fraud rate Stage 1: {y_train_stage1.mean()*100:.2f}% (identique au train complet)")

# ÉTAPE 1: Grille grossière sur sous-échantillon
start_stage1 = time.time()
grid_stage1 = GridSearchCV(
    model, param_grid,
    scoring='f1',  # Une seule métrique pour stage 1
    refit=True,
    cv=cv,
    verbose=2,
    n_jobs=n_jobs,
    return_train_score=False
)

grid_stage1.fit(X_train_stage1, y_train_stage1)
stage1_time = time.time() - start_stage1

print(f"\n✅ ÉTAPE 1 terminée en {stage1_time:.1f}s")
print(f"Meilleur score Stage 1: {grid_stage1.best_score_:.4f}")
print(f"Meilleurs params Stage 1: {grid_stage1.best_params_}")

# Analyser les résultats Stage 1 pour définir Stage 2
results_stage1 = pd.DataFrame(grid_stage1.cv_results_)
# ÉTAPE 1 utilise scoring='f1' → colonne 'mean_test_score'
top_configs = results_stage1.nlargest(5, 'mean_test_score')

print(f"\n🔍 Top 3 configurations Stage 1:")
for i, (idx, row) in enumerate(top_configs.head(3).iterrows()):
    print(f"  {i+1}. F1: {row['mean_test_score']:.4f} | {row['params']}")

# 📊 ÉTAPE 2: FINE-TUNING autour des meilleures configurations - ADAPTÉ Dataset20
print(f"\n🎯 === ÉTAPE 2/2: FINE-TUNING (Dataset20 OPTIMISÉ) ===")
print(f"Fine-tuning autour des meilleurs résultats Stage 1")
print(f"Stratégie: Raffinement précis avec contraintes Dataset20 (BANK CARD FRAUD)")

# Extraire les meilleures valeurs de Stage 1
best_params = grid_stage1.best_params_
best_depth = best_params['max_depth']
best_lr = best_params['learning_rate']
best_subsample = best_params['subsample']
best_min_child = best_params['min_child_weight']
best_gamma = best_params['gamma']

# Construire grille Stage 2 ADAPTÉE Dataset25 (15K lignes - taille MOYENNE, fraud MODÉRÉ-ÉLEVÉ)
param_grid_stage2 = {
    # Fine-tuning ADAPTÉ - Dataset25 avec 15K lignes (taille MOYENNE, fraud MODÉRÉ-ÉLEVÉ 10.24%)
    'max_depth': [
        max(3, best_depth-1),                # Min 3 (fraud modéré-élevé → profondeur moyenne)
        best_depth, 
        min(7, best_depth+1)                 # Max 7 (dataset 15K + fraud modéré-élevé)
    ],
    'learning_rate': [
        max(0.07, best_lr-0.03),             # Min 0.07 (convergence rapide)
        best_lr, 
        min(0.22, best_lr+0.03)              # Max 0.22 (rapide pour classe modérée-élevée)
    ],
    'subsample': [
        max(0.70, best_subsample-0.1),       # Min 0.70 (classe modérée-élevée)
        best_subsample, 
        min(0.90, best_subsample+0.05)       # Max 0.90 (adapté 15K lignes)
    ],
    'min_child_weight': [
        max(1, best_min_child-1),            # Min 1 (feuilles petites pour 1,536 frauds)
        best_min_child, 
        min(3, best_min_child+1)             # Max 3 (1,536 frauds = MODÉRÉ)
    ],
    'gamma': [
        max(0.0, best_gamma-0.05),           # Min 0.0 (pruning minimal)
        best_gamma, 
        min(0.15, best_gamma+0.08)           # Max 0.15 (modéré - fraud MODÉRÉ-ÉLEVÉ)
    ],
    
    # Exploration paramètres secondaires - OPTIMISÉS Dataset25 (15K, 10.24% fraud, ~40-45 features)
    'colsample_bytree': [0.75, 0.85],        # OPTIMISÉ: 75-85% des ~45 features (24 colonnes source)
    'reg_alpha': [0.0, 0.05],                # OPTIMISÉ: L1 léger (classe modérée-élevée)
    'reg_lambda': [0.5, 0.8],                # OPTIMISÉ: L2 standard pour 15K lignes
    'n_estimators': [350],                   # RÉDUIT: Moins d'arbres (taille moyenne + classe modérée-élevée)
    
    # Scale pos weight - FINE-TUNING pour imbalance ~8.8:1 (Dataset25 - MODÉRÉMENT DÉSÉQUILIBRÉ)
    'scale_pos_weight': [
        scale_pos_weight * 0.85,             # Sous-pénalisation (~7.5)
        scale_pos_weight,                    # Optimal (~8.8)
        scale_pos_weight * 1.15              # Sur-pénalisation (~10.1)
    ]
}

stage2_combinations = len(list(ParameterGrid(param_grid_stage2)))
print(f"Grille fine Stage 2 (Dataset25 - 15K lignes avec fraud MODÉRÉ-ÉLEVÉ): {stage2_combinations} combinaisons")
print(f"   • max_depth: 3-7 → Profondeur MOYENNE pour 15K lignes + 10.24% fraud MODÉRÉ-ÉLEVÉ")
print(f"   • colsample_bytree: 0.75-0.85 (2 valeurs) → Adapté aux ~45 features")
print(f"   • gamma: 0.0-0.15 (modéré) → Pruning MODÉRÉ pour classe modérée-élevée")
print(f"   • reg_alpha/lambda: 2 valeurs chacun (standard pour classe modérée-élevée)")
print(f"   • n_estimators: 350 (réduit pour taille moyenne + classe modérée-élevée)")
print(f"   • 3×3×3×3×3×2×2×2×1×3 = ~2,916 combinaisons (~120-180 min pour 32K lignes)")


print(f"\n🎯 === ÉTAPE 2/2: FINE-TUNING PRÉCIS (100% données) ===")
print(f"Grille raffinée autour des meilleurs params Stage 1")
print(f"Objectif: Optimisation MAXIMALE avec toutes les données")
print(f"📊 Données Stage 2: {len(X_train):,} lignes (100% du train)")

# ÉTAPE 2: Fine-tuning sur TOUTES les données d'entraînement
start_stage2 = time.time()
grid_stage2 = GridSearchCV(
    model, param_grid_stage2,
    scoring=scoring_metrics,  # Toutes les métriques pour Stage 2
    refit='f1',
    cv=cv,
    verbose=2,
    n_jobs=n_jobs,
    return_train_score=False
)

grid_stage2.fit(X_train, y_train)  # 100% des données train
stage2_time = time.time() - start_stage2

# Le meilleur modèle final est celui du Stage 2
grid = grid_stage2
training_time = stage1_time + stage2_time

print(f"\n✅ === RECHERCHE PAR ÉTAPES TERMINÉE ===")
print(f"Temps Stage 1: {stage1_time:.1f}s ({stage1_time/60:.1f} min)")
print(f"Temps Stage 2: {stage2_time:.1f}s ({stage2_time/60:.1f} min)")
print(f"Temps TOTAL: {training_time:.1f}s ({training_time/60:.1f} minutes)")
print(f"")
print(f"🏆 MEILLEUR MODÈLE FINAL (Stage 2):")
print(f"   Score F1: {grid.best_score_:.4f}")
print(f"   Paramètres: {grid.best_params_}")
print(f"")
print(f"📊 AMÉLIORATION Stage 1 → Stage 2:")
print(f"   Stage 1: {grid_stage1.best_score_:.4f}")
print(f"   Stage 2: {grid.best_score_:.4f}")
print(f"   Gain: {((grid.best_score_ / grid_stage1.best_score_) - 1) * 100:+.2f}%")

# Extraire et trier les résultats pour obtenir le top-5
results_df = pd.DataFrame(grid.cv_results_)

# Afficher les top-5 pour différentes métriques
print("=== ANALYSE DES TOP-5 CONFIGURATIONS ===")

for metric in ['f1', 'precision', 'recall', 'roc_auc']:
    col_name = f'mean_test_{metric}'
    if col_name in results_df.columns:
        top_5_metric = results_df.nlargest(5, col_name)
        print(f"\n🏆 Top 5 pour {metric.upper()} :")
        for i, (idx, row) in enumerate(top_5_metric.iterrows()):
            print(f"  {i+1}. {metric}: {row[col_name]:.4f} | Params: {row['params']}")

print(f"\n🎯 MEILLEURE CONFIGURATION :")
print(f"Paramètres : {grid.best_params_}")
print(f"Score F1 CV : {grid.best_score_:.4f}")

# Afficher un aperçu du top-5 (juste pour info)
top_5_preview = results_df.nlargest(5, 'mean_test_f1')
print(f"\n📋 Aperçu Top-5 (sauvé dans {DATASET_NAME}_grid_search_results.json) :")
for i, (_, row) in enumerate(top_5_preview.iterrows(), 1):
    print(f"  {i}. F1: {row['mean_test_f1']:.4f} | Params: {dict(list(row['params'].items())[:3])}...")

# Vérifier le surapprentissage : comparer train vs test
best_model = grid.best_estimator_

# Afficher les infos d'early stopping
if hasattr(best_model, 'model_') and hasattr(best_model.model_, 'best_iteration'):
    best_iteration = best_model.model_.best_iteration
    total_estimators = best_model.n_estimators
    print(f"🛑 Early Stopping : {best_iteration}/{total_estimators} estimators utilisés")
    print(f"   Économie : {total_estimators - best_iteration} estimators évités")
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

from sklearn.metrics import f1_score
train_f1 = f1_score(y_train, y_pred_train)
test_f1 = f1_score(y_test, y_pred_test)

print(f"\n📊 DÉTECTION SURAPPRENTISSAGE :")
print(f"Score F1 Train : {train_f1:.4f}")
print(f"Score F1 Test  : {test_f1:.4f}")
print(f"Différence     : {abs(train_f1 - test_f1):.4f}")

if abs(train_f1 - test_f1) > 0.05:
    print("⚠️  SURAPPRENTISSAGE DÉTECTÉ (différence > 0.05)")
else:
    print("✅ Pas de surapprentissage détecté")

# Prédictions avec le meilleur modèle
y_pred = y_pred_test

# Évaluation
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Sauvegarder le meilleur modèle XGBoost et TOUS les résultats - Nouvelle organisation
import joblib
joblib.dump(grid.best_estimator_, f'data/models/{DATASET_NAME}_xgb_model.joblib')

# Sauvegarder TOUS les résultats pour diverse_top5_selector.py
results_df.to_json(f'data/results/{DATASET_NAME}_grid_search_results.json', orient='records', indent=2)

print(f'✅ Modèle XGBoost sauvegardé dans data/models/{DATASET_NAME}_xgb_model.joblib')
print(f'✅ Tous les résultats GridSearch sauvegardés dans data/results/{DATASET_NAME}_grid_search_results.json')
print('🎯 Prochaine étape: python diverse_top5_selector.py')
