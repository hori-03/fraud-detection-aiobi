"""
🤖 AUTO FEATURE ENGINEERING
Utilise la structure du dataset pour générer automatiquement les features optimales
Version améliorée avec matching sémantique des colonnes
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Import du column matcher pour détection sémantique
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.column_matcher import ColumnMatcher


class AutoFeatureEngineer:
    """
    Génère automatiquement les features basées sur la structure du dataset
    Utilise les patterns appris de Dataset1-7
    """
    
    def __init__(self, structure_path=None):
        """
        Args:
            structure_path: Chemin vers le fichier structure JSON (optionnel)
        """
        self.structure = None
        if structure_path:
            with open(structure_path, 'r') as f:
                self.structure = json.load(f)
        
        self.label_encoders = {}
        self.feature_names_generated = []
        self.column_matcher = ColumnMatcher(fuzzy_threshold=0.7)
    
    def detect_column_types(self, df):
        """
        Détecte automatiquement les types de colonnes avec matching sémantique.
        Ne dépend plus des noms exacts mais utilise les groupes sémantiques.
        """
        
        column_types = {
            'id_columns': [],
            'amount_columns': [],
            'time_columns': [],
            'date_columns': [],
            'categorical_columns': [],
            'numeric_columns': [],
            'name_columns': [],
            'country_columns': [],
            'bank_columns': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            n_unique = df[col].nunique()
            
            # Utiliser le matching sémantique pour détecter le type
            semantic_group = self.column_matcher.get_semantic_group(col)
            
            # Mapping sémantique → type de colonne
            if semantic_group == 'transaction_id' or 'id' in semantic_group:
                column_types['id_columns'].append(col)
            
            elif semantic_group == 'amount':
                column_types['amount_columns'].append(col)
            
            elif semantic_group == 'timestamp':
                if 'date' in col_lower:
                    column_types['date_columns'].append(col)
                else:
                    column_types['time_columns'].append(col)
            
            elif semantic_group == 'country' or semantic_group == 'location':
                column_types['country_columns'].append(col)
            
            elif semantic_group == 'account':
                column_types['bank_columns'].append(col)
            
            # Les groupes sémantiques suivants sont généralement catégoriels
            elif semantic_group in ['card', 'merchant', 'status', 'channel', 'currency']:
                column_types['categorical_columns'].append(col)
            
            # Fallback sur l'analyse du nom si pas de groupe sémantique
            elif semantic_group == 'unknown':
                # Détection par mots-clés (ancien système en fallback)
                unique_ratio = n_unique / len(df)
                
                # Amount/Money columns (VÉRIFIER EN PREMIER avant ID!)
                if any(kw in col_lower for kw in ['amount', 'montant', 'price', 'prix', 'value', 'balance']):
                    column_types['amount_columns'].append(col)
                
                # ID columns (très haute cardinalité, unique)
                elif unique_ratio > 0.95 or 'id' in col_lower:
                    column_types['id_columns'].append(col)
                
                # Time columns
                elif any(kw in col_lower for kw in ['time', 'heure', 'hour']):
                    column_types['time_columns'].append(col)
                
                # Date columns
                elif any(kw in col_lower for kw in ['date', 'timestamp', 'datetime']):
                    column_types['date_columns'].append(col)
                
                # Name columns (haute cardinalité)
                elif any(kw in col_lower for kw in ['name', 'nom', 'sender', 'receiver', 'payee', 'payer']):
                    column_types['name_columns'].append(col)
                
                # Country columns
                elif any(kw in col_lower for kw in ['country', 'pays', 'nation']):
                    column_types['country_columns'].append(col)
                
                # Bank columns
                elif any(kw in col_lower for kw in ['bank', 'banque', 'institution']):
                    column_types['bank_columns'].append(col)
                
                # Categorical (basse cardinalité)
                elif df[col].dtype == 'object' and unique_ratio < 0.1:
                    column_types['categorical_columns'].append(col)
                
                # Numeric
                elif df[col].dtype in ['int64', 'float64']:
                    column_types['numeric_columns'].append(col)
        
        return column_types
    
    def engineer_amount_features(self, df, amount_cols):
        """Crée automatiquement des features à partir des montants"""
        
        for col in amount_cols:
            if col not in df.columns:
                continue
            
            print(f"  📊 Engineering amount features from: {col}")
            
            # CRITIQUE: Vérifier que la colonne est bien numérique
            if df[col].dtype == 'object':
                # Vérifier si c'est une colonne catégorielle texte (ex: "Low", "Medium", "High")
                sample_values = df[col].dropna().head(10).tolist()
                if any(isinstance(v, str) for v in sample_values):
                    print(f"     ⚠️  Skipped: '{col}' contains text values (categorical range)")
                    continue
            
            # Essayer de convertir en numérique si ce n'est pas déjà le cas
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    print(f"     ⚠️  Skipped: '{col}' cannot be converted to numeric")
                    continue
            
            # Vérifier qu'il reste des valeurs numériques après conversion
            if df[col].isna().all():
                print(f"     ⚠️  Skipped: '{col}' has no valid numeric values")
                continue
            
            # ⚠️ Éviter les transformations sur des features déjà transformées
            col_lower = col.lower()
            is_already_transformed = any(kw in col_lower for kw in [
                'zscore', 'z_score', 'normalized', 'scaled', 
                'is_', 'flag', 'binary', 'indicator'
            ])
            
            # Si c'est déjà un z-score ou normalisé, faire moins de transformations
            if 'zscore' in col_lower or 'z_score' in col_lower:
                # Z-scores bénéficient surtout des exponentielles
                df[f'{col}_squared'] = df[col] ** 2
                self.feature_names_generated.append(f'{col}_squared')
                continue
            
            # Si c'est déjà un flag binaire (0/1), skip les transformations
            if col_lower.startswith('is_') or 'flag' in col_lower:
                unique_vals = df[col].nunique()
                if unique_vals <= 5:  # Probablement déjà un flag
                    continue
            
            # Transformations mathématiques (seulement pour montants bruts)
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
            df[f'{col}_sqrt'] = np.sqrt(df[col].clip(lower=0))
            df[f'{col}_squared'] = df[col] ** 2
            
            # Bins
            df[f'{col}_bin'] = pd.cut(df[col], bins=10, labels=False, duplicates='drop')
            
            # Flags (utiles pour détecter montants suspects)
            df[f'{col}_is_high'] = (df[col] > df[col].quantile(0.95)).astype(int)
            df[f'{col}_is_low'] = (df[col] < df[col].quantile(0.05)).astype(int)
            df[f'{col}_is_round'] = (df[col] % 1000 == 0).astype(int)
            
            self.feature_names_generated.extend([
                f'{col}_log', f'{col}_sqrt', f'{col}_squared', f'{col}_bin',
                f'{col}_is_high', f'{col}_is_low', f'{col}_is_round'
            ])
    
    def engineer_temporal_features(self, df, date_cols, time_cols):
        """Crée automatiquement des features temporelles"""
        
        # Combiner date + time si disponibles
        if date_cols and time_cols:
            date_col = date_cols[0]
            time_col = time_cols[0]
            
            if date_col in df.columns and time_col in df.columns:
                print(f"  📅 Engineering temporal features from: {date_col} + {time_col}")
                
                df['timestamp'] = pd.to_datetime(
                    df[date_col].astype(str) + ' ' + df[time_col].astype(str),
                    errors='coerce'
                )
                
                # Features temporelles
                df['transaction_hour'] = df['timestamp'].dt.hour
                df['transaction_day'] = df['timestamp'].dt.day
                df['transaction_month'] = df['timestamp'].dt.month
                df['transaction_weekday'] = df['timestamp'].dt.weekday
                df['transaction_is_weekend'] = (df['transaction_weekday'] >= 5).astype(int)
                df['is_business_hours'] = ((df['transaction_hour'] >= 8) & 
                                           (df['transaction_hour'] <= 18)).astype(int)
                df['is_night'] = ((df['transaction_hour'] >= 22) | 
                                 (df['transaction_hour'] <= 6)).astype(int)
                
                self.feature_names_generated.extend([
                    'transaction_hour', 'transaction_day', 'transaction_month',
                    'transaction_weekday', 'transaction_is_weekend',
                    'is_business_hours', 'is_night'
                ])
    
    def engineer_categorical_pairs(self, df, country_cols, bank_cols, name_cols):
        """Crée des features de comparaison entre paires"""
        
        # Comparer pays sender/receiver
        sender_country = [c for c in country_cols if 'sender' in c.lower() or 'from' in c.lower()]
        receiver_country = [c for c in country_cols if 'receiver' in c.lower() or 'to' in c.lower()]
        
        if sender_country and receiver_country:
            s_col = sender_country[0]
            r_col = receiver_country[0]
            
            if s_col in df.columns and r_col in df.columns:
                print(f"  🌍 Engineering country features: {s_col} vs {r_col}")
                df['is_same_country'] = (df[s_col] == df[r_col]).astype(int)
                df['is_international'] = (df[s_col] != df[r_col]).astype(int)
                self.feature_names_generated.extend(['is_same_country', 'is_international'])
        
        # Comparer banques sender/receiver
        sender_bank = [c for c in bank_cols if 'sender' in c.lower() or 'from' in c.lower()]
        receiver_bank = [c for c in bank_cols if 'receiver' in c.lower() or 'to' in c.lower()]
        
        if sender_bank and receiver_bank:
            s_col = sender_bank[0]
            r_col = receiver_bank[0]
            
            if s_col in df.columns and r_col in df.columns:
                print(f"  🏦 Engineering bank features: {s_col} vs {r_col}")
                df['is_same_bank'] = (df[s_col] == df[r_col]).astype(int)
                self.feature_names_generated.append('is_same_bank')
        
        # Features de noms (longueur, mots)
        for name_col in name_cols:
            if name_col in df.columns and df[name_col].dtype == 'object':
                print(f"  ✍️  Engineering name features: {name_col}")
                df[f'{name_col}_length'] = df[name_col].str.len()
                df[f'{name_col}_words'] = df[name_col].str.split().str.len()
                self.feature_names_generated.extend([
                    f'{name_col}_length', f'{name_col}_words'
                ])
    
    def engineer_interactions(self, df, amount_cols):
        """Crée des interactions entre features"""
        
        # Interactions amount × hour
        if 'transaction_hour' in df.columns and amount_cols:
            amount_col = amount_cols[0]
            if f'{amount_col}_log' in df.columns:
                print(f"  🔗 Engineering interactions: hour × amount")
                df['hour_amount_interaction'] = df['transaction_hour'] * df[f'{amount_col}_log']
                self.feature_names_generated.append('hour_amount_interaction')
    
    def encode_categoricals(self, df, categorical_cols):
        """Encode les variables catégorielles"""
        
        for col in categorical_cols:
            if col in df.columns and df[col].dtype == 'object':
                print(f"  🔢 Encoding categorical: {col}")
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
    
    def fit_transform(self, df, target_col=None):
        """
        Pipeline complet de feature engineering automatique
        
        Args:
            df: DataFrame brut
            target_col: Nom de la colonne target (à exclure du processing)
        
        Returns:
            DataFrame avec features engineerées
        """
        
        print("🤖 AUTO FEATURE ENGINEERING STARTED")
        print(f"   Input: {df.shape[0]} rows × {df.shape[1]} columns")
        
        df = df.copy()
        
        # 1. Détection automatique des types
        print("\n📋 Step 1: Detecting column types...")
        column_types = self.detect_column_types(df)
        
        for type_name, cols in column_types.items():
            if cols:
                print(f"   {type_name}: {cols}")
        
        # 2. Feature engineering par type
        print("\n🔧 Step 2: Engineering features...")
        
        # Amount features
        if column_types['amount_columns']:
            self.engineer_amount_features(df, column_types['amount_columns'])
        
        # Temporal features
        if column_types['date_columns'] or column_types['time_columns']:
            self.engineer_temporal_features(
                df,
                column_types['date_columns'],
                column_types['time_columns']
            )
        
        # Categorical pairs (same country, same bank, etc.)
        self.engineer_categorical_pairs(
            df,
            column_types['country_columns'],
            column_types['bank_columns'],
            column_types['name_columns']
        )
        
        # Interactions
        if column_types['amount_columns']:
            self.engineer_interactions(df, column_types['amount_columns'])
        
        # 3. Encodage catégorielles
        print("\n🔢 Step 3: Encoding categorical variables...")
        
        # Exclure target, IDs, dates, noms
        exclude_from_encoding = (
            column_types['id_columns'] +
            column_types['date_columns'] +
            column_types['name_columns'] +
            ([target_col] if target_col else [])
        )
        
        categorical_to_encode = [
            col for col in column_types['categorical_columns'] + 
                          column_types['country_columns'] + 
                          column_types['bank_columns']
            if col not in exclude_from_encoding
        ]
        
        self.encode_categoricals(df, categorical_to_encode)
        
        # 4. Nettoyage final
        print("\n🧹 Step 4: Final cleaning...")
        
        # Remplacer NaN/inf
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median(numeric_only=True))
        
        # Supprimer colonnes non-features
        columns_to_drop = (
            column_types['id_columns'] +
            column_types['date_columns'] +
            column_types['name_columns'] +
            (['timestamp'] if 'timestamp' in df.columns else [])
        )
        
        if target_col and target_col in df.columns:
            columns_to_drop.append(target_col)
        
        df_features = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
        
        # Convertir tout en numérique
        for col in df_features.columns:
            if df_features[col].dtype == 'object':
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        
        df_features = df_features.fillna(0)
        
        print(f"\n✅ AUTO FEATURE ENGINEERING COMPLETED")
        print(f"   Output: {df_features.shape[0]} rows × {df_features.shape[1]} columns")
        print(f"   Features generated: {len(self.feature_names_generated)}")
        
        return df_features
    
    def transform(self, df):
        """
        Applique les transformations apprises sur de nouvelles données (production)
        Utilise la même logique que fit_transform mais sans réapprendre
        
        Args:
            df: DataFrame à transformer
        
        Returns:
            DataFrame transformé avec les mêmes features
        """
        # Réutilise fit_transform sans target (pas de réapprentissage pour ce simple engineer)
        return self.fit_transform(df, target_col=None)


if __name__ == "__main__":
    # Test sur Dataset8
    print("=" * 60)
    print("TESTING AUTO FEATURE ENGINEER ON DATASET8")
    print("=" * 60)
    
    df = pd.read_csv('data/datasets/Dataset8.csv')
    
    # Auto feature engineering
    engineer = AutoFeatureEngineer()
    X = engineer.fit_transform(df, target_col='is_fraud')
    
    print(f"\n📊 RESULT:")
    print(f"   Original columns: {df.shape[1]}")
    print(f"   Final features: {X.shape[1]}")
    print(f"\n   Sample features generated:")
    for feat in engineer.feature_names_generated[:10]:
        print(f"     - {feat}")
