"""
XGBoost Config Predictor
Utilise le Meta-Transformer entraîné pour prédire automatiquement
les hyperparamètres XGBoost optimaux pour un nouveau dataset

Usage: python predict_xgboost_config.py Dataset8.csv
"""

import pandas as pd
import numpy as np
import torch
import pickle
import json
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

class XGBoostConfigPredictor:
    """Classe pour prédire les configs XGBoost avec le Meta-Transformer"""
    
    def __init__(self, 
                 model_path='data/models/metatransformer_model.pth',
                 processor_path='data/models/metatransformer_processor.pkl'):
        """
        Initialiser le prédicteur
        
        Args:
            model_path: Chemin vers le modèle PyTorch entraîné
            processor_path: Chemin vers les preprocessors
        """
        self.model_path = Path(model_path)
        self.processor_path = Path(processor_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Vérifier que les fichiers existent
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
        if not self.processor_path.exists():
            raise FileNotFoundError(f"Preprocessors non trouvés: {self.processor_path}")
        
        self.model = None
        self.processors = None
        self.feature_names = None
        self.target_names = None
        
    def load_model(self):
        """Charger le modèle et les preprocessors"""
        print("📂 Chargement du Meta-Transformer...")
        
        # Charger le modèle
        from train_metatransformer import MetaTransformer

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_config = checkpoint.get('model_config', {})

        # Récupérer les noms des cibles pour déduire output_dim
        self.target_names = checkpoint.get('target_names', [])
        output_dim = len(self.target_names)

        # Déduire num_layers à partir du state_dict (compatibilité rétroactive)
        state_keys = list(checkpoint.get('model_state_dict', {}).keys())
        import re
        layer_idxs = set()
        for k in state_keys:
            m = re.search(r'transformer_encoder\.layers\.(\d+)\.', k)
            if m:
                layer_idxs.add(int(m.group(1)))
        if layer_idxs:
            num_layers = max(layer_idxs) + 1
        else:
            # fallback raisonnable
            num_layers = model_config.get('num_layers', 4)

        # Hidden dim fallback
        hidden_dim = model_config.get('hidden_dim', 128)

        self.model = MetaTransformer(
            input_dim=model_config.get('input_dim', hidden_dim),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim  # Déduit du nombre de target_names
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.feature_names = checkpoint['feature_names']
        # target_names déjà récupéré plus haut
        
        print(f"   ✅ Modèle chargé ({sum(p.numel() for p in self.model.parameters()):,} paramètres)")
        
        # Charger les preprocessors
        with open(self.processor_path, 'rb') as f:
            self.processors = pickle.load(f)
        
        print(f"   ✅ Preprocessors chargés")
        print(f"   ✅ Features attendues: {len(self.feature_names)}")
        print(f"   ✅ Hyperparamètres à prédire: {len(self.target_names)}")
        
    def extract_features_from_csv(self, csv_path):
        """
        Extraire les features d'un nouveau dataset CSV
        Même logique que create_unified_metatransformer_dataset.py
        
        Args:
            csv_path: Chemin vers le CSV du nouveau dataset
            
        Returns:
            dict: Features extraites
        """
        print(f"\n📊 Analyse du dataset: {csv_path}")
        
        df = pd.read_csv(csv_path)
        dataset_name = Path(csv_path).stem
        
        print(f"   Shape: {df.shape}")
        print(f"   Colonnes: {df.columns.tolist()[:5]}...")
        
        # Identifier la colonne target
        target_cols = [col for col in df.columns if any(x in col.lower() for x in ['target', 'fraud', 'label', 'class', 'suspicious'])]
        target_col = target_cols[0] if target_cols else 'unknown'
        
        if target_col != 'unknown':
            # Vérifier si c'est une colonne binaire (0/1, True/False, oui/non, etc.)
            unique_vals = df[target_col].unique()
            if len(unique_vals) <= 2:
                # Mapper les valeurs binaires
                if df[target_col].dtype == 'object':
                    # Valeurs textuelles (oui/non, yes/no, etc.)
                    positive_values = ['oui', 'yes', 'true', '1', 'fraud', 'positive']
                    df[target_col + '_binary'] = df[target_col].astype(str).str.lower().isin(positive_values).astype(int)
                    fraud_rate = df[target_col + '_binary'].mean()
                else:
                    # Valeurs numériques
                    fraud_rate = df[target_col].mean()
            else:
                fraud_rate = 0.05
            print(f"   Target: {target_col} (fraud_rate: {fraud_rate:.4f})")
        else:
            fraud_rate = 0.05
            print(f"   ⚠️  Target non détecté, fraud_rate par défaut: {fraud_rate}")
        
        # Types de colonnes
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Exclure la target
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        actual_numeric_count = len(numeric_cols)
        actual_categorical_count = len(categorical_cols)
        
        # Missing values
        missing_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        
        # Statistiques numériques
        if numeric_cols:
            numeric_df = df[numeric_cols]
            numeric_mean_std = numeric_df.std().mean()
            numeric_mean_skewness = numeric_df.skew().mean()
            
            # Corrélations
            corr_matrix = numeric_df.corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)
            numeric_correlation_strength = corr_matrix.values.mean()
        else:
            numeric_mean_std = 0.0
            numeric_mean_skewness = 0.0
            numeric_correlation_strength = 0.0
        
        # Feature importance simulée (on ne l'a pas pour un nouveau dataset)
        # On utilise des valeurs moyennes basées sur l'entraînement
        total_features_engineered = df.shape[1]
        top_feature_importance = 0.25  # Valeur typique
        avg_feature_importance = 0.05  # Valeur typique
        feature_importance_entropy = 2.5  # Valeur typique
        top5_feature_concentration = 0.65  # Valeur typique
        
        # Complexity score (estimation)
        complexity_score = (df.shape[0] * df.shape[1]) / 100000
        structure_target_distribution = fraud_rate if target_col != 'unknown' else 0.05
        
        # Construire le dictionnaire de features
        features = {
            'dataset_rows': df.shape[0],
            'dataset_columns': df.shape[1],
            'fraud_rate': fraud_rate,
            'actual_numeric_count': actual_numeric_count,
            'actual_categorical_count': actual_categorical_count,
            'actual_missing_rate': missing_rate,
            'numeric_mean_std': numeric_mean_std,
            'numeric_mean_skewness': numeric_mean_skewness,
            'numeric_correlation_strength': numeric_correlation_strength,
            'total_features_engineered': total_features_engineered,
            'top_feature_importance': top_feature_importance,
            'avg_feature_importance': avg_feature_importance,
            'feature_importance_entropy': feature_importance_entropy,
            'top5_feature_concentration': top5_feature_concentration,
            'complexity_score': complexity_score,
            'structure_target_distribution': structure_target_distribution,
        }
        
        print(f"\n✅ Features extraites:")
        print(f"   Rows: {features['dataset_rows']:,}")
        print(f"   Columns: {features['dataset_columns']}")
        print(f"   Fraud rate: {features['fraud_rate']:.4f}")
        print(f"   Numeric features: {features['actual_numeric_count']}")
        print(f"   Categorical features: {features['actual_categorical_count']}")
        print(f"   Missing rate: {features['actual_missing_rate']:.4f}")
        
        return features, dataset_name, target_col
    
    def prepare_input(self, features, dataset_name):
        """
        Préparer les features pour l'inférence (normalisation, encoding)
        
        Args:
            features: dict des features extraites
            dataset_name: nom du dataset
            
        Returns:
            torch.Tensor: Features normalisées prêtes pour le modèle
        """
        # Option 1 : Ignorer l'encoding du dataset_name (meilleure généralisation)
        # Le dataset_name est un identifiant arbitraire sans sens sémantique
        features['dataset_encoded'] = 0.0  # Valeur neutre
        
        # Option 2 : Encoder uniquement si dataset connu (commenté)
        # try:
        #     dataset_encoded = self.processors['dataset_encoder'].transform([dataset_name])[0]
        #     features['dataset_encoded'] = dataset_encoded
        # except:
        #     features['dataset_encoded'] = 0.0  # Valeur neutre pour datasets inconnus
        #     print(f"   ⚠️  Dataset inconnu '{dataset_name}', utilisation valeur neutre (0.0)")
        
        # Encoder target_column (on ne l'a pas, utiliser valeur par défaut)
        features['target_column_encoded'] = 0
        features['target_score_category_encoded'] = 0
        
        # Créer le vecteur de features dans le bon ordre
        feature_vector = []
        for fname in self.feature_names:
            if fname in features:
                feature_vector.append(features[fname])
            else:
                feature_vector.append(0.0)  # Valeur par défaut
        
        # Convertir en numpy array
        X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        
        # Normaliser avec le même scaler
        X_scaled = self.processors['feature_scaler'].transform(X)
        
        # Convertir en tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        return X_tensor
    
    def predict(self, X_tensor):
        """
        Faire la prédiction
        
        Args:
            X_tensor: Features normalisées
            
        Returns:
            dict: Hyperparamètres prédits
        """
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor)
            y_pred_scaled = y_pred_scaled.cpu().numpy()
        
        # Dénormaliser
        y_pred = self.processors['target_scaler'].inverse_transform(y_pred_scaled)[0]
        
        # Créer le dictionnaire de config
        config = {}
        for i, name in enumerate(self.target_names):
            value = float(y_pred[i])
            
            # Arrondir les valeurs entières
            if name in ['max_depth', 'min_child_weight', 'n_estimators']:
                value = int(round(value))
            
            # Assurer les contraintes
            if name == 'max_depth':
                value = max(3, min(10, value))
            elif name == 'learning_rate':
                value = max(0.001, min(0.3, value))
            elif name in ['subsample', 'colsample_bytree']:
                value = max(0.5, min(1.0, value))
            elif name == 'gamma':
                value = max(0.0, min(5.0, value))
            elif name == 'min_child_weight':
                value = max(1, min(20, value))
            elif name == 'n_estimators':
                value = max(50, min(1000, value))
            elif name == 'scale_pos_weight':
                value = max(1.0, min(150.0, value))
            elif name in ['reg_alpha', 'reg_lambda']:
                value = max(0.0, min(10.0, value))
            
            config[name] = value
        
        return config
    
    def generate_config_file(self, config, dataset_name, target_col, features, output_dir='data/predicted_configs'):
        """
        Sauvegarder la configuration en JSON
        
        Args:
            config: dict des hyperparamètres prédits
            dataset_name: nom du dataset
            target_col: nom de la colonne target
            features: features extraites
            output_dir: répertoire de sortie
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{dataset_name}_predicted_config.json"
        
        result = {
            'dataset_name': dataset_name,
            'prediction_date': datetime.now().isoformat(),
            'model_version': 'MetaTransformer_v1.0',
            'target_column': target_col,
            'dataset_characteristics': {
                'rows': features['dataset_rows'],
                'columns': features['dataset_columns'],
                'fraud_rate': features['fraud_rate'],
                'numeric_features': features['actual_numeric_count'],
                'categorical_features': features['actual_categorical_count'],
                'missing_rate': features['actual_missing_rate']
            },
            'predicted_xgboost_config': {
                'hyperparameters': config,
                'confidence': 'medium',  # À améliorer avec calibration
                'recommendation': 'Test this config first, then try variations if needed'
            },
            'usage_instructions': {
                'step_1': 'Load your dataset and prepare X, y',
                'step_2': f'Use these hyperparameters with XGBoost',
                'step_3': 'Train with cross-validation to verify performance',
                'step_4': 'If unsatisfied, adjust learning_rate or max_depth manually',
                'python_example': f"""
import xgboost as xgb
from sklearn.model_selection import cross_val_score

model = xgb.XGBClassifier(
    max_depth={config['max_depth']},
    learning_rate={config['learning_rate']:.4f},
    n_estimators={config['n_estimators']},
    subsample={config['subsample']:.3f},
    colsample_bytree={config['colsample_bytree']:.3f},
    gamma={config['gamma']:.3f},
    min_child_weight={config['min_child_weight']},
    scale_pos_weight={config['scale_pos_weight']:.2f},
    reg_alpha={config['reg_alpha']:.3f},
    reg_lambda={config['reg_lambda']:.3f},
    random_state=42
)

scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f'F1 Score: {{scores.mean():.4f}} ± {{scores.std():.4f}}')
"""
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Configuration sauvegardée: {output_file}")
        
        return output_file
    
    def run_prediction_pipeline(self, csv_path):
        """
        Pipeline complet de prédiction
        
        Args:
            csv_path: Chemin vers le CSV du nouveau dataset
            
        Returns:
            dict: Configuration prédite
        """
        print("\n" + "="*70)
        print("🎯 XGBOOST CONFIG PREDICTION")
        print("="*70)
        
        # 1. Charger le modèle
        if self.model is None:
            self.load_model()
        
        # 2. Extraire les features
        features, dataset_name, target_col = self.extract_features_from_csv(csv_path)
        
        # 3. Préparer l'input
        print(f"\n🔄 Préparation des features pour l'inférence...")
        X_tensor = self.prepare_input(features, dataset_name)
        
        # 4. Prédire
        print(f"\n🧠 Prédiction des hyperparamètres...")
        config = self.predict(X_tensor)
        
        # 5. Afficher les résultats
        print(f"\n✅ CONFIGURATION XGBOOST PRÉDITE:")
        print(f"   {'Hyperparamètre':<20s} | Valeur")
        print(f"   {'-'*20} | {'-'*10}")
        for name, value in config.items():
            if isinstance(value, int):
                print(f"   {name:<20s} | {value}")
            else:
                print(f"   {name:<20s} | {value:.4f}")
        
        # 6. Sauvegarder
        output_file = self.generate_config_file(config, dataset_name, target_col, features)
        
        print("\n" + "="*70)
        print("✅ PRÉDICTION TERMINÉE AVEC SUCCÈS!")
        print("="*70)
        
        return config


def main():
    """Fonction principale"""
    
    if len(sys.argv) != 2:
        print("Usage: python predict_xgboost_config.py <dataset.csv>")
        print("\nExemple:")
        print("  python predict_xgboost_config.py data/datasets/Dataset8.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"❌ Erreur: Fichier non trouvé: {csv_path}")
        sys.exit(1)
    
    # Créer le prédicteur
    predictor = XGBoostConfigPredictor()
    
    # Lancer la prédiction
    config = predictor.run_prediction_pipeline(csv_path)
    
    print(f"\n💡 PROCHAINES ÉTAPES:")
    print(f"   1. Charger la config JSON générée")
    print(f"   2. Entraîner XGBoost avec ces hyperparamètres")
    print(f"   3. Valider avec cross-validation")
    print(f"   4. Déployer en production si satisfait !")


if __name__ == "__main__":
    main()
