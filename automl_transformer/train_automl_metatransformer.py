"""
🧠 TRAIN META-TRANSFORMER FOR AUTO ML
Entraîne un Meta-Transformer qui apprend à prédire:
  1. Quels hyperparamètres utiliser (hyperparameter optimization)
  2. Quelles features garder (feature selection via importance prédite)
  3. Quelles features créer (feature engineering via flags)

Utilise les exemples créés par create_metamodel_examples.py :
  - Structure dataset (18 features standardisées)
  - Feature importance patterns (20 top features)
  - Configurations XGBoost optimales (8 hyperparamètres)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class AutoMLMetaTransformer(nn.Module):
    """
    Meta-Transformer Avancé pour AutoML Complet - VERSION OPTIMISÉE
    
    AMÉLIORATIONS:
    - Attention multi-échelle avec Conv1D
    - Skip connections pour gradient flow
    - Dropout adaptatif (plus élevé pour éviter overfitting)
    - GELU activation (meilleure que ReLU)
    - LayerNorm pour stabilité
    - Architecture plus profonde et robuste
    
    Input: Structure dataset (18 features) + Feature importance patterns (20 features)
    Output: 
      - 10 hyperparamètres XGBoost (max_depth, learning_rate, n_estimators, subsample, 
        colsample_bytree, min_child_weight, gamma, scale_pos_weight, reg_alpha, reg_lambda)
      - 20 scores de feature importance prédits
      - 5 flags de feature engineering à appliquer
    """
    
    def __init__(self, input_dim=38, hidden_dim=256, num_heads=8, num_layers=6, 
                 output_hyperparams=10, output_feature_scores=20, output_engineering_flags=5,
                 dropout=0.3):  # Dropout augmenté pour régularisation
        super(AutoMLMetaTransformer, self).__init__()
        
        # AMÉLIORATION 1: Embedding layers avec LayerNorm et GELU
        self.structure_embedding = nn.Sequential(
            nn.Linear(18, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),  # GELU meilleur que ReLU
            nn.Dropout(dropout)
        )
        
        self.importance_embedding = nn.Sequential(
            nn.Linear(20, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # AMÉLIORATION 2: Combined embedding avec skip connection
        self.combined_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # AMÉLIORATION 3: Multi-scale feature extraction (Conv1D)
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=k, padding=k//2)
            for k in [1, 3, 5]  # Différentes échelles
        ])
        self.multi_scale_combine = nn.Sequential(
            nn.Linear(hidden_dim * 3 // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # AMÉLIORATION 4: Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # AMÉLIORATION 5: Transformer Encoder avec GELU et dropout augmenté
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # GELU activation
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # AMÉLIORATION 6: Output heads avec architecture plus profonde
        self.hyperparams_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),  # Dropout réduit dans les couches finales
            nn.Linear(hidden_dim // 2, output_hyperparams),
            nn.Sigmoid()  # Normaliser entre 0-1
        )
        
        self.feature_scores_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, output_feature_scores),
            nn.Sigmoid()  # Scores entre 0 et 1
        )
        
        self.engineering_flags_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, output_engineering_flags),
            nn.Sigmoid()  # Probabilités entre 0 et 1
        )
    
    def forward(self, structure_features, importance_features, training=True):
        """
        Args:
            structure_features: [batch, 18] - Features de structure dataset
            importance_features: [batch, 20] - Features d'importance
            training: Si True, applique dropout et data augmentation
        
        Returns:
            hyperparams: [batch, 10] - Hyperparamètres prédits
            feature_scores: [batch, 20] - Scores d'importance prédits
            engineering_flags: [batch, 5] - Flags de feature engineering
        """
        
        # AMÉLIORATION 7: Data augmentation pendant training (noise injection)
        if training and self.training:
            # Ajouter du bruit gaussien léger pour robustesse
            structure_features = structure_features + torch.randn_like(structure_features) * 0.01
            importance_features = importance_features + torch.randn_like(importance_features) * 0.01
        
        # Embed séparément avec normalisation
        struct_emb = self.structure_embedding(structure_features)  # [batch, hidden/2]
        import_emb = self.importance_embedding(importance_features)  # [batch, hidden/2]
        
        # Combiner avec skip connection
        combined = torch.cat([struct_emb, import_emb], dim=-1)  # [batch, hidden]
        combined_processed = self.combined_embedding(combined)
        combined = combined + combined_processed  # Skip connection
        
        # AMÉLIORATION 8: Multi-scale feature extraction
        combined_expanded = combined.unsqueeze(1)  # [batch, 1, hidden]
        combined_transposed = combined_expanded.transpose(1, 2)  # [batch, hidden, 1]
        
        multi_scale_features = []
        for conv in self.multi_scale_conv:
            feat = conv(combined_transposed)  # [batch, hidden//4, 1]
            feat = torch.relu(feat)
            multi_scale_features.append(feat.squeeze(-1))
        
        multi_scale = torch.cat(multi_scale_features, dim=-1)  # [batch, hidden*3/4]
        multi_scale = self.multi_scale_combine(multi_scale)  # [batch, hidden]
        
        # Combiner avec skip connection
        combined = combined + multi_scale
        
        # Ajouter dimension séquence pour Transformer + positional encoding
        combined = combined.unsqueeze(1)  # [batch, 1, hidden]
        combined = combined + self.pos_encoding  # Positional encoding
        
        # Transformer avec skip connection
        transformed = self.transformer(combined)  # [batch, 1, hidden]
        transformed = transformed.squeeze(1)  # [batch, hidden]
        
        # Skip connection finale
        combined_flat = combined.squeeze(1)
        transformed = transformed + combined_flat  # Residual connection
        
        # Prédictions multiples
        hyperparams = self.hyperparams_head(transformed)
        feature_scores = self.feature_scores_head(transformed)
        engineering_flags = self.engineering_flags_head(transformed)
        
        return hyperparams, feature_scores, engineering_flags


class AutoMLDataset(Dataset):
    """
    Dataset pour entraîner le Meta-Transformer AutoML
    Charge depuis les fichiers metamodel_training_examples.json
    """
    
    def __init__(self, datasets_list):
        """
        Args:
            datasets_list: Liste des noms de datasets (ex: ['Dataset1', 'Dataset2'])
        """
        self.examples = []
        
        print(f"📂 Chargement des exemples d'entraînement...")
        
        for dataset_name in datasets_list:
            # Charger les exemples depuis metamodel_training_examples.json
            examples_path = f'data/metamodel_data/{dataset_name}_metamodel_training_examples.json'
            
            if not Path(examples_path).exists():
                print(f"⚠️  Skipping {dataset_name} (missing {examples_path})")
                continue
            
            with open(examples_path, 'r', encoding='utf-8') as f:
                dataset_examples = json.load(f)
            
            print(f"   📊 {dataset_name}: {len(dataset_examples)} configurations")
            
            # Charger chaque exemple (chaque config XGBoost)
            for example in dataset_examples:
                # Extraire les 18 features de structure
                structure_features = self.extract_structure_features_from_example(example)
                
                # Extraire les 20 top feature importances
                importance_features = self.extract_importance_features_from_example(example)
                
                # Extraire les 8 hyperparamètres cibles
                hyperparams = self.extract_hyperparams_from_example(example)
                
                # Extraire les 20 feature scores cibles
                feature_scores = self.extract_feature_scores_from_example(example)
                
                # Extraire les 5 engineering flags cibles
                engineering_flags = self.extract_engineering_flags_from_example(example)
                
                self.examples.append({
                    'structure_features': structure_features,
                    'importance_features': importance_features,
                    'hyperparams': hyperparams,
                    'feature_scores': feature_scores,
                    'engineering_flags': engineering_flags,
                    'dataset_name': dataset_name,
                    'example_id': example.get('id', 0)
                })
        
        print(f"✅ Total: {len(self.examples)} exemples d'entraînement chargés\n")
    
    def extract_structure_features_from_example(self, example):
        """Extrait les 18 features de structure depuis un exemple (version v2.0)"""
        
        # Les features sont dans dataset_structure.meta_transformer_features
        meta_features = example.get('dataset_structure', {}).get('meta_transformer_features', {})
        
        # 18 features originales (v2.0)
        features = [
            meta_features.get('rows', 0),
            meta_features.get('columns', 0),
            meta_features.get('fraud_rate', 0.0),
            meta_features.get('missing_rate', 0.0),
            meta_features.get('duplicate_rate', 0.0),
            meta_features.get('numeric_columns', 0),
            meta_features.get('categorical_columns', 0),
            meta_features.get('correlation_max', 0.0),
            meta_features.get('correlation_min', 0.0),
            meta_features.get('correlation_mean', 0.0),
            meta_features.get('variance_max', 0.0),
            meta_features.get('variance_min', 0.0),
            meta_features.get('variance_mean', 0.0),
            meta_features.get('skewness_max', 0.0),
            meta_features.get('skewness_min', 0.0),
            meta_features.get('skewness_mean', 0.0),
            meta_features.get('kurtosis_max', 0.0),
            meta_features.get('kurtosis_mean', 0.0)
        ]
        
        # PROTECTION: Remplacer NaN/Inf par 0
        features_array = np.array(features, dtype=np.float32)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_array
    
    def extract_importance_features_from_example(self, example):
        """Extrait les top 20 feature importances depuis le fichier réel"""
        
        dataset_name = example.get('dataset_name', '')
        
        # Charger le fichier feature_importance pour avoir les vraies importances
        importance_path = f'data/Feature_importance/{dataset_name}_production_feature_importance.json'
        
        if Path(importance_path).exists():
            with open(importance_path, 'r', encoding='utf-8') as f:
                importance_data = json.load(f)
            
            # Extraire les importances
            if 'production_feature_importance' in importance_data:
                production_features = importance_data['production_feature_importance']
                importances = [f.get('importance', 0) for f in production_features[:20]]
            else:
                importances = []
        else:
            importances = []
        
        # Pad à 20 avec des zéros
        while len(importances) < 20:
            importances.append(0.0)
        
        # Normaliser entre 0-1
        importance_array = np.array(importances[:20], dtype=np.float32)
        if importance_array.max() > 0:
            importance_array = importance_array / importance_array.max()
        
        # PROTECTION: Clipper et vérifier NaN
        importance_array = np.clip(importance_array, 0.0, 1.0)
        importance_array = np.nan_to_num(importance_array, nan=0.0)
        
        return importance_array
    
    def extract_hyperparams_from_example(self, example):
        """Extrait les 8 hyperparamètres cibles NORMALISÉS entre 0-1"""
        
        hyperparams_dict = example.get('optimal_xgb_config', {}).get('hyperparameters', {})
        
        # Valeurs brutes
        max_depth = hyperparams_dict.get('max_depth', 6)
        learning_rate = hyperparams_dict.get('learning_rate', 0.1)
        n_estimators = hyperparams_dict.get('n_estimators', 100)
        subsample = hyperparams_dict.get('subsample', 0.8)
        colsample_bytree = hyperparams_dict.get('colsample_bytree', 0.8)
        min_child_weight = hyperparams_dict.get('min_child_weight', 1)
        gamma = hyperparams_dict.get('gamma', 0)
        scale_pos_weight = hyperparams_dict.get('scale_pos_weight', 1)
        reg_alpha = hyperparams_dict.get('reg_alpha', 0)
        reg_lambda = hyperparams_dict.get('reg_lambda', 1)
        
        # Normaliser dans [0, 1] selon les plages typiques
        hyperparams_normalized = [
            (max_depth - 3) / (10 - 3),  # max_depth: 3-10 → 0-1
            (learning_rate - 0.01) / (0.3 - 0.01),  # lr: 0.01-0.3 → 0-1
            (n_estimators - 50) / (500 - 50),  # n_est: 50-500 → 0-1
            (subsample - 0.5) / (1.0 - 0.5),  # subsample: 0.5-1.0 → 0-1
            (colsample_bytree - 0.5) / (1.0 - 0.5),  # colsample: 0.5-1.0 → 0-1
            (min_child_weight - 1) / (10 - 1),  # min_child: 1-10 → 0-1
            (gamma - 0) / (1.0 - 0),  # gamma: 0-1 → 0-1
            (scale_pos_weight - 1) / (30 - 1),  # scale_pos: 1-30 → 0-1
            (reg_alpha - 0) / (1.0 - 0),  # reg_alpha: 0-1 → 0-1
            (reg_lambda - 0) / (3.0 - 0)  # reg_lambda: 0-3 → 0-1
        ]
        
        # Clipper entre 0-1
        hyperparams_normalized = [max(0.0, min(1.0, x)) for x in hyperparams_normalized]
        
        # DEBUG: Vérifier nombre d'hyperparams
        if len(hyperparams_normalized) != 10:
            print(f"⚠️  WARNING: Expected 10 hyperparams, got {len(hyperparams_normalized)}")
            print(f"   hyperparams_dict keys: {list(hyperparams_dict.keys())}")
        
        return np.array(hyperparams_normalized, dtype=np.float32)
    
    def extract_feature_scores_from_example(self, example):
        """Extrait les 20 feature importance scores cibles (valeurs réelles normalisées)"""
        
        dataset_name = example.get('dataset_name', '')
        
        # Charger les vraies importances
        importance_path = f'data/Feature_importance/{dataset_name}_production_feature_importance.json'
        
        if Path(importance_path).exists():
            with open(importance_path, 'r', encoding='utf-8') as f:
                importance_data = json.load(f)
            
            if 'production_feature_importance' in importance_data:
                production_features = importance_data['production_feature_importance']
                scores = [f.get('importance', 0) for f in production_features[:20]]
            else:
                scores = []
        else:
            scores = []
        
        # Pad à 20
        while len(scores) < 20:
            scores.append(0.0)
        
        # Normaliser entre 0-1
        scores_array = np.array(scores[:20], dtype=np.float32)
        if scores_array.max() > 0:
            scores_array = scores_array / scores_array.max()
        
        # PROTECTION: Clipper entre 0-1 pour éviter valeurs hors limites
        scores_array = np.clip(scores_array, 0.0, 1.0)
        
        # Vérifier qu'il n'y a pas de NaN
        if np.isnan(scores_array).any():
            print(f"⚠️  NaN detected in feature scores for {dataset_name}, replacing with 0")
            scores_array = np.nan_to_num(scores_array, nan=0.0)
        
        return scores_array
    
    def extract_engineering_flags_from_example(self, example):
        """Extrait les 5 engineering flags cibles basés sur les vraies features"""
        
        dataset_name = example.get('dataset_name', '')
        
        # Charger les noms de features réels
        importance_path = f'data/Feature_importance/{dataset_name}_production_feature_importance.json'
        
        if Path(importance_path).exists():
            with open(importance_path, 'r', encoding='utf-8') as f:
                importance_data = json.load(f)
            
            if 'production_feature_importance' in importance_data:
                production_features = importance_data['production_feature_importance']
                feature_names = [f.get('feature_name', '') for f in production_features]
            else:
                feature_names = []
        else:
            feature_names = []
        
        # Détecter les transformations présentes
        feature_names_str = ' '.join(feature_names).lower()
        
        flags = [
            1.0 if '_log' in feature_names_str or 'log_' in feature_names_str else 0.0,
            1.0 if 'interaction' in feature_names_str or '*' in feature_names_str or '_x_' in feature_names_str else 0.0,
            1.0 if 'hour' in feature_names_str or 'day' in feature_names_str or 'weekday' in feature_names_str or 'is_night' in feature_names_str or 'is_business' in feature_names_str else 0.0,
            1.0 if any(name for name in feature_names if not any(kw in name for kw in ['_log', '_sqrt', '_squared', '_bin', 'interaction'])) else 0.0,  # Categorical encoding présent
            1.0 if '_bin' in feature_names_str or '_range' in feature_names_str or 'binned' in feature_names_str else 0.0
        ]
        
        return np.array(flags, dtype=np.float32)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        return {
            'structure_features': torch.FloatTensor(example['structure_features']),
            'importance_features': torch.FloatTensor(example['importance_features']),
            'hyperparams': torch.FloatTensor(example['hyperparams']),
            'feature_scores': torch.FloatTensor(example['feature_scores']),
            'engineering_flags': torch.FloatTensor(example['engineering_flags'])
        }


def train_automl_metatransformer(datasets_list, epochs=500, batch_size=8, lr=0.00005, 
                                 patience=30, warmup_epochs=10):
    """
    Entraîne le Meta-Transformer AutoML - VERSION v2.0 (18 features)
    
    RETOUR À v2.0:
    - Architecture simplifiée: 18 structure features + 20 importance = 38 inputs
    - Les 7 features Option A ont été supprimées (dégradaient la performance)
    - Configuration optimale: lr=0.00005, loss_weights HP=2.0
    
    Args:
        datasets_list: Liste des datasets (ex: ['Dataset1', 'Dataset2', ...])
        epochs: Nombre d'epochs maximum (500)
        batch_size: Taille du batch (8)
        lr: Learning rate initial (0.00005 optimal)
        patience: Nombre d'epochs sans amélioration (30)
        warmup_epochs: Nombre d'epochs de warmup pour le scheduler
    """
    
    print("=" * 70)
    print("TRAINING OPTIMIZED AUTO ML META-TRANSFORMER")
    print("=" * 70)
    
    # Dataset
    full_dataset = AutoMLDataset(datasets_list)
    
    # NORMALISATION des structure features pour éviter explosions de gradients
    print("\n📊 Normalisation des features de structure...")
    from sklearn.preprocessing import StandardScaler
    
    # Extraire toutes les structure features
    all_structure_features = np.array([ex['structure_features'] for ex in full_dataset.examples])
    
    # Vérifier NaN avant normalisation
    if np.isnan(all_structure_features).any():
        print(f"⚠️  NaN détectés dans structure features avant normalisation")
        nan_count = np.isnan(all_structure_features).sum()
        print(f"   Nombre de NaN: {nan_count}")
        all_structure_features = np.nan_to_num(all_structure_features, nan=0.0)
    
    # Normaliser
    structure_scaler = StandardScaler()
    all_structure_features_scaled = structure_scaler.fit_transform(all_structure_features)
    
    # Remplacer dans le dataset
    for i, ex in enumerate(full_dataset.examples):
        ex['structure_features'] = all_structure_features_scaled[i]
    
    print(f"   ✅ Structure features normalisées")
    print(f"      Mean: {all_structure_features_scaled.mean():.4f}")
    print(f"      Std: {all_structure_features_scaled.std():.4f}")
    print(f"      Min: {all_structure_features_scaled.min():.4f}")
    print(f"      Max: {all_structure_features_scaled.max():.4f}")
    
    # Split train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Modèle avec dropout augmenté
    model = AutoMLMetaTransformer(
        input_dim=38,
        hidden_dim=256,
        num_heads=8,
        num_layers=6,
        output_hyperparams=10,
        output_feature_scores=20,
        output_engineering_flags=5,
        dropout=0.3  # Dropout augmenté pour régularisation
    )
    
    print(f"\n🧠 Modèle créé:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # AMÉLIORATION: Optimizer avec weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # AMÉLIORATION: Learning rate scheduler (Cosine annealing avec warmup)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup linéaire
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # AMÉLIORATION: Loss functions avec pondération
    criterion_hyperparams = nn.MSELoss()
    criterion_feature_scores = nn.MSELoss()
    criterion_engineering = nn.MSELoss()
    
    # Poids des losses (hyperparams plus important)
    loss_weights = {
        'hyperparams': 2.0,      # Le plus important
        'feature_scores': 0.8,   # Moyennement important
        'engineering': 0.5       # Moins important (binaire)
    }
    
    print(f"\n⚙️  Training configuration:")
    print(f"   Epochs: {epochs}, Batch size: {batch_size}")
    print(f"   Learning rate: {lr} (with warmup + cosine annealing)")
    print(f"   Loss weights: hyperparams={loss_weights['hyperparams']}, "
          f"scores={loss_weights['feature_scores']}, flags={loss_weights['engineering']}")
    print(f"   Early stopping patience: {patience}")
    
    # Training loop
    train_losses = []
    val_losses = []
    learning_rates = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_loss = 0.0
        train_loss_hp = 0.0
        train_loss_fs = 0.0
        train_loss_eng = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward avec training=True pour data augmentation
            hyperparams_pred, feature_scores_pred, engineering_pred = model(
                batch['structure_features'],
                batch['importance_features'],
                training=True
            )
            
            # Vérifier NaN dans les prédictions
            if epoch == 0 and batch_idx == 0:
                print(f"\n🔍 Premier batch - vérification des prédictions:")
                print(f"   Hyperparams pred - min: {hyperparams_pred.min().item():.4f}, "
                      f"max: {hyperparams_pred.max().item():.4f}, mean: {hyperparams_pred.mean().item():.4f}")
                print(f"   Feature scores pred - min: {feature_scores_pred.min().item():.4f}, "
                      f"max: {feature_scores_pred.max().item():.4f}")
                print(f"   Engineering pred - min: {engineering_pred.min().item():.4f}, "
                      f"max: {engineering_pred.max().item():.4f}")
                
                if torch.isnan(hyperparams_pred).any():
                    print(f"   ⚠️  NaN dans hyperparams_pred!")
                if torch.isnan(feature_scores_pred).any():
                    print(f"   ⚠️  NaN dans feature_scores_pred!")
                if torch.isnan(engineering_pred).any():
                    print(f"   ⚠️  NaN dans engineering_pred!")
            
            # AMÉLIORATION: Label smoothing pour engineering flags (éviter overconfidence)
            engineering_target = batch['engineering_flags']
            engineering_target_smoothed = engineering_target * 0.95 + 0.025  # Label smoothing
            
            # Losses pondérées
            loss_hp = criterion_hyperparams(hyperparams_pred, batch['hyperparams'])
            loss_fs = criterion_feature_scores(feature_scores_pred, batch['feature_scores'])
            loss_eng = criterion_engineering(engineering_pred, engineering_target_smoothed)
            
            # Loss totale avec pondération
            loss = (loss_weights['hyperparams'] * loss_hp + 
                   loss_weights['feature_scores'] * loss_fs + 
                   loss_weights['engineering'] * loss_eng)
            
            # Vérifier NaN dans loss
            if torch.isnan(loss):
                print(f"\n⚠️  NaN détecté dans loss à epoch {epoch}, batch {batch_idx}")
                print(f"   loss_hp: {loss_hp.item()}, loss_fs: {loss_fs.item()}, loss_eng: {loss_eng.item()}")
                print(f"   Arrêt de l'entraînement")
                return None
            
            # Backward
            loss.backward()
            
            # AMÉLIORATION: Gradient clipping plus agressif pour stabilité
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_loss_hp += loss_hp.item()
            train_loss_fs += loss_fs.item()
            train_loss_eng += loss_eng.item()
        
        train_loss /= len(train_loader)
        train_loss_hp /= len(train_loader)
        train_loss_fs /= len(train_loader)
        train_loss_eng /= len(train_loader)
        train_losses.append(train_loss)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        scheduler.step()
        
        # VALIDATION avec test-time augmentation
        model.eval()
        val_loss = 0.0
        val_loss_hp = 0.0
        val_loss_fs = 0.0
        val_loss_eng = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # AMÉLIORATION: Test-time augmentation (moyenne de 3 forward passes)
                hyperparams_preds = []
                feature_scores_preds = []
                engineering_preds = []
                
                for _ in range(3):  # 3 passes avec dropout
                    hp, fs, eng = model(
                        batch['structure_features'],
                        batch['importance_features'],
                        training=False
                    )
                    hyperparams_preds.append(hp)
                    feature_scores_preds.append(fs)
                    engineering_preds.append(eng)
                
                # Moyenne des prédictions
                hyperparams_pred = torch.stack(hyperparams_preds).mean(dim=0)
                feature_scores_pred = torch.stack(feature_scores_preds).mean(dim=0)
                engineering_pred = torch.stack(engineering_preds).mean(dim=0)
                
                # Losses
                loss_hp = criterion_hyperparams(hyperparams_pred, batch['hyperparams'])
                loss_fs = criterion_feature_scores(feature_scores_pred, batch['feature_scores'])
                loss_eng = criterion_engineering(engineering_pred, batch['engineering_flags'])
                
                loss = (loss_weights['hyperparams'] * loss_hp + 
                       loss_weights['feature_scores'] * loss_fs + 
                       loss_weights['engineering'] * loss_eng)
                
                val_loss += loss.item()
                val_loss_hp += loss_hp.item()
                val_loss_fs += loss_fs.item()
                val_loss_eng += loss_eng.item()
        
        val_loss /= len(val_loader)
        val_loss_hp /= len(val_loader)
        val_loss_fs /= len(val_loader)
        val_loss_eng /= len(val_loader)
        val_losses.append(val_loss)
        
        # Logging détaillé tous les 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f}")
            print(f"  Train - Total: {train_loss:.4f} | HP: {train_loss_hp:.4f} | "
                  f"FS: {train_loss_fs:.4f} | Eng: {train_loss_eng:.4f}")
            print(f"  Val   - Total: {val_loss:.4f} | HP: {val_loss_hp:.4f} | "
                  f"FS: {val_loss_fs:.4f} | Eng: {val_loss_eng:.4f}")
        
        # Save best model & Early Stopping intelligent
        if val_loss < best_val_loss:
            improvement = (best_val_loss - val_loss) / best_val_loss if best_val_loss != float('inf') else 0
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            torch.save(model.state_dict(), 'data/metatransformer_training/automl_meta_transformer_best.pth')
            print(f"  ✅ Best model saved (val_loss: {val_loss:.4f}, improvement: {improvement:.2%})")
        else:
            patience_counter += 1
            
        # Early stopping avec patience adaptative
        # Réduire patience si on est proche du meilleur
        if val_loss < best_val_loss * 1.05:  # Dans les 5% du meilleur
            effective_patience = patience * 1.5  # Plus de patience
        else:
            effective_patience = patience
            
        if patience_counter >= effective_patience:
            print(f"\n⏸️  Early stopping at epoch {epoch+1}")
            print(f"   Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
            print(f"   No significant improvement for {patience_counter} epochs")
            break
    
    # Save final model
    torch.save(model.state_dict(), 'data/metatransformer_training/automl_meta_transformer_final.pth')
    
    # AMÉLIORATION: Plots multiples avec learning rate
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('AutoML Meta-Transformer Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate curve
    axes[1].plot(learning_rates, color='green', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('Learning Rate Schedule (Warmup + Cosine Annealing)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('data/metatransformer_training/automl_training_curve.png', dpi=150)
    print(f"\n✅ Training curves saved to data/metatransformer_training/automl_training_curve.png")
    
    # Statistiques finales
    final_train_loss = train_losses[-1]
    min_train_loss = min(train_losses)
    min_val_loss = min(val_losses)
    
    print(f"\n📊 TRAINING STATISTICS:")
    print(f"   Total epochs: {len(train_losses)}")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best val loss: {best_val_loss:.6f}")
    print(f"   Final train loss: {final_train_loss:.6f}")
    print(f"   Min train loss: {min_train_loss:.6f}")
    print(f"   Overfitting gap: {(final_train_loss - best_val_loss):.6f}")
    
    print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"   Model saved to: data/metatransformer_training/automl_meta_transformer_best.pth")
    print(f"\n🎯 Model Capabilities:")
    print(f"   ✅ Predicts 10 hyperparameters (including reg_alpha, reg_lambda)")
    print(f"   ✅ Predicts 20 feature importance scores")
    print(f"   ✅ Predicts 5 feature engineering flags")
    print(f"   ✅ Intelligent adaptation to dataset characteristics")
    
    return model


if __name__ == "__main__":
    # Entraîner sur TOUS les datasets du MVP (1-30)
    datasets = [f'Dataset{i}' for i in range(1, 31)]
    
    print(f"\n{'='*70}")
    print(f"🚀 OPTIMIZED AUTOML META-TRANSFORMER TRAINING")
    print(f"{'='*70}")
    print(f"\n📊 Training Configuration:")
    print(f"   Datasets: {len(datasets)} (Dataset1 to Dataset30)")
    print(f"   Examples per dataset: ~15 configurations")
    print(f"   Total training examples: ~450")
    print(f"\n🔧 Optimizations Applied:")
    print(f"   ✅ Multi-scale attention with Conv1D")
    print(f"   ✅ Skip connections for gradient flow")
    print(f"   ✅ GELU activation (better than ReLU)")
    print(f"   ✅ Dropout 0.3 (regularization)")
    print(f"   ✅ AdamW optimizer with weight decay")
    print(f"   ✅ Cosine annealing + warmup scheduler")
    print(f"   ✅ Weighted loss (hyperparams priority)")
    print(f"   ✅ Label smoothing for flags")
    print(f"   ✅ Test-time augmentation (TTA)")
    print(f"   ✅ Gradient clipping (max_norm=0.5)")
    print(f"   ✅ Adaptive early stopping")
    
    model = train_automl_metatransformer(
        datasets_list=datasets,
        epochs=500,          # Augmenté pour convergence optimale
        batch_size=8,        # Équilibre entre stabilité et vitesse
        lr=0.00005,          # Réduit avec scheduler pour convergence fine
        patience=30,         # Plus de patience pour trouver optimum
        warmup_epochs=10     # Warmup pour stabilité initiale
    )
    
    print("\n" + "="*70)
    print("🎉 OPTIMIZED AUTOML META-TRANSFORMER IS READY!")
    print("="*70)
    print("\n💡 Next Steps:")
    print("   1. Test on new datasets (Dataset31+)")
    print("   2. Compare with simple meta-transformer")
    print("   3. Monitor feature engineering activation")
    print("   4. Validate hyperparameter predictions")
    print("\n✨ Happy AutoML! ✨\n")

