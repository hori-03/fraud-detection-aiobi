"""
Meta-Transformer Training Script
Entraîne un modèle Transformer pour prédire les hyperparamètres XGBoost optimaux
basé sur le dataset unifié metatransformer_dataset.csv
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class MetaTransformer(nn.Module):
    """
    Architecture Transformer optimisée pour prédire les hyperparamètres XGBoost
    AMÉLIORATIONS: Attention multi-scale, Skip connections, Label smoothing
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_heads=8, num_layers=6, 
                 output_dim=10, dropout=0.3):
        super(MetaTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # NOUVEAU: Input projection avec skip connection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # GELU au lieu de ReLU
            nn.Dropout(dropout)
        )
        
        # NOUVEAU: Multi-scale feature extraction
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=k, padding=k//2)
            for k in [1, 3, 5]
        ])
        self.multi_scale_combine = nn.Linear(hidden_dim * 3 // 4, hidden_dim)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # NOUVEAU: Multi-scale feature extraction
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=k, padding=k//2)
            for k in [1, 3, 5]
        ])
        self.multi_scale_combine = nn.Linear(hidden_dim * 3 // 4, hidden_dim)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Transformer Encoder avec dropout augmenté
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # NOUVEAU: Attention pooling pour mieux capturer l'info
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Output head - AMÉLIORÉ avec plus de capacité
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 3),
            nn.LayerNorm(hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier initialization pour meilleure convergence"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        
        # Project to hidden dimension
        x_proj = self.input_projection(x)  # (batch_size, hidden_dim)
        
        # NOUVEAU: Multi-scale feature extraction
        x_for_conv = x_proj.unsqueeze(1).permute(0, 2, 1)  # (batch_size, hidden_dim, 1)
        multi_scale_features = []
        for conv in self.multi_scale_conv:
            feat = conv(x_for_conv)  # (batch_size, hidden_dim//4, 1)
            multi_scale_features.append(feat.squeeze(-1))
        multi_scale = torch.cat(multi_scale_features, dim=1)  # (batch_size, 3*hidden_dim//4)
        multi_scale = self.multi_scale_combine(multi_scale)  # (batch_size, hidden_dim)
        
        # Combiner avec projection originale (skip connection)
        x = x_proj + multi_scale  # (batch_size, hidden_dim)
        
        # Add batch dimension for transformer: (batch_size, 1, hidden_dim)
        x = x.unsqueeze(1)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch_size, 1, hidden_dim)
        
        # NOUVEAU: Attention pooling au lieu de simple squeeze
        attention_weights = self.attention_pool(x)  # (batch_size, 1, 1)
        x = (x * attention_weights).sum(dim=1)  # (batch_size, hidden_dim)
        
        # Output prediction
        output = self.output_head(x)  # (batch_size, output_dim)
        
        return output


class MetaTransformerTrainer:
    """Classe pour gérer l'entraînement du Meta-Transformer"""
    
    def __init__(self, dataset_path='data/metatransformer_training/unified_metatransformer_dataset.csv'):
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device: {self.device}")
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.dataset_encoder = LabelEncoder()
        
        # Pour sauvegarder les noms
        self.feature_names = []
        self.target_names = ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 
                            'colsample_bytree', 'gamma', 'min_child_weight', 
                            'scale_pos_weight', 'reg_alpha', 'reg_lambda']  # 10 hyperparams
        
    def load_and_prepare_data(self):
        """Charger et préparer le dataset unifié"""
        print("📂 Chargement du dataset unifié...")
        
        df = pd.read_csv(self.dataset_path)
        print(f"   Shape: {df.shape}")
        print(f"   Nb exemples: {len(df)}")
        
        # Note: dataset_name/dataset_encoded supprimés pour améliorer généralisation
        # Le modèle apprend maintenant uniquement sur les caractéristiques structurelles
        
        # Encoder target_column
        if 'target_column' in df.columns:
            target_encoder = LabelEncoder()
            df['target_column_encoded'] = target_encoder.fit_transform(df['target_column'].fillna('unknown'))
        
        # Encoder target_score_category
        if 'target_score_category' in df.columns:
            score_cat_encoder = LabelEncoder()
            df['target_score_category_encoded'] = score_cat_encoder.fit_transform(df['target_score_category'].fillna('unknown'))
        
        # Séparer features et targets
        # AMÉLIORÉ: Utiliser les features normalisées si disponibles
        exclude_cols = ['target_cv_score', 'target_test_score', 'target_rank', 'target_score_category',
                       'target_precision', 'target_recall', 'target_roc_auc',
                       'max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree',
                       'gamma', 'min_child_weight', 'scale_pos_weight', 'reg_alpha', 'reg_lambda',
                       'target_column']  # On utilise target_column_encoded si disponible
        
        # Priorité aux features normalisées (_norm suffix)
        available_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Séparer normalisées et non-normalisées
        norm_cols = [col for col in available_cols if col.endswith('_norm')]
        non_norm_cols = [col for col in available_cols if not col.endswith('_norm')]
        
        # STRATÉGIE: Utiliser uniquement les normalisées si disponibles, sinon tout
        if len(norm_cols) > 0:
            print(f"\n✅ Utilisation des features normalisées: {len(norm_cols)} colonnes")
            feature_cols = norm_cols + ['target_column_encoded', 'target_score_category_encoded']
        else:
            print(f"\n⚠️  Aucune feature normalisée, utilisation de toutes: {len(non_norm_cols)}")
            feature_cols = available_cols
        
        # Nettoyer les colonnes qui n'existent pas
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df[self.target_names].values
        
        print(f"\n✅ Features: {X.shape} - {len(feature_cols)} colonnes")
        print(f"✅ Targets: {y.shape} - {len(self.target_names)} hyperparamètres")
        
        # Afficher quelques features
        print(f"\n📊 Quelques features utilisées:")
        for i, name in enumerate(feature_cols[:10]):
            print(f"   {i+1}. {name}")
        if len(feature_cols) > 10:
            print(f"   ... et {len(feature_cols)-10} autres")
        
        # Vérifier les valeurs manquantes
        nan_counts = np.isnan(X).sum(axis=0)
        if nan_counts.sum() > 0:
            print(f"\n⚠️  Valeurs manquantes détectées:")
            for i, count in enumerate(nan_counts):
                if count > 0:
                    print(f"   {feature_cols[i]}: {count} NaN")
            # AMÉLIORATION: Imputation par médiane au lieu de 0
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            self.imputer = imputer  # Sauvegarder pour production
        
        # Normalisation - AMÉLIORATION: Robuste aux outliers
        print(f"\n🔄 Normalisation des données...")
        from sklearn.preprocessing import RobustScaler
        self.feature_scaler = RobustScaler()  # Plus robuste que StandardScaler
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)
        
        # Split train/val/test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_scaled, y_scaled, test_size=0.3, random_state=RANDOM_SEED
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED
        )
        
        print(f"\n📊 Split des données:")
        print(f"   Train: {X_train.shape[0]} exemples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"   Val:   {X_val.shape[0]} exemples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"   Test:  {X_test.shape[0]} exemples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Convertir en tensors
        self.train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        self.val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        self.test_dataset = TensorDataset(
            torch.FloatTensor(X_test), 
            torch.FloatTensor(y_test)
        )
        
        return X_train.shape[1]  # input_dim
    
    def create_dataloaders(self, batch_size=32):  # AMÉLIORATION: batch_size augmenté
        """Créer les DataLoaders avec data augmentation"""
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Pour Windows
            pin_memory=True if torch.cuda.is_available() else False
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"✅ DataLoaders créés (batch_size={batch_size})")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
        print(f"   Test batches: {len(self.test_loader)}")
    
    def train_epoch(self, model, optimizer, criterion, use_mixup=True, mixup_alpha=0.2):
        """Entraîner une epoch avec MixUp data augmentation"""
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in self.train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # NOUVEAU: MixUp augmentation pour régularisation
            if use_mixup and np.random.rand() < 0.5:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                batch_size = batch_X.size(0)
                index = torch.randperm(batch_size).to(self.device)
                
                mixed_X = lam * batch_X + (1 - lam) * batch_X[index]
                mixed_y = lam * batch_y + (1 - lam) * batch_y[index]
                
                optimizer.zero_grad()
                predictions = model(mixed_X)
                loss = criterion(predictions, mixed_y)
            else:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
            
            loss.backward()
            
            # Gradient clipping pour stabilité
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self, model, criterion):
        """Valider le modèle"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                total_loss += loss.item()
                
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # Métriques
        mae = mean_absolute_error(all_targets, all_preds)
        mse = mean_squared_error(all_targets, all_preds)
        rmse = np.sqrt(mse)
        
        return total_loss / len(self.val_loader), mae, rmse
    
    def train(self, num_epochs=300, batch_size=32, learning_rate=0.0005, 
              hidden_dim=256, num_heads=8, num_layers=6, dropout=0.3):
        """Pipeline d'entraînement complet AMÉLIORÉ"""
        
        print("\n" + "="*70)
        print("🚀 DÉMARRAGE ENTRAÎNEMENT META-TRANSFORMER (VERSION OPTIMISÉE)")
        print("="*70)
        
        # 1. Préparer les données
        input_dim = self.load_and_prepare_data()
        self.create_dataloaders(batch_size)
        
        # 2. Créer le modèle
        print(f"\n🏗️  Construction du modèle...")
        model = MetaTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=len(self.target_names),
            dropout=dropout
        ).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total paramètres: {total_params:,}")
        print(f"   Paramètres entraînables: {trainable_params:,}")
        
        # 3. Optimizer et Loss AMÉLIORÉS
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # NOUVEAU: Cosine Annealing avec warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # NOUVEAU: Huber Loss plus robuste que MSE
        criterion = nn.SmoothL1Loss()  # Robuste aux outliers
        
        # 4. Training loop
        print(f"\n🎯 Entraînement sur {num_epochs} epochs...")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Batch size: {batch_size}")
        
        history = {
            'train_loss': [], 'val_loss': [], 
            'val_mae': [], 'val_rmse': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = 30  # AMÉLIORATION: Patience augmentée
        
        for epoch in range(num_epochs):
            # Train avec MixUp
            train_loss = self.train_epoch(model, optimizer, criterion, use_mixup=True)
            
            # Validate
            val_loss, val_mae, val_rmse = self.validate(model, criterion)
            
            # Scheduler step
            scheduler.step()
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            history['val_rmse'].append(val_rmse)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                self.best_model_state = model.state_dict()
            else:
                patience_counter += 1
            
            # Affichage détaillé
            if (epoch + 1) % 5 == 0 or epoch == 0:  # Plus fréquent
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train: {train_loss:.6f} | "
                      f"Val: {val_loss:.6f} | "
                      f"MAE: {val_mae:.6f} | "
                      f"RMSE: {val_rmse:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Patience: {patience_counter}/{patience_limit}")
            
            # Early stopping
            if patience_counter >= patience_limit:
                print(f"\n⏹️  Early stopping à epoch {epoch+1} (patience={patience_limit})")
                break
        
        # 5. Charger le meilleur modèle
        model.load_state_dict(self.best_model_state)
        
        print(f"\n✅ Entraînement terminé!")
        print(f"   Meilleure val_loss: {best_val_loss:.6f}")
        
        return model, history
    
    def evaluate_on_test(self, model):
        """Évaluer sur le test set"""
        print("\n" + "="*70)
        print("📊 ÉVALUATION SUR TEST SET")
        print("="*70)
        
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X = batch_X.to(self.device)
                predictions = model(batch_X)
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # Inverse transform pour obtenir les vraies valeurs
        all_preds_real = self.target_scaler.inverse_transform(all_preds)
        all_targets_real = self.target_scaler.inverse_transform(all_targets)
        
        # Métriques globales
        mae = mean_absolute_error(all_targets_real, all_preds_real)
        mse = mean_squared_error(all_targets_real, all_preds_real)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets_real, all_preds_real)
        
        print(f"\n📈 Métriques globales:")
        print(f"   MAE:  {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²:   {r2:.4f}")
        
        # Métriques par hyperparamètre
        print(f"\n📊 Métriques par hyperparamètre:")
        for i, name in enumerate(self.target_names):
            mae_hp = mean_absolute_error(all_targets_real[:, i], all_preds_real[:, i])
            r2_hp = r2_score(all_targets_real[:, i], all_preds_real[:, i])
            print(f"   {name:20s} - MAE: {mae_hp:8.4f} | R²: {r2_hp:6.3f}")
        
        return {
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'predictions': all_preds_real,
            'targets': all_targets_real
        }
    
    def plot_training_history(self, history, save_path='data/models'):
        """Visualiser l'historique d'entraînement"""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Meta-Transformer Training History', fontsize=16, fontweight='bold')
        
        # Loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # MAE
        axes[0, 1].plot(history['val_mae'], label='Val MAE', color='green', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Validation MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # RMSE
        axes[1, 0].plot(history['val_rmse'], label='Val RMSE', color='orange', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('Validation RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Learning curve (overfitting check)
        axes[1, 1].plot(history['train_loss'], label='Train', linewidth=2)
        axes[1, 1].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Overfitting Check')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        save_file = Path(save_path) / 'training_history.png'
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Graphique sauvegardé: {save_file}")
        plt.close()
    
    def save_model(self, model, save_dir='data/models'):
        """Sauvegarder le modèle et les preprocessors"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Sauvegarde du modèle...")
        
        # 1. Sauvegarder le modèle PyTorch
        model_path = save_dir / 'metatransformer_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': model.input_dim,
                'hidden_dim': model.hidden_dim,
            },
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'timestamp': datetime.now().isoformat()
        }, model_path)
        print(f"   ✅ Modèle PyTorch: {model_path}")
        
        # 2. Sauvegarder les scalers et encoders
        processor_path = save_dir / 'metatransformer_processor.pkl'
        with open(processor_path, 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'dataset_encoder': self.dataset_encoder,
                'feature_names': self.feature_names,
                'target_names': self.target_names
            }, f)
        print(f"   ✅ Preprocessors: {processor_path}")
        
        # 3. Sauvegarder les métadonnées
        metadata_path = save_dir / 'metatransformer_metadata.json'
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_architecture': 'MetaTransformer',
            'input_dim': model.input_dim,
            'hidden_dim': model.hidden_dim,
            'num_features': len(self.feature_names),
            'num_targets': len(self.target_names),
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'note': 'dataset_name removed to improve OOD generalization',
            'total_parameters': sum(p.numel() for p in model.parameters()),
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Métadonnées: {metadata_path}")
        
        print(f"\n✅ Modèle sauvegardé avec succès!")


def main():
    """Fonction principale"""
    
    print("\n" + "="*70)
    print("🎯 META-TRANSFORMER TRAINING")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    config = {
        'num_epochs': 200,
        'batch_size': 16,
        'learning_rate': 0.001,
        'hidden_dim': 128,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.2
    }
    
    print(f"\n⚙️  Configuration:")
    for key, value in config.items():
        print(f"   {key:20s}: {value}")
    
    # 1. Créer le trainer
    trainer = MetaTransformerTrainer()
    
    # 2. Entraîner le modèle
    model, history = trainer.train(**config)
    
    # 3. Évaluer sur test
    test_results = trainer.evaluate_on_test(model)
    
    # 4. Visualiser l'historique
    trainer.plot_training_history(history)
    
    # 5. Sauvegarder le modèle
    trainer.save_model(model)
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*70)
    print(f"\n📊 RÉSULTATS FINAUX:")
    print(f"   Test MAE:  {test_results['mae']:.4f}")
    print(f"   Test RMSE: {test_results['rmse']:.4f}")
    print(f"   Test R²:   {test_results['r2']:.4f}")
    
    print(f"\n💡 PROCHAINES ÉTAPES:")
    print(f"   1. Utiliser auto_xgboost_generator.py pour générer des configs")
    print(f"   2. Tester sur de nouveaux datasets")
    print(f"   3. Affiner les hyperparamètres si nécessaire")


if __name__ == "__main__":
    main()
