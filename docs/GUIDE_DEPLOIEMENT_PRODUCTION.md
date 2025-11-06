# 🚀 GUIDE DE DÉPLOIEMENT EN PRODUCTION

## 🔥 PROBLÈME: Fichiers Locaux Non Disponibles sur Railway

Railway est **éphémère** : les fichiers disparaissent à chaque redémarrage. Vos 40 modèles XGBoost (`.joblib`, ~2 GB total) ne seront PAS persistants !

---

## 💡 SOLUTION IMPLÉMENTÉE: Architecture Hybride

### 📊 PostgreSQL (Railway - Gratuit)
- **Stocke:** Métadonnées des 40 modèles (rapide, <100ms)
- **Table:** `reference_models` avec colonnes complètes
- **Usage:** Auto-matching, statistiques d'utilisation

### ☁️ AWS S3 (Production - Recommandé)
- **Stocke:** Fichiers `.joblib` (modèles XGBoost)
- **Coût:** ~$0.05-0.10/mois pour 40 modèles
- **Avantages:** Persistant, rapide, scalable

### 💾 Cache Local (Railway - Temporaire)
- **Dossier:** `/tmp/model_cache/`
- **Usage:** Télécharge une fois de S3, réutilise
- **Expiration:** Nettoyé au redémarrage (acceptable)

---

## 🏗️ ARCHITECTURE DE PRODUCTION

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER UPLOADS CSV                         │
│                     (unlabeled dataset)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FLASK APP (Railway)                          │
│                                                                 │
│  1. Extract columns: ['tx_amount', 'merchant', 'timestamp']    │
│                                                                 │
│  2. Query PostgreSQL:                                           │
│     ReferenceModel.find_best_match(columns)                     │
│     → Returns: dataset16 (similarity: 56%)                      │
│                                                                 │
│  3. Load model from S3 (via ModelStorageService):              │
│     - Check cache: /tmp/model_cache/dataset16/                 │
│     - If not cached: Download from S3                           │
│     - Load: xgboost_model.joblib (30 MB)                       │
│                                                                 │
│  4. Apply ensemble predictions:                                 │
│     - Top-3 models: dataset16, dataset13, dataset10            │
│     - Weighted average predictions                              │
│                                                                 │
│  5. Anomaly detection + Calibration                            │
│                                                                 │
│  6. Return predictions CSV                                      │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ├─────────────────┬────────────────────────────────┐
            │                 │                                │
            ▼                 ▼                                ▼
    ┌───────────────┐  ┌──────────────┐          ┌──────────────────┐
    │   PostgreSQL  │  │   AWS S3     │          │   Local Cache    │
    │   (Railway)   │  │              │          │   (/tmp/)        │
    ├───────────────┤  ├──────────────┤          ├──────────────────┤
    │ ✓ Métadonnées │  │ ✓ .joblib    │          │ ✓ Temp storage   │
    │ ✓ Fast query  │  │ ✓ Persistent │          │ ✓ Fast access    │
    │ ✓ Statistiques│  │ ✓ Scalable   │          │ ✗ Ephemeral      │
    └───────────────┘  └──────────────┘          └──────────────────┘
```

---

## 📝 ÉTAPES DE DÉPLOIEMENT

### ✅ ÉTAPE 1: Préparer AWS S3 (5 minutes)

#### 1.1 Créer un compte AWS (si nécessaire)
- Allez sur https://aws.amazon.com/free/
- 12 mois gratuits (5 GB S3 inclus)

#### 1.2 Créer un bucket S3
```bash
# Installer AWS CLI
pip install awscli

# Configurer les credentials
aws configure
# AWS Access Key ID: <votre_access_key>
# AWS Secret Access Key: <votre_secret_key>
# Default region: us-east-1
# Default output format: json

# Créer le bucket
aws s3 mb s3://fraud-detection-models

# Vérifier
aws s3 ls
```

#### 1.3 Configurer les permissions (IAM)
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::fraud-detection-models/*",
        "arn:aws:s3:::fraud-detection-models"
      ]
    }
  ]
}
```

---

### ✅ ÉTAPE 2: Migrer les Modèles Locaux → S3

#### 2.1 Installer boto3
```bash
cd APP_autoML
pip install boto3
```

#### 2.2 Estimer les coûts
```bash
python migrate_models_to_s3.py --estimate
```

**Output attendu:**
```
================================================================================
S3 COST ESTIMATION
================================================================================
  Total models: 40
  Total size: 1875.34 MB (1.832 GB)

  Costs (monthly):
    Storage: $0.0421/month
    Upload (one-time): $0.0008
    Yearly storage: $0.51/year

  💡 Tip: Use S3 Intelligent-Tiering for cost optimization
================================================================================
```

#### 2.3 Dry-run (simulation)
```bash
python migrate_models_to_s3.py --bucket fraud-detection-models --dry-run
```

#### 2.4 Migration réelle
```bash
python migrate_models_to_s3.py --bucket fraud-detection-models
```

**Output attendu:**
```
Migrating: dataset1
  Local path: data/automl_models/dataset1
  S3 path: s3://fraud-detection-models/automl_models/dataset1/
    Uploading xgboost_model.joblib...
    ✓ xgboost_model.joblib uploaded
    Uploading feature_engineer.joblib...
    ✓ feature_engineer.joblib uploaded
    ...
  ✅ Migration successful

...

================================================================================
MIGRATION SUMMARY
================================================================================
  Total models: 40
  ✅ Migrated: 40
  ❌ Failed: 0

✅ Migration complete!
```

#### 2.5 Vérifier la migration
```bash
python migrate_models_to_s3.py --bucket fraud-detection-models --verify
```

---

### ✅ ÉTAPE 3: Configurer Railway

#### 3.1 Ajouter les variables d'environnement
Dans Railway Dashboard → Variables:

```env
# AWS Credentials
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1

# S3 Bucket
S3_MODEL_BUCKET=fraud-detection-models

# PostgreSQL (auto-configuré par Railway)
DATABASE_URL=postgresql://...

# Flask
FLASK_ENV=production
SECRET_KEY=<votre_secret_key>
```

#### 3.2 Mettre à jour requirements.txt
```bash
# Ajouter boto3
echo "boto3==1.34.42" >> requirements.txt
```

#### 3.3 Pousser sur Railway
```bash
git add .
git commit -m "feat: Add S3 storage support for production models"
git push origin main
```

Railway va automatiquement:
1. Rebuild l'image Docker
2. Créer la table `reference_models` (via migrations)
3. Configurer les variables d'environnement

---

### ✅ ÉTAPE 4: Peupler la Base de Données PostgreSQL

#### 4.1 SSH dans Railway (ou run via dashboard)
```bash
# Dans Railway Dashboard → Shell
cd /app
python populate_reference_models.py
```

**OU via script local:**
```bash
# Avec DATABASE_URL de Railway
export DATABASE_URL="postgresql://user:pass@host:5432/db"
python populate_reference_models.py
```

**Output attendu:**
```
================================================================================
📦 POPULATING REFERENCE MODELS FROM data/automl_models/
================================================================================

Processing dataset1...
  ✓ Model metadata loaded
  ✓ Performance metrics loaded
  ✓ Storage type: s3
  ✓ S3 path: s3://fraud-detection-models/automl_models/dataset1/
  ✅ Added: dataset1

...

================================================================================
✅ POPULATION COMPLETED
================================================================================
  Total: 40 models added
  Skipped: 0 (already exists)
  Failed: 0
================================================================================
```

---

### ✅ ÉTAPE 5: Tester en Production

#### 5.1 Via l'interface web
1. Allez sur https://your-app.railway.app/upload
2. Uploadez un CSV non étiqueté
3. Cochez "Dataset non étiqueté"
4. Cliquez "Appliquer le modèle"
5. Vérifiez les logs:

```
🔍 Finding best matching reference models...
✅ Best match: dataset16 (similarity: 56.3%)

📦 ModelStorageService: Loading model dataset16 (storage: s3)
  Loading from S3: s3://fraud-detection-models/automl_models/dataset16/
  ✓ Using cached version: /tmp/model_cache/dataset16
  ✓ Model loaded

🤖 Applying ensemble predictions (top 3 models)...
  Top-3 models: dataset16, dataset13, dataset10
  ✅ Predictions complete (5000 transactions)
```

#### 5.2 Via API (cURL)
```bash
curl -X POST https://your-app.railway.app/api/apply_unlabeled \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "filepath": "uploads/test.csv",
    "model_name": "unlabeled_predictions"
  }'
```

**Response attendu:**
```json
{
  "success": true,
  "message": "Prédictions générées avec succès sur 5000 transactions",
  "predictions_file": "uploads/predictions/1_20251104_unlabeled.csv",
  "download_url": "/download/predictions/1_20251104_unlabeled.csv",
  "stats": {
    "total_transactions": 5000,
    "high_risk": 50,
    "medium_risk": 200,
    "low_risk": 4750,
    "anomalies_detected": 25
  },
  "methods_used": {
    "ensemble": true,
    "top_k_models": 3,
    "anomaly_detection": true,
    "calibration": true,
    "best_model": "dataset16",
    "similarity_score": 0.563
  }
}
```

---

## 🔧 MAINTENANCE

### Ajouter un Nouveau Modèle

```bash
# 1. Entraîner localement
python full_automl.py --dataset new_data.csv

# 2. Upload vers S3
python migrate_models_to_s3.py --bucket fraud-detection-models

# 3. Ajouter à PostgreSQL
python populate_reference_models.py
```

### Nettoyer le Cache

```python
from app.services.model_storage import get_storage_service

storage = get_storage_service()
storage.clear_cache()  # Nettoie tout le cache
storage.clear_cache('dataset16')  # Nettoie un modèle spécifique
```

### Rollback vers Local (développement)

```bash
python migrate_models_to_s3.py --rollback
```

---

## 💰 COÛTS ESTIMÉS

### AWS S3 (Storage)
- **40 modèles** (~1.8 GB total)
- **$0.023/GB/month** = **$0.04/month**
- **$0.50/year**

### AWS S3 (Data Transfer)
- **GET requests:** $0.0004 per 1,000 requests
- **Estimation:** 1000 prédictions/jour = 3 GET requests/prédiction = 90,000 requests/mois
- **Coût:** $0.036/month
- **Avec cache:** Divisé par 10 = **$0.004/month**

### Railway (Base de Données PostgreSQL)
- **Gratuit** (jusqu'à 500 MB)
- Table `reference_models`: ~5 MB
- **$0/month**

### TOTAL ESTIMÉ: **$0.05-0.10/month** (~€0.05-0.10)

---

## 🚨 ALTERNATIVES (Si S3 trop complexe)

### Option B: Railway Volumes
```bash
# Dans Railway Dashboard
# Add Volume: /data (1 GB = $0.25/month)
# Mount à: /app/data
# Copier modèles dans le volume
```

**Avantages:** Simple, intégré Railway  
**Inconvénients:** Coût plus élevé ($3/year vs $0.50/year)

### Option C: Docker Image (Simple mais volumineux)
```dockerfile
# Dans Dockerfile
COPY data/automl_models/ /app/data/automl_models/
```

**Avantages:** Ultra simple, gratuit  
**Inconvénients:** Image énorme (>2 GB), build lent, pas de mise à jour dynamique

---

## 📊 COMPARAISON DES SOLUTIONS

| Solution | Coût/mois | Persistant | Scalable | Complexité | Recommandation |
|----------|-----------|------------|----------|------------|----------------|
| **AWS S3** | $0.05 | ✅ | ✅✅✅ | Medium | ⭐⭐⭐⭐⭐ |
| Railway Volumes | $0.25 | ✅ | ✅ | Low | ⭐⭐⭐⭐ |
| Docker Image | $0 | ✅ | ❌ | Very Low | ⭐⭐⭐ |
| PostgreSQL BLOB | $0 | ✅ | ✅ | Medium | ⭐⭐ |

**Recommandation finale:** **AWS S3** (meilleur rapport qualité/prix/scalabilité)

---

## ✅ CHECKLIST DE DÉPLOIEMENT

- [ ] Compte AWS créé
- [ ] Bucket S3 créé (`fraud-detection-models`)
- [ ] AWS credentials configurées
- [ ] boto3 installé (`pip install boto3`)
- [ ] Coûts estimés (`python migrate_models_to_s3.py --estimate`)
- [ ] Migration dry-run testée
- [ ] Migration vers S3 complète (40 modèles)
- [ ] Migration vérifiée (`--verify`)
- [ ] Variables d'environnement Railway configurées
- [ ] `requirements.txt` mis à jour avec `boto3`
- [ ] Code poussé sur Railway
- [ ] Table `reference_models` créée (migrations)
- [ ] Table populée avec 40 modèles
- [ ] Test avec CSV non étiqueté via web
- [ ] Test avec API endpoint
- [ ] Logs vérifiés (S3 download + cache)
- [ ] Predictions CSV téléchargées avec succès

---

## 🎉 RÉSULTAT FINAL

Votre app est maintenant **production-ready** :

1. ✅ **PostgreSQL** (Railway): Métadonnées rapides
2. ✅ **AWS S3**: Modèles persistants
3. ✅ **Cache local**: Optimisation performance
4. ✅ **Auto-matching**: Top-3 modèles similaires
5. ✅ **Ensemble + Anomaly + Calibration**: Prédictions robustes
6. ✅ **Coût**: ~$0.05/mois (~€0.05)

**Les utilisateurs peuvent uploader des CSV non étiquetés et obtenir des prédictions fiables en quelques secondes !** 🚀

---

**Document créé:** 4 novembre 2025  
**Auteur:** Fraud Detection AutoML System v2.0  
**Status:** ✅ PRÊT POUR PRODUCTION
