# 🚀 Checklist de Déploiement Railway - Aïobi Fraud Detection

## ✅ Pré-requis Vérifiés

### 1. Configuration Base de Données ✅
- [x] PostgreSQL Railway configuré
- [x] URL dans `.env`: `postgresql+psycopg://...`
- [x] Driver psycopg v3 installé (`psycopg==3.2.5`)
- [x] Fix URL automatique dans `config.py` (postgres:// → postgresql://)

### 2. Configuration AWS S3 ✅
- [x] Bucket S3 créé: `fraud-detection-ml-models`
- [x] 40 modèles AutoML uploadés dans S3
- [x] Credentials AWS dans `.env`:
  ```
  AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID_HERE
  AWS_SECRET_ACCESS_KEY=***
  AWS_DEFAULT_REGION=eu-north-1
  S3_MODEL_BUCKET=fraud-detection-ml-models
  STORAGE_TYPE=s3
  ```
- [x] boto3==1.34.42 dans requirements.txt

### 3. Configuration Google OAuth ✅
- [x] Client ID et Secret configurés
- [x] Redirect URI: `http://127.0.0.1:5000/auth/google/callback` (à mettre à jour pour production)
- [x] Bibliothèques installées: google-auth, google-auth-oauthlib

### 4. Métatransformer AutoML ✅
- [x] Modèle: `data/metatransformer_training/automl_meta_transformer_best.pth`
- [x] Sera copié dans l'image Docker
- [x] Chemin relatif dans `full_automl.py`: `Path(__file__).parent.parent / 'data' / ...`

### 5. Dockerfile ✅
- [x] Python 3.11-slim
- [x] Copie `automl_transformer/` depuis parent
- [x] Copie `data/metatransformer_training/` depuis parent
- [x] PYTHONPATH configuré: `/app:/`
- [x] Gunicorn avec 2 workers, timeout 300s

### 6. Fichiers de Configuration ✅
- [x] `Procfile`: gunicorn avec config optimale
- [x] `railway.json`: builder DOCKERFILE
- [x] `.dockerignore`: exclut fichiers inutiles
- [x] `.gitignore`: protège .env et credentials

## 🔧 Variables d'Environnement Railway

⚠️ **IMPORTANT**: Voir le guide détaillé [`docs/VARIABLES_ENVIRONNEMENT.md`](docs/VARIABLES_ENVIRONNEMENT.md)

À configurer dans Railway Dashboard → Variables:

```bash
# Flask (⚠️ Générer SECRET_KEY avec: python generate_secret_key.py)
SECRET_KEY=<générer-clé-sécurisée-production>
FLASK_ENV=production
FLASK_DEBUG=0

# Database (auto-injectée par Railway si PostgreSQL plugin ajouté)
# DATABASE_URL=<railway-postgresql-url> ← PAS BESOIN de la mettre, Railway l'injecte!

# AWS S3 (identique dev et prod)
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID_HERE
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY_HERE
AWS_DEFAULT_REGION=eu-north-1
S3_MODEL_BUCKET=fraud-detection-ml-models
STORAGE_TYPE=s3

# Google OAuth (client ID/secret identiques dev et prod)
GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID_HERE
GOOGLE_CLIENT_SECRET=YOUR_GOOGLE_CLIENT_SECRET_HERE

# ⚠️ GOOGLE_REDIRECT_URI: WORKFLOW EN 2 ÉTAPES
# ÉTAPE 1 (Premier déploiement): Utiliser valeur temporaire
GOOGLE_REDIRECT_URI=http://127.0.0.1:5000/auth/google/callback

# ÉTAPE 2 (Après déploiement): Mettre à jour avec l'URL Railway
# 1. Noter l'URL Railway (ex: https://fraud-detection-production.railway.app)
# 2. Mettre à jour cette variable vers: https://<ton-app>.railway.app/auth/google/callback
# 3. Ajouter cette URI dans Google Cloud Console (APIs & Services → Credentials)
# GOOGLE_REDIRECT_URI=https://<ton-app>.railway.app/auth/google/callback

# Optionnel
MAX_UPLOAD_SIZE=524288000
MODEL_CACHE_DIR=/tmp/model_cache
```

**📖 Guide complet**: [`docs/VARIABLES_ENVIRONNEMENT.md`](docs/VARIABLES_ENVIRONNEMENT.md) explique:
- Différences entre développement et production
- Pourquoi GOOGLE_REDIRECT_URI doit être mis à jour en 2 étapes
- Comment configurer Google Cloud Console
- Troubleshooting des erreurs courantes

## 📦 Structure du Projet

```
fraud-project/
├── APP_autoML/              # Application Flask (à déployer)
│   ├── app/                 # Code application
│   ├── config/              # Configuration
│   ├── migrations/          # Migrations DB
│   ├── Dockerfile          # ✅ Image Docker
│   ├── requirements.txt    # ✅ Dépendances
│   ├── run.py              # ✅ Point d'entrée
│   └── Procfile            # ✅ Commande démarrage
├── automl_transformer/      # Copié dans Docker ✅
│   ├── full_automl.py
│   └── ...
├── data/
│   ├── automl_models/       # ⚠️ Pas dans Docker (sur S3)
│   └── metatransformer_training/  # ✅ Copié dans Docker
│       └── automl_meta_transformer_best.pth
```

## 🚀 Commandes de Déploiement

### Option 1: Via Railway CLI

```bash
# Installer Railway CLI
npm i -g @railway/cli

# Se connecter
railway login

# Créer nouveau projet
railway init

# Ajouter PostgreSQL
railway add --database postgresql

# Déployer
cd APP_autoML
railway up
```

### Option 2: Via GitHub (Recommandé)

1. **Créer repo GitHub** (si pas déjà fait)
2. **Connecter Railway à GitHub**
3. **Railway détecte automatiquement** `Dockerfile` et `railway.json`
4. **Configuration auto** avec les variables d'environnement
5. **Déploiement auto** à chaque push

## ⚠️ Points d'Attention

### 1. Chemins Relatifs ✅ RÉSOLU
- Le Dockerfile copie `automl_transformer` et `data` au bon endroit
- PYTHONPATH configuré pour importer correctement

### 2. Google OAuth Redirect URI ⚠️ À METTRE À JOUR
```python
# Dans Google Cloud Console, ajouter:
https://<ton-app>.railway.app/auth/google/callback
```

### 3. Migrations Base de Données
```bash
# Après premier déploiement, exécuter:
railway run flask db upgrade

# Ou dans Railway shell:
flask db upgrade
```

### 4. Populate Reference Models
```bash
# Si besoin de re-populer les modèles de référence:
railway run python populate_reference_models.py
```

## 🔍 Vérifications Post-Déploiement

### 1. Santé de l'Application
```bash
curl https://<ton-app>.railway.app/health
```

### 2. Connexion Base de Données
```bash
# Vérifier les logs Railway
railway logs
```

### 3. Accès S3
```bash
# Tester téléchargement modèle depuis S3
# Logs doivent montrer: "✅ Model downloaded from S3"
```

### 4. Test Complet
1. S'inscrire avec email
2. Se connecter avec Google OAuth
3. Upload dataset
4. Lancer training AutoML
5. Faire prédictions

## 📊 Monitoring

### Logs Railway
```bash
railway logs --tail
```

### Métriques
- CPU usage
- Memory usage  
- Request latency
- Database connections

## 🆘 Troubleshooting

### Erreur: Module not found 'automl_transformer'
**Solution**: Vérifier que PYTHONPATH inclut `/` et que le dossier est copié

### Erreur: FileNotFoundError metatransformer
**Solution**: Vérifier que `COPY ../data/metatransformer_training` s'exécute correctement

### Erreur: S3 Access Denied
**Solution**: Vérifier les credentials AWS dans Railway env vars

### Erreur: Database connection
**Solution**: Vérifier que DATABASE_URL est bien injectée par Railway

## ✅ Checklist Finale Avant Déploiement

- [ ] `.env` n'est PAS commité dans Git
- [ ] Toutes les variables d'env sont dans Railway Dashboard
- [ ] Google OAuth redirect URI mis à jour avec domaine Railway
- [ ] SECRET_KEY production généré (pas dev-secret-key)
- [ ] requirements.txt à jour
- [ ] Dockerfile testé localement (`docker build -t test .`)
- [ ] `.dockerignore` exclut fichiers sensibles
- [ ] Migrations DB prêtes (`flask db migrate`)

## 🎉 Déploiement Final

Une fois tout vérifié:

```bash
git add .
git commit -m "Production ready - Railway deployment"
git push origin main
```

Railway va automatiquement:
1. Détecter le push
2. Build l'image Docker
3. Appliquer les variables d'environnement
4. Déployer l'application
5. Fournir une URL publique

**URL finale**: `https://<ton-app>.railway.app`
