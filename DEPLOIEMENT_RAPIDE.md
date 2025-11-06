# 🚀 Guide Rapide de Déploiement

## Configuration Railway

Railway buildra depuis **la racine du projet** (`fraud-project/`) grâce au fichier `railway.toml`.

### Structure attendue:
```
fraud-project/              ← Contexte de build Railway
├── railway.toml           ← Configure le build depuis la racine
├── .dockerignore          ← Exclut les fichiers inutiles
├── APP_autoML/
│   ├── Dockerfile         ← Image Docker
│   ├── requirements.txt
│   ├── run.py
│   └── ...
├── automl_transformer/    ← Copié dans l'image
└── data/
    └── metatransformer_training/
        └── *.pth          ← Copié dans l'image
```

## 🧪 Test Local (IMPORTANT!)

Avant de déployer sur Railway, teste le build:

```bash
# Windows
test_docker_build.bat

# Linux/Mac
chmod +x test_docker_build.sh
./test_docker_build.sh
```

Le script va:
1. ✅ Build l'image depuis la racine
2. ✅ Vérifier que `/automl_transformer/` existe
3. ✅ Vérifier que `/data/metatransformer_training/*.pth` existe
4. ✅ Optionnel: Lancer le container en local sur port 5001

## 📋 Checklist Avant Déploiement

### 1. Fichiers à la Racine
- [x] `railway.toml` → Configure le contexte de build
- [x] `.dockerignore` → Exclut fichiers inutiles
- [x] `.gitignore` → Protège `.env`

### 2. Metatransformer
```bash
# Vérifier que le fichier existe:
dir data\metatransformer_training\*.pth

# Doit afficher: automl_meta_transformer_best.pth
```

### 3. Variables d'Environnement Railway

⚠️ **Voir le guide complet**: [`APP_autoML/docs/VARIABLES_ENVIRONNEMENT.md`](APP_autoML/docs/VARIABLES_ENVIRONNEMENT.md)

**Variables critiques à configurer dans Railway Dashboard**:

```bash
# Flask (⚠️ Générer SECRET_KEY avec: python generate_secret_key.py)
SECRET_KEY=<générer-clé-sécurisée-production>
FLASK_ENV=production
FLASK_DEBUG=0

# AWS S3 (identique dev et prod)
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID_HERE
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY_HERE
AWS_DEFAULT_REGION=eu-north-1
S3_MODEL_BUCKET=fraud-detection-ml-models
STORAGE_TYPE=s3

# Google OAuth Client (identique dev et prod)
GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID_HERE
GOOGLE_CLIENT_SECRET=YOUR_GOOGLE_CLIENT_SECRET_HERE

# ⚠️ GOOGLE_REDIRECT_URI: À mettre à jour APRÈS déploiement!
# 1. Déployer d'abord avec cette valeur temporaire
# 2. Noter l'URL Railway (ex: https://mon-app.railway.app)
# 3. Mettre à jour vers: https://<ton-app>.railway.app/auth/google/callback
# 4. Ajouter cette URI dans Google Cloud Console
GOOGLE_REDIRECT_URI=http://127.0.0.1:5000/auth/google/callback

# PostgreSQL (⚠️ Auto-injectée par Railway, pas besoin de la mettre!)
# DATABASE_URL=<sera-fournie-automatiquement-par-railway>
```

**📖 Guide complet des différences Dev vs Prod**: [`VARIABLES_ENVIRONNEMENT.md`](APP_autoML/docs/VARIABLES_ENVIRONNEMENT.md)

### 4. Google Cloud Console
```
1. Aller sur: https://console.cloud.google.com/apis/credentials
2. Modifier les "Authorized redirect URIs"
3. Ajouter: https://<ton-app>.railway.app/auth/google/callback
```

## 🚂 Déploiement Railway

### Option A: Via GitHub (Recommandé)

1. **Push vers GitHub**:
```bash
git add .
git commit -m "Railway deployment ready"
git push origin main
```

2. **Railway Dashboard**:
   - New Project → Deploy from GitHub
   - Sélectionner le repo `fraud-project`
   - Railway détecte automatiquement `railway.toml` ✅
   - Ajouter plugin **PostgreSQL**
   - Configurer les variables d'environnement
   - Deploy! 🚀

### Option B: Via Railway CLI

```bash
# Installer
npm i -g @railway/cli

# Se connecter
railway login

# Depuis fraud-project/ (racine)
railway init
railway add --database postgresql

# Déployer
railway up

# Voir les logs
railway logs --tail
```

## 📊 Post-Déploiement

### 1. Migrations DB
```bash
railway run flask db upgrade
```

### 2. Populate Reference Models (si nécessaire)
```bash
railway run python populate_reference_models.py
```

### 3. Vérifications
- [ ] Accès à l'app: `https://<ton-app>.railway.app`
- [ ] Login Google OAuth fonctionne
- [ ] Upload dataset fonctionne
- [ ] Téléchargement modèle depuis S3 fonctionne
- [ ] Logs Railway sans erreur

### 4. Monitoring
```bash
# Voir logs en temps réel
railway logs --tail

# Metrics dans Dashboard
- CPU usage
- Memory usage
- Request count
- Response time
```

## 🆘 Troubleshooting

### Erreur: "No such file or directory: automl_transformer"
**Solution**: Vérifier que `railway.toml` est à la **racine** du repo

### Erreur: "Cannot import name 'full_automl'"
**Solution**: Vérifier PYTHONPATH dans Dockerfile (`ENV PYTHONPATH=/app:/`)

### Erreur: "FileNotFoundError: automl_meta_transformer_best.pth"
**Solution**: Vérifier que le fichier est bien dans `data/metatransformer_training/` et pas exclu par `.dockerignore`

### Erreur: "Access Denied" S3
**Solution**: Vérifier les variables AWS_ACCESS_KEY_ID et AWS_SECRET_ACCESS_KEY dans Railway

## ✅ C'est Prêt!

Si tous les tests locaux passent:
```bash
git add .
git commit -m "✅ Ready for Railway production deployment"
git push origin main
```

Railway va build et déployer automatiquement! 🎉
