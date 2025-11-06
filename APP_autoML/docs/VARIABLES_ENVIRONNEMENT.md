# 🔐 Guide Configuration Variables d'Environnement

## 📋 Différences Développement vs Production

### **GOOGLE_REDIRECT_URI** (⚠️ IMPORTANT)

#### Développement Local:
```bash
GOOGLE_REDIRECT_URI=http://127.0.0.1:5000/auth/google/callback
```

#### Production Railway:
```bash
GOOGLE_REDIRECT_URI=https://<ton-app>.railway.app/auth/google/callback
```

**Action requise:**
1. Déployer sur Railway pour obtenir l'URL (ex: `https://fraud-detection-production.railway.app`)
2. Mettre à jour `GOOGLE_REDIRECT_URI` dans Railway → Variables
3. Ajouter cette URI dans **Google Cloud Console**:
   - Aller sur https://console.cloud.google.com/apis/credentials
   - Cliquer sur ton OAuth 2.0 Client ID
   - Section "Authorized redirect URIs"
   - Ajouter: `https://<ton-app>.railway.app/auth/google/callback`
   - Sauvegarder

---

## 🔑 Variables d'Environnement Complètes

### **1. FLASK**

| Variable | Développement | Production |
|----------|---------------|------------|
| `FLASK_ENV` | `development` | `production` |
| `FLASK_DEBUG` | `1` | `0` |
| `SECRET_KEY` | `dev-secret-key-change-in-production` | Générer avec `python generate_secret_key.py` |

⚠️ **CRITIQUE**: Ne JAMAIS utiliser la même SECRET_KEY en dev et prod!

---

### **2. BASE DE DONNÉES**

#### Développement (Connexion depuis ton PC):
```bash
# URL publique Railway avec driver psycopg v3
DATABASE_URL=postgresql+psycopg://postgres:rWrQsGaGlBUqQLtXFUVRMRgBrudpIPJX@switchyard.proxy.rlwy.net:45478/railway
```

#### Production Railway:
```bash
# Railway injecte automatiquement DATABASE_URL
# PAS BESOIN de la mettre manuellement dans les variables!
# Railway utilise l'URL interne: postgresql://postgres:...@postgres.railway.internal:5432/railway
```

**Note**: Le fichier `config.py` contient un fix automatique qui convertit `postgres://` en `postgresql://` si nécessaire.

---

### **3. AWS S3** (Identique dev et prod)

```bash
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID_HERE
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY_HERE
AWS_DEFAULT_REGION=eu-north-1
S3_MODEL_BUCKET=fraud-detection-ml-models
STORAGE_TYPE=s3
```

✅ Ces credentials sont les mêmes en développement et production.

---

### **4. GOOGLE OAUTH**

```bash
GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID_HERE
GOOGLE_CLIENT_SECRET=YOUR_GOOGLE_CLIENT_SECRET_HERE
```

✅ Client ID et Secret sont identiques.

#### GOOGLE_REDIRECT_URI:

**Développement**:
```bash
GOOGLE_REDIRECT_URI=http://127.0.0.1:5000/auth/google/callback
```

**Production**:
```bash
GOOGLE_REDIRECT_URI=https://fraud-detection-production.railway.app/auth/google/callback
```

⚠️ Remplacer `fraud-detection-production` par le nom de ton app Railway.

---

## 🚀 Étapes de Configuration Railway

### Étape 1: Créer le Projet Railway
1. Aller sur https://railway.app/
2. New Project → Deploy from GitHub
3. Sélectionner le repo `fraud-project`

### Étape 2: Ajouter PostgreSQL
1. Dans le projet Railway → New → Database → PostgreSQL
2. Railway va automatiquement:
   - Créer la base de données
   - Injecter `DATABASE_URL` dans l'environnement
   - Pas besoin de la copier manuellement!

### Étape 3: Configurer les Variables
1. Cliquer sur ton service web (pas la DB)
2. Onglet "Variables"
3. Ajouter chaque variable:

```bash
# Flask
SECRET_KEY=<généré-avec-generate_secret_key.py>
FLASK_ENV=production
FLASK_DEBUG=0

# AWS S3
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID_HERE
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY_HERE
AWS_DEFAULT_REGION=eu-north-1
S3_MODEL_BUCKET=fraud-detection-ml-models
STORAGE_TYPE=s3

# Google OAuth (⚠️ METTRE À JOUR après déploiement)
GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID_HERE
GOOGLE_CLIENT_SECRET=YOUR_GOOGLE_CLIENT_SECRET_HERE
GOOGLE_REDIRECT_URI=https://<TON-APP>.railway.app/auth/google/callback
```

### Étape 4: Mettre à jour GOOGLE_REDIRECT_URI

Après le premier déploiement, Railway va te donner une URL type:
```
https://fraud-detection-production.railway.app
```

Alors:

1. **Dans Railway**:
   - Variables → Edit `GOOGLE_REDIRECT_URI`
   - Remplacer par: `https://fraud-detection-production.railway.app/auth/google/callback`
   - Sauvegarder (redéploie automatiquement)

2. **Dans Google Cloud Console**:
   - https://console.cloud.google.com/apis/credentials
   - Modifier ton OAuth Client ID
   - "Authorized redirect URIs" → Add URI
   - Ajouter: `https://fraud-detection-production.railway.app/auth/google/callback`
   - Save

---

## 🧪 Tester en Local avec .env Production

Si tu veux tester la config production en local:

1. **Copier .env vers .env.production**:
```cmd
copy .env .env.production
```

2. **Modifier .env.production**:
```bash
FLASK_ENV=production
FLASK_DEBUG=0
SECRET_KEY=<nouvelle-clé-générée>
GOOGLE_REDIRECT_URI=http://127.0.0.1:5000/auth/google/callback  # Garder localhost pour test local
```

3. **Lancer avec .env.production**:
```cmd
# Renommer temporairement
ren .env .env.dev
ren .env.production .env

# Lancer
python run.py

# Remettre
ren .env .env.production
ren .env.dev .env
```

---

## ✅ Checklist Variables

Avant de déployer:

- [ ] `SECRET_KEY` générée avec `generate_secret_key.py`
- [ ] `FLASK_ENV=production` et `FLASK_DEBUG=0`
- [ ] Variables AWS S3 configurées
- [ ] `GOOGLE_CLIENT_ID` et `GOOGLE_CLIENT_SECRET` corrects
- [ ] `GOOGLE_REDIRECT_URI` temporairement à `http://127.0.0.1:5000/auth/google/callback`

Après premier déploiement:

- [ ] Noter l'URL Railway (ex: `https://mon-app.railway.app`)
- [ ] Mettre à jour `GOOGLE_REDIRECT_URI` dans Railway
- [ ] Ajouter l'URI dans Google Cloud Console
- [ ] Tester le login Google

---

## 🆘 Troubleshooting

### Erreur: "redirect_uri_mismatch" Google OAuth

**Cause**: `GOOGLE_REDIRECT_URI` ne correspond pas à celle dans Google Cloud Console

**Solution**:
1. Vérifier l'URL exacte dans Railway
2. S'assurer qu'elle se termine par `/auth/google/callback`
3. Vérifier qu'elle est bien ajoutée dans Google Cloud Console

### Erreur: "SECRET_KEY not configured"

**Cause**: Variable `SECRET_KEY` manquante dans Railway

**Solution**:
1. Générer une clé: `python generate_secret_key.py`
2. Ajouter dans Railway → Variables

### Erreur: "Database connection failed"

**Cause**: Plugin PostgreSQL pas ajouté

**Solution**:
1. Railway → New → Database → PostgreSQL
2. Railway injecte automatiquement `DATABASE_URL`
3. Redéployer si nécessaire

---

## 📝 Résumé

| Variable | Dev | Prod | Même valeur? |
|----------|-----|------|--------------|
| `SECRET_KEY` | `dev-secret-key...` | Généré | ❌ Différent |
| `DATABASE_URL` | URL publique Railway | Auto-injecté | ❌ Différent |
| `GOOGLE_REDIRECT_URI` | `http://127.0.0.1:5000/...` | `https://<app>.railway.app/...` | ❌ Différent |
| `AWS_ACCESS_KEY_ID` | `AKIA5ROAPP4...` | `AKIA5ROAPP4...` | ✅ Identique |
| `AWS_SECRET_ACCESS_KEY` | `1JorGcwmqzz...` | `1JorGcwmqzz...` | ✅ Identique |
| `GOOGLE_CLIENT_ID` | `277664819609...` | `277664819609...` | ✅ Identique |
| `GOOGLE_CLIENT_SECRET` | `GOCSPX-QIRk...` | `GOCSPX-QIRk...` | ✅ Identique |
