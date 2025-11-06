# 📚 Index de la Documentation

## 🚀 Déploiement

| Document | Description | Pour qui ? |
|----------|-------------|------------|
| [DEPLOIEMENT_RAPIDE.md](../DEPLOIEMENT_RAPIDE.md) | Guide condensé étape par étape | 🟢 Débutant |
| [RAILWAY_DEPLOYMENT.md](../RAILWAY_DEPLOYMENT.md) | Documentation technique complète | 🟡 Avancé |
| [VARIABLES_ENVIRONNEMENT.md](VARIABLES_ENVIRONNEMENT.md) | Guide détaillé des variables .env | 🟢 Tous |

## 🔐 Configuration

| Document | Description | Quand l'utiliser ? |
|----------|-------------|-------------------|
| [GOOGLE_REDIRECT_URI_AIDE_MEMOIRE.md](GOOGLE_REDIRECT_URI_AIDE_MEMOIRE.md) | Explique les différences dev/prod | Avant déploiement |
| `.env.example` | Template de configuration | Nouvelle installation |

## 🛠️ Scripts Utiles

| Script | Commande | Description |
|--------|----------|-------------|
| `view_env_config.py` | `python view_env_config.py` | Vérifie la configuration actuelle |
| `generate_secret_key.py` | `python generate_secret_key.py` | Génère une SECRET_KEY sécurisée |
| `test_docker_build.bat` | `test_docker_build.bat` | Teste le build Docker localement |

## 📖 Workflow Recommandé

### 1️⃣ Installation Locale
```bash
# Suivre: DEPLOIEMENT_RAPIDE.md → Section "Test Local"
cd APP_autoML
cp .env.example .env
# Éditer .env avec vos credentials
python view_env_config.py  # Vérifier la config
python run.py
```

### 2️⃣ Préparation Déploiement
```bash
# Suivre: DEPLOIEMENT_RAPIDE.md → Section "Checklist Avant Déploiement"
python generate_secret_key.py  # Noter la clé générée
cd ..
test_docker_build.bat  # Tester le build
```

### 3️⃣ Déploiement Railway
```bash
# Suivre: RAILWAY_DEPLOYMENT.md → Section "Commandes de Déploiement"
git add .
git commit -m "Ready for Railway"
git push origin main

# Configurer Railway Dashboard
# Voir: VARIABLES_ENVIRONNEMENT.md
```

### 4️⃣ Configuration Google OAuth
```bash
# Suivre: GOOGLE_REDIRECT_URI_AIDE_MEMOIRE.md
# 1. Noter l'URL Railway
# 2. Mettre à jour GOOGLE_REDIRECT_URI
# 3. Ajouter dans Google Cloud Console
```

## 🆘 Troubleshooting

| Problème | Document à consulter | Section |
|----------|---------------------|---------|
| Erreur "redirect_uri_mismatch" | [GOOGLE_REDIRECT_URI_AIDE_MEMOIRE.md](GOOGLE_REDIRECT_URI_AIDE_MEMOIRE.md) | Erreurs Courantes |
| Variables d'environnement | [VARIABLES_ENVIRONNEMENT.md](VARIABLES_ENVIRONNEMENT.md) | Troubleshooting |
| Docker build échoue | [DEPLOIEMENT_RAPIDE.md](../DEPLOIEMENT_RAPIDE.md) | Troubleshooting |
| Connexion S3 échoue | [VARIABLES_ENVIRONNEMENT.md](VARIABLES_ENVIRONNEMENT.md) | Troubleshooting |

## 🎯 Quick Links

- [📦 Structure du Projet](../RAILWAY_DEPLOYMENT.md#-structure-du-projet)
- [🔧 Variables Critiques](VARIABLES_ENVIRONNEMENT.md#-variables-denvironnement-complètes)
- [✅ Checklist Déploiement](../DEPLOIEMENT_RAPIDE.md#-checklist-avant-déploiement)
- [🔍 Vérifier Config](../RAILWAY_DEPLOYMENT.md#-vérifications-post-déploiement)

## 📱 Contacts & Support

- **GitHub Issues**: [Créer un ticket](https://github.com/votre-repo/issues)
- **Email**: support@aiobi.com
- **Documentation**: [docs.aiobi.com](https://docs.aiobi.com)

---

**Dernière mise à jour**: 2025-11-06
