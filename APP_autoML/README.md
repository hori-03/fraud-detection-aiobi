# 🚀 AML AutoML Application

Application web Flask pour la détection de fraude et le blanchiment d'argent (AML) utilisant un métamodèle XGBoost.

## 📋 Fonctionnalités

- 🔐 **Authentification** : Login interne + OAuth2 Google
- 📊 **Dashboard utilisateur** : Interface moderne avec Tailwind CSS
- 🤖 **AutoML** : Génération, entraînement et comparaison automatique de modèles XGBoost
- 💾 **Gestion des modèles** : Sauvegarde et historique des modèles entraînés
- 📈 **Visualisations** : Métriques et résultats en temps réel
- 🎫 **Licences** : Système de gestion des licences utilisateurs

## 🛠️ Stack Technique

- **Backend** : Flask 3.0 + SQLAlchemy
- **Frontend** : Tailwind CSS + JavaScript vanilla
- **Base de données** : PostgreSQL
- **ML** : XGBoost + Scikit-learn
- **Déploiement** : Docker + Railway
- **Auth** : Flask-Login + Google OAuth2

## 📁 Structure du Projet

```
APP_autoML/
├── app/
│   ├── __init__.py           # Application factory
│   ├── models/               # Modèles SQLAlchemy (User, License, History)
│   ├── routes/               # Routes Flask (auth, dashboard, api)
│   ├── services/             # Services métier (automl_service, model_service)
│   ├── static/
│   │   ├── css/             # Tailwind CSS compilé
│   │   └── js/              # Scripts JavaScript
│   └── templates/            # Templates Jinja2
├── config/
│   └── config.py            # Configuration Flask
├── models/
│   └── xgboost_models/      # Modèles entraînés sauvegardés
├── uploads/                  # Datasets uploadés
├── logs/                     # Logs application
├── tests/                    # Tests unitaires
├── run.py                    # Point d'entrée application
├── requirements.txt          # Dépendances Python
├── Dockerfile               # Configuration Docker
├── Procfile                 # Configuration Railway/Heroku
├── railway.json             # Configuration Railway
└── .env.example             # Template variables d'environnement
```

## 🚀 Installation Locale

### Prérequis

- Python 3.11+
- PostgreSQL 14+
- Git

### Étapes

1. **Cloner le dépôt**
```bash
cd APP_autoML
```

2. **Créer environnement virtuel**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Installer dépendances**
```bash
pip install -r requirements.txt
```

4. **Configuration**
```bash
# Copier le template
copy .env.example .env

# Éditer .env avec vos valeurs
# - SECRET_KEY
# - DATABASE_URL (PostgreSQL local)
# - GOOGLE_CLIENT_ID (si OAuth)
# - GOOGLE_CLIENT_SECRET (si OAuth)
```

5. **Initialiser la base de données**
```bash
python
>>> from app import create_app, db
>>> app = create_app()
>>> with app.app_context():
...     db.create_all()
>>> exit()
```

6. **Lancer l'application**
```bash
python run.py
```

Application disponible sur `http://localhost:5000`

## 🐳 Déploiement Railway

### Prérequis Railway

1. Compte Railway ([railway.app](https://railway.app))
2. Projet Railway créé
3. PostgreSQL ajouté dans Railway

### Étapes

1. **Connecter le dépôt GitHub**
   - Pusher le code sur GitHub
   - Connecter Railway à votre repo

2. **Ajouter PostgreSQL**
   - Dans Railway : Add Service → Database → PostgreSQL
   - Railway génère automatiquement `DATABASE_URL`

3. **Variables d'environnement**
   ```
   SECRET_KEY=votre-clé-secrète
   FLASK_ENV=production
   GOOGLE_CLIENT_ID=votre-client-id
   GOOGLE_CLIENT_SECRET=votre-client-secret
   GOOGLE_REDIRECT_URI=https://votre-app.railway.app/auth/callback
   ```

4. **Déploiement automatique**
   - Railway détecte le `Dockerfile`
   - Build et déploiement automatiques
   - URL générée : `https://votre-app.railway.app`

## 📊 Utilisation

### Dashboard AutoML

1. **Uploader un dataset**
   - Format : CSV ou JSON
   - Colonnes : features + target (is_fraud, fraud_flag, etc.)

2. **Lancer l'entraînement**
   - Le métamodèle génère plusieurs configurations XGBoost
   - Entraîne et compare automatiquement
   - Sélectionne le meilleur modèle

3. **Visualiser les résultats**
   - Métriques : Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Temps d'entraînement
   - Feature importance

4. **Sauvegarder le modèle**
   - Modèle enregistré en `.joblib`
   - Historique persistant en DB

## 🧪 Tests

```bash
# Lancer tous les tests
pytest

# Avec couverture
pytest --cov=app tests/
```

## 📝 API Endpoints

### Authentification
- `POST /auth/login` - Login
- `POST /auth/register` - Inscription
- `GET /auth/google` - OAuth Google
- `GET /auth/callback` - Callback OAuth
- `GET /auth/logout` - Déconnexion

### Dashboard
- `GET /dashboard` - Page principale
- `POST /dashboard/upload` - Upload dataset
- `POST /dashboard/train` - Lancer entraînement
- `GET /dashboard/models` - Liste des modèles
- `GET /dashboard/history` - Historique

### API
- `GET /api/models` - Liste modèles (JSON)
- `GET /api/model/<id>` - Détails modèle
- `POST /api/predict` - Prédiction

## 🔒 Sécurité

- ✅ Mots de passe hashés (werkzeug.security)
- ✅ Protection CSRF (Flask-WTF)
- ✅ Variables d'environnement sécurisées
- ✅ Validation des uploads
- ✅ Timeout requests
- ✅ HTTPS en production (Railway)

## 🤝 Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Pull Request

## 📄 Licence

MIT License - Voir `LICENSE` pour plus de détails

## 🐛 Bugs & Support

Ouvrir une issue sur GitHub avec :
- Description du problème
- Étapes de reproduction
- Logs pertinents

## 📚 Documentation Complète

Voir `/docs` pour :
- Architecture détaillée
- Guide développeur
- API documentation
- Schémas base de données

---

**Développé avec ❤️ pour la détection de fraude AML**
