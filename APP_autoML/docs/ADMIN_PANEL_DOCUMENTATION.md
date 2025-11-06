# Panneau d'Administration - Documentation

## 📋 Vue d'ensemble

Le panneau d'administration permet aux utilisateurs avec le rôle `is_admin=True` de gérer l'ensemble de la plateforme : utilisateurs, licences, et statistiques.

## 🔐 Accès

**URL** : `/admin`

**Restriction** : Seuls les utilisateurs avec `is_admin=True` peuvent accéder au panneau admin.

Le lien "Admin" apparaît automatiquement dans la barre de navigation pour les administrateurs.

## ✨ Fonctionnalités

### 1. Dashboard Admin (`/admin`)
- **Statistiques rapides** :
  - Total utilisateurs (actifs/inactifs)
  - Total licences (actives/expirées)
  - Nouveaux inscrits (7 derniers jours)
  - Licences expirées nécessitant action
  
- **Répartition des licences** par type (Trial, Basic, Premium, Enterprise)
  
- **Accès rapides** vers :
  - Gestion utilisateurs
  - Gestion licences
  - Statistiques détaillées

### 2. Gestion des Utilisateurs (`/admin/users`)
- **Liste paginée** (20 utilisateurs par page)
- **Filtres** :
  - Recherche par nom, email, username
  - Statut (actif/inactif)
  
- **Informations affichées** :
  - Avatar, nom d'utilisateur, badge ADMIN
  - Type de connexion (Google OAuth ou Email)
  - Date d'inscription
  - Statut actif/inactif

### 3. Détails Utilisateur (`/admin/users/<user_id>`)
- **Informations complètes** :
  - Avatar et identité
  - Statut du compte
  - Type de connexion (Google/Email)
  - Date d'inscription
  
- **Licences de l'utilisateur** :
  - Liste de toutes les licences
  - Type, clé, expiration, limites
  - Actions : Activer/Désactiver, Prolonger
  - Bouton pour créer une nouvelle licence
  
- **Historique d'entraînement** :
  - 10 derniers entraînements
  - Date, nom du modèle, statut
  
- **Actions administratives** :
  - Activer/Désactiver le compte
  - Promouvoir/Révoquer rôle admin
  - Supprimer l'utilisateur (avec confirmation)
  
- **Protections** :
  - Impossible de se désactiver soi-même
  - Impossible de se retirer les droits admin
  - Impossible de se supprimer soi-même

### 4. Gestion des Licences (`/admin/licenses`)
- **Liste paginée** (20 licences par page)
- **Filtres** :
  - Type de licence (Trial, Basic, Premium, Enterprise)
  - Statut (Active/Expirée)
  
- **Informations affichées** :
  - Utilisateur (avec lien vers profil)
  - Type de licence (badge coloré)
  - Clé de licence (code)
  - Date d'expiration
  - Limites (modèles, prédictions)
  - Statut actif/expiré
  
- **Actions rapides** :
  - Activer/Désactiver
  - Prolonger (modal avec saisie jours)

### 5. Création de Licence (`/admin/licenses/create`)
- **Formulaire** :
  - Sélection utilisateur
  - Type de licence
  - Durée en jours (1-3650)
  
- **Limites automatiques** par type :
  - **Trial** : 3 modèles, 1000 prédictions
  - **Basic** : 10 modèles, 10000 prédictions
  - **Premium** : 50 modèles, 100000 prédictions
  - **Enterprise** : Illimité
  
- **Affichage informatif** des limites selon le type sélectionné
- **Clé générée** affichée après création avec bouton copier

### 6. Statistiques (`/admin/stats`)
- **Statistiques globales** :
  - Total utilisateurs
  - Licences actives
  - Licences expirées
  - Nouveaux utilisateurs (7 jours)
  
- **Graphique d'inscriptions** (30 derniers jours)
  - Chart.js - Ligne avec remplissage
  
- **Graphique de répartition** des licences
  - Chart.js - Donut avec couleurs par type
  
- **Tableau détaillé** par type de licence :
  - Nombre total
  - Actives vs Expirées
  - Pourcentage avec barre de progression

## 🎨 Design

- **Style cohérent** avec le reste de l'application (Tailwind CSS)
- **Thème sombre** Aïobi
- **Animations** smooth sur hover
- **Icônes** Font Awesome
- **Responsive** : Desktop et mobile
- **Badges colorés** pour différencier les types/statuts

## 🔒 Sécurité

- **Décorateur `@admin_required`** : Vérifie `is_authenticated` ET `is_admin`
- **Redirection automatique** pour les non-admins vers dashboard avec message d'erreur
- **Protections auto-action** : Empêche admin de se nuire à lui-même
- **Confirmation JavaScript** sur suppression utilisateur

## 🛠️ Backend

**Fichier** : `app/routes/admin.py`

**Routes** :
- `GET /admin` - Dashboard
- `GET /admin/users` - Liste utilisateurs (avec filtres)
- `GET /admin/users/<id>` - Détails utilisateur
- `POST /admin/users/<id>/toggle-status` - Activer/Désactiver
- `POST /admin/users/<id>/toggle-admin` - Promouvoir/Révoquer admin
- `POST /admin/users/<id>/delete` - Supprimer utilisateur
- `GET /admin/licenses` - Liste licences (avec filtres)
- `POST /admin/licenses/<id>/toggle-status` - Activer/Désactiver licence
- `POST /admin/licenses/<id>/extend` - Prolonger licence
- `GET/POST /admin/licenses/create` - Créer licence
- `GET /admin/stats` - Statistiques détaillées

**Imports nécessaires** :
```python
from functools import wraps
from datetime import datetime, timedelta
import secrets
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from sqlalchemy import func
from app import db
from app.models.user import User
from app.models.license import License
from app.models.training import TrainingHistory
```

## 🧪 Test

1. **Créer un admin** :
   ```sql
   UPDATE users SET is_admin = true WHERE email = 'votre@email.com';
   ```

2. **Accéder au panneau** : Connectez-vous et cliquez sur "Admin" dans la navigation

3. **Tester les fonctionnalités** :
   - ✅ Voir les statistiques
   - ✅ Lister les utilisateurs avec filtres
   - ✅ Voir détails utilisateur
   - ✅ Activer/Désactiver compte
   - ✅ Promouvoir utilisateur en admin
   - ✅ Lister licences avec filtres
   - ✅ Créer nouvelle licence
   - ✅ Prolonger licence
   - ✅ Voir graphiques statistiques

## 📱 Navigation

Le lien "Admin" apparaît dans la barre de navigation **uniquement** pour les administrateurs :

```html
{% if current_user.is_admin %}
<a href="{{ url_for('admin.index') }}" class="nav-link-aiobi">
    <i class="fas fa-shield-alt mr-2"></i> Admin
</a>
{% endif %}
```

## 🚀 Déploiement

**Fichiers créés/modifiés** :
- ✅ `app/routes/admin.py` (nouveau)
- ✅ `app/__init__.py` (modifié - blueprint enregistré)
- ✅ `app/templates/base.html` (modifié - lien admin ajouté)
- ✅ `app/templates/admin/index.html` (nouveau)
- ✅ `app/templates/admin/users.html` (nouveau)
- ✅ `app/templates/admin/user_detail.html` (nouveau)
- ✅ `app/templates/admin/licenses.html` (nouveau)
- ✅ `app/templates/admin/create_license.html` (nouveau)
- ✅ `app/templates/admin/stats.html` (nouveau)

**Aucune migration nécessaire** - Utilise les modèles existants (User, License, TrainingHistory)

## 💡 Notes

- Les erreurs de linting sur les templates sont normales (Jinja + JavaScript)
- Chart.js chargé via CDN pour les graphiques
- Pagination à 20 éléments par page
- Protection CSRF automatique via Flask-WTF
- Messages flash pour feedback utilisateur
