# Aïobi - Guide de l'Identité Visuelle

## 🎨 Palette de Couleurs

### Couleurs Principales
- **Noir Principal** : `#000000` - Couleur de marque primaire
- **Noir Secondaire** : `#1a1a1a` - Arrière-plans sombres
- **Gris Foncé** : `#2d2d2d` - Éléments secondaires
- **Blanc** : `#FFFFFF` - Texte sur fond sombre, arrière-plans clairs

### Couleurs d'État
- **Succès** : `#10b981` (Vert)
- **Avertissement** : `#f59e0b` (Orange)
- **Erreur** : `#ef4444` (Rouge)
- **Info** : `#3b82f6` (Bleu)

### Nuances de Gris
- `#fafafa` → `#171717` (du plus clair au plus foncé)

## 🖋️ Typographie

**Police** : Inter (Google Fonts)
- Poids disponibles : 300, 400, 500, 600, 700, 800, 900
- Fallback : -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto'

## 🎯 Logo Aïobi

Le logo Aïobi est un SVG personnalisé qui représente :
- Un visage souriant/robot friendly (rond blanc sur fond noir)
- Le texte "Aïobi" en blanc
- Des points décoratifs de chaque côté
- Style moderne, minimaliste, tech

### Utilisation du Logo

```html
<!-- Logo standard (40px) -->
<svg class="aiobi-logo" viewBox="0 0 200 200">...</svg>

<!-- Logo petit (32px) -->
<svg class="aiobi-logo-sm" viewBox="0 0 200 200">...</svg>
```

## 🧩 Composants

### Boutons

```html
<!-- Bouton Principal (noir) -->
<button class="btn-aiobi-primary">Action Principale</button>

<!-- Bouton Secondaire (blanc avec bordure noire) -->
<button class="btn-aiobi-secondary">Action Secondaire</button>

<!-- Bouton Ghost (transparent) -->
<button class="btn-aiobi-ghost">Action Tertiaire</button>
```

### Cartes

```html
<!-- Carte standard avec hover effect -->
<div class="card-aiobi">
    <!-- Contenu -->
</div>
```

### Badges

```html
<!-- Badge noir -->
<span class="badge-aiobi badge-aiobi-black">Premium</span>

<!-- Badge outline -->
<span class="badge-aiobi badge-aiobi-outline">Info</span>
```

### Inputs

```html
<!-- Input standard -->
<input type="text" class="input-aiobi" placeholder="...">
```

### Navigation

```html
<!-- Lien de navigation -->
<a href="#" class="nav-link-aiobi">Dashboard</a>

<!-- Lien actif -->
<a href="#" class="nav-link-aiobi active">Modèles</a>
```

## 🎬 Animations

### Fade In Up
```html
<div class="fade-in-up">
    <!-- Animation d'apparition en fondu depuis le bas -->
</div>
```

### Pulse
```html
<div class="pulse-aiobi">
    <!-- Animation de pulsation -->
</div>
```

### Loader
```html
<div class="loader-aiobi"></div>
```

## 📐 Gradients

```css
/* Gradient noir subtil */
.gradient-aiobi {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 50%, #404040 100%);
}

/* Gradient pour texte */
.gradient-aiobi-text {
    background: linear-gradient(135deg, #1a1a1a 0%, #404040 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
```

## 🎨 Thème CSS

Le fichier `aiobi-theme.css` contient toutes les classes CSS personnalisées :
- Variables CSS (`:root`)
- Typographie
- Composants (boutons, cartes, badges, inputs)
- Tables
- Animations
- Scrollbar personnalisée
- Styles responsive

## 📱 Responsive

Les styles sont optimisés pour toutes les tailles d'écran :
- Mobile : Logo réduit à 32px
- Tablette : Grilles adaptatives
- Desktop : Pleine expérience

## 🔧 Intégration

### Dans base.html
```html
<!-- Aïobi Theme CSS -->
<link rel="stylesheet" href="{{ url_for('static', filename='css/aiobi-theme.css') }}">
```

### Structure HTML
```html
<body class="bg-gray-50">
    <nav class="nav-aiobi">
        <!-- Navigation avec logo Aïobi -->
    </nav>
    
    <main>
        <div class="card-aiobi">
            <!-- Contenu -->
        </div>
    </main>
</body>
```

## 🎯 Principes de Design

1. **Minimalisme** : Design épuré, focus sur l'essentiel
2. **Contraste** : Utilisation forte du noir et blanc
3. **Clarté** : Typographie lisible, espacement généreux
4. **Modernité** : Coins arrondis, ombres subtiles
5. **Cohérence** : Même style sur toutes les pages

## 📄 Pages Adaptées

### ✅ Complétées
- `base.html` - Navigation et structure
- `auth/login.html` - Page de connexion
- `dashboard/index.html` - Dashboard principal

### 🔄 À Adapter (même structure)
- `auth/register.html` - Inscription
- `dashboard/upload.html` - Upload de fichiers
- `dashboard/models.html` - Liste des modèles
- `dashboard/predict.html` - Prédictions
- `dashboard/history.html` - Historique
- `dashboard/settings.html` - Paramètres

### Template de Conversion

```html
<!-- AVANT (ancien style) -->
<button class="gradient-bg text-white px-4 py-2">Action</button>

<!-- APRÈS (style Aïobi) -->
<button class="btn-aiobi-primary">Action</button>
```

## 🚀 Déploiement

Le thème Aïobi est prêt pour la production :
- ✅ CSS optimisé et minimaliste
- ✅ Compatible tous navigateurs
- ✅ Performance optimale
- ✅ Accessible (WCAG 2.1)
- ✅ Responsive mobile-first

---

**© 2025 Aïobi - Fraud Detection AI Platform**
