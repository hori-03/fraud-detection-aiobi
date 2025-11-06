# 🎨 AÏOBI ANIMATIONS - Guide d'utilisation

## 📦 Animations implémentées

### ✨ Animations CSS (animations.css)

#### 1. **Animations de chargement de page**
- `.fade-in` - Fade-in avec translation
- `.slide-in-left` - Slide depuis la gauche
- `.slide-in-right` - Slide depuis la droite
- `.delay-100` à `.delay-500` - Délais d'animation échelonnés

#### 2. **Effets de survol sur les cartes**
- `.card-hover` - Flottement au survol
- `.card-3d` - Effet 3D au survol
- `.shine-effect` - Brillance qui traverse au survol

#### 3. **Boutons animés**
- `.btn-ripple` - Effet ripple au clic
- `.btn-pulse` - Pulsation continue
- `.btn-gradient-animated` - Gradient animé

#### 4. **Badges**
- `.badge-pulse` - Pulsation douce
- `.badge-glow` - Effet de glow

#### 5. **Progress bars**
- `.progress-glow` - Glow animé
- `.progress-fill` - Animation de remplissage

#### 6. **Skeleton loaders**
- `.skeleton` - Shimmer loading effect

#### 7. **Icônes**
- `.icon-spin` - Rotation au survol
- `.icon-bounce` - Rebond au survol
- `.icon-shake` - Tremblement au survol

#### 8. **Utilitaires**
- `.smooth-transition` - Transition fluide
- `.hover-scale` - Zoom au survol
- `.hover-glow` - Glow au survol
- `.gradient-animated` - Gradient de fond animé

### 🎯 Fonctions JavaScript (animations.js)

#### Toast Notifications
```javascript
window.AiobiAnimations.showToast(message, type, duration)
// Types: 'success', 'error', 'warning', 'info'
// Exemple: showToast('Opération réussie !', 'success', 3000)
```

#### Confetti
```javascript
window.AiobiAnimations.celebrateSuccess()
// Lance des confettis depuis le centre de l'écran
```

#### Loading Spinner
```javascript
window.AiobiAnimations.showLoadingSpinner('Chargement...')
window.AiobiAnimations.hideLoadingSpinner()
```

#### Modals
```javascript
window.AiobiAnimations.showModal('modal-id')
window.AiobiAnimations.hideModal('modal-id')
```

#### Progress Bar
```javascript
window.AiobiAnimations.animateProgressBar(element, targetPercent, duration)
```

#### Skeleton Loader
```javascript
window.AiobiAnimations.showSkeleton('container-id')
window.AiobiAnimations.hideSkeleton('container-id', content)
```

#### Smooth Scroll
```javascript
window.AiobiAnimations.smoothScrollTo('target-id')
```

## 🚀 Utilisation

### Dans les templates HTML

```html
<!-- Card avec effet de flottement -->
<div class="card-aiobi card-hover fade-in">
    <h3>Titre</h3>
    <p>Contenu</p>
</div>

<!-- Bouton avec gradient animé et brillance -->
<button class="btn-aiobi-primary btn-gradient-animated shine-effect">
    <i class="fas fa-plus icon-bounce"></i>
    Action
</button>

<!-- Badge avec pulse -->
<span class="badge-aiobi badge-pulse">
    <i class="fas fa-crown icon-spin"></i>
    PREMIUM
</span>

<!-- Icônes animées -->
<i class="fas fa-home icon-bounce"></i>
<i class="fas fa-cog icon-spin"></i>
<i class="fas fa-bell icon-shake"></i>
```

### Dans JavaScript

```javascript
// Notification de succès
window.AiobiAnimations.showToast('✅ Modèle créé !', 'success');

// Erreur
window.AiobiAnimations.showToast('❌ Erreur de connexion', 'error');

// Succès avec confetti
window.AiobiAnimations.celebrateSuccess();
window.AiobiAnimations.showToast('🎉 Entraînement terminé !', 'success');

// Loading pendant une opération
window.AiobiAnimations.showLoadingSpinner('⏳ Traitement...');
// ... opération asynchrone ...
window.AiobiAnimations.hideLoadingSpinner();
```

## 📍 Déjà implémenté

### ✅ Navbar
- Logo avec effet hover-scale
- Liens de navigation avec icon-bounce
- Badge de licence avec animations (pulse pour trial, glow pour premium/enterprise)
- Gradient animé sur le fond de la navbar

### ✅ Dashboard (index.html)
- Fade-in échelonné sur toutes les cards
- Hover effects sur les stat cards
- Bouton "Nouveau modèle" avec gradient animé
- Icônes avec animations bounce

### ✅ Page Modèles (models.html)
- Cards avec effet 3D (card-3d)
- Animations fade-in échelonnées
- Header avec shine-effect
- Icônes animées
- Métriques avec hover-glow
- Notifications avec confetti lors de la suppression

### ✅ Page Predict (predict.html)
- Fade-in sur tous les steps
- Cards avec hover effects
- Zone de drop avec shine-effect
- Confetti lors du succès des prédictions
- Loading spinner pendant le traitement
- Toast notifications pour les erreurs/succès

## 🎨 Personnalisation

### Changer les couleurs du confetti
Dans `animations.js`, ligne 13 :
```javascript
const colors = ['#4f46e5', '#7c3aed', '#ec4899', '#10b981', '#f59e0b'];
```

### Modifier la vitesse des animations
Dans `animations.css`, ajustez les `animation-duration` :
```css
.fade-in {
    animation: fadeIn 0.6s ease-out; /* Changez 0.6s */
}
```

### Désactiver une animation
Ajoutez la classe `.no-hover` ou `.no-ripple` :
```html
<div class="card-aiobi no-hover">...</div>
<button class="btn-aiobi no-ripple">...</button>
```

## 🎯 Prochaines étapes possibles

- [ ] Animations sur la page Upload
- [ ] Animations sur la page History
- [ ] Animations sur les modals
- [ ] Particules interactives au mouvement de la souris
- [ ] Animations au scroll (AOS - Animate On Scroll)
- [ ] Transitions entre pages (HTMX/Barba.js)
- [ ] Dark/Light mode avec animations
- [ ] Cursor personnalisé avec traînée

## 📝 Notes

- Toutes les animations sont **optimisées pour la performance** (GPU-accelerated)
- Les animations **respectent les préférences utilisateur** (prefers-reduced-motion)
- **Compatible** avec tous les navigateurs modernes
- **Aucune dépendance externe** (pur CSS + Vanilla JS)

## 🐛 Troubleshooting

### Les animations ne s'affichent pas
1. Vérifiez que `animations.css` est bien chargé dans `base.html`
2. Vérifiez la console pour les erreurs JavaScript
3. Assurez-vous que `animations.js` est chargé après Alpine.js

### Les confetti ne s'affichent pas
1. Vérifiez que `window.AiobiAnimations` est défini
2. Ouvrez la console et testez : `window.AiobiAnimations.celebrateSuccess()`

### Les icônes ne s'animent pas
1. Vérifiez que Font Awesome est bien chargé
2. Assurez-vous que les classes `.icon-*` sont bien appliquées

---

**Créé avec ❤️ pour Aïobi**
