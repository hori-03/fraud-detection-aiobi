/**
 * 🎨 AÏOBI ANIMATIONS JAVASCRIPT
 * Animations et interactions avancées pour une UX dynamique
 */

// ============================================
// 🎊 CONFETTI EFFECT
// ============================================

function createConfetti(x, y) {
    const colors = ['#4f46e5', '#7c3aed', '#ec4899', '#10b981', '#f59e0b'];
    const confettiCount = 50;
    
    for (let i = 0; i < confettiCount; i++) {
        const confetti = document.createElement('div');
        confetti.className = 'confetti';
        confetti.style.left = x + 'px';
        confetti.style.top = y + 'px';
        confetti.style.background = colors[Math.floor(Math.random() * colors.length)];
        confetti.style.animationDuration = (Math.random() * 2 + 2) + 's';
        confetti.style.animationDelay = (Math.random() * 0.5) + 's';
        
        document.body.appendChild(confetti);
        
        // Supprimer après l'animation
        setTimeout(() => confetti.remove(), 3000);
    }
}

// Confetti automatique sur succès
function celebrateSuccess() {
    const centerX = window.innerWidth / 2;
    const centerY = window.innerHeight / 3;
    createConfetti(centerX, centerY);
    
    // Son de succès (optionnel)
    playSuccessSound();
}

function playSuccessSound() {
    // Petit son de succès avec Web Audio API
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 800;
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.3);
}

// ============================================
// 🔔 TOAST NOTIFICATIONS
// ============================================

function showToast(message, type = 'success', duration = 3000) {
    const toast = document.createElement('div');
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    
    const colors = {
        success: 'bg-green-600',
        error: 'bg-red-600',
        warning: 'bg-yellow-600',
        info: 'bg-blue-600'
    };
    
    toast.className = `fixed top-4 right-4 ${colors[type]} text-white px-6 py-4 rounded-lg shadow-2xl toast-enter z-50 flex items-center space-x-3`;
    toast.innerHTML = `
        <span class="text-2xl">${icons[type]}</span>
        <span class="font-medium">${message}</span>
    `;
    
    document.body.appendChild(toast);
    
    // Auto-remove
    setTimeout(() => {
        toast.classList.remove('toast-enter');
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ============================================
// 💫 PAGE LOAD ANIMATIONS
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    // Fade-in pour tous les éléments principaux
    const animatedElements = document.querySelectorAll('.card, .stat-card, .model-card');
    
    animatedElements.forEach((el, index) => {
        el.classList.add('fade-in');
        el.style.animationDelay = `${index * 0.1}s`;
    });
    
    // Ajouter les effets de survol automatiquement
    addHoverEffects();
    
    // Initialiser les particules en arrière-plan (subtil)
    initParticles();
});

// ============================================
// ✨ HOVER EFFECTS AUTOMATIQUES
// ============================================

function addHoverEffects() {
    // Cards avec effet de flottement
    const cards = document.querySelectorAll('.bg-gray-800:not(.no-hover)');
    cards.forEach(card => {
        if (!card.classList.contains('card-hover')) {
            card.classList.add('card-hover', 'smooth-transition');
        }
    });
    
    // Boutons avec effet ripple
    const buttons = document.querySelectorAll('button:not(.no-ripple)');
    buttons.forEach(btn => {
        if (!btn.classList.contains('btn-ripple')) {
            btn.classList.add('btn-ripple');
        }
    });
    
    // Icônes avec animations
    const icons = document.querySelectorAll('.fa-solid, .fa-regular, .fas, .far');
    icons.forEach(icon => {
        const parent = icon.parentElement;
        if (parent && parent.tagName === 'A') {
            icon.classList.add('icon-bounce');
        }
    });
}

// ============================================
// 🎨 PARTICULES EN ARRIÈRE-PLAN
// ============================================

function initParticles() {
    const particleCount = 15; // Subtil
    const container = document.body;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        const size = Math.random() * 50 + 20;
        particle.style.width = size + 'px';
        particle.style.height = size + 'px';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animationDuration = (Math.random() * 10 + 5) + 's';
        particle.style.animationDelay = (Math.random() * 5) + 's';
        
        container.appendChild(particle);
    }
}

// ============================================
// 📊 PROGRESS BAR ANIMÉE
// ============================================

function animateProgressBar(element, targetPercent, duration = 1500) {
    const progressBar = element.querySelector('.progress-fill');
    if (!progressBar) return;
    
    progressBar.style.width = '0%';
    progressBar.classList.add('progress-glow');
    
    setTimeout(() => {
        progressBar.style.transition = `width ${duration}ms ease-out`;
        progressBar.style.width = targetPercent + '%';
    }, 100);
}

// ============================================
// 🎯 MODAL ANIMATIONS
// ============================================

function showModal(modalId) {
    const modal = document.getElementById(modalId);
    if (!modal) return;
    
    modal.classList.remove('hidden');
    const modalContent = modal.querySelector('.modal-content, [class*="bg-"]');
    if (modalContent) {
        modalContent.classList.add('modal-enter');
    }
}

function hideModal(modalId) {
    const modal = document.getElementById(modalId);
    if (!modal) return;
    
    const modalContent = modal.querySelector('.modal-content, [class*="bg-"]');
    if (modalContent) {
        modalContent.classList.remove('modal-enter');
        modalContent.classList.add('modal-exit');
        
        setTimeout(() => {
            modal.classList.add('hidden');
            modalContent.classList.remove('modal-exit');
        }, 300);
    } else {
        modal.classList.add('hidden');
    }
}

// ============================================
// 💾 SKELETON LOADER
// ============================================

function showSkeleton(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = `
        <div class="space-y-4">
            <div class="skeleton h-8 w-3/4 rounded"></div>
            <div class="skeleton h-4 w-full rounded"></div>
            <div class="skeleton h-4 w-5/6 rounded"></div>
            <div class="skeleton h-32 w-full rounded"></div>
        </div>
    `;
}

function hideSkeleton(containerId, content) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = content;
    container.classList.add('fade-in');
}

// ============================================
// 🎪 LOADING SPINNER
// ============================================

function showLoadingSpinner(text = 'Chargement...') {
    const spinner = document.createElement('div');
    spinner.id = 'loading-spinner';
    spinner.className = 'fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50';
    spinner.innerHTML = `
        <div class="text-center">
            <div class="inline-block animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-indigo-500 mb-4"></div>
            <p class="text-white text-lg font-medium">${text}</p>
        </div>
    `;
    
    document.body.appendChild(spinner);
}

function hideLoadingSpinner() {
    const spinner = document.getElementById('loading-spinner');
    if (spinner) {
        spinner.classList.add('fade-out');
        setTimeout(() => spinner.remove(), 300);
    }
}

// ============================================
// 🎨 GRADIENT ANIMÉ SUR LOGO
// ============================================

function animateLogo() {
    const logo = document.querySelector('img[alt*="Aïobi"]');
    if (!logo) return;
    
    logo.classList.add('hover-scale', 'smooth-transition');
    
    // Effet de glow au survol
    logo.addEventListener('mouseenter', () => {
        logo.style.filter = 'drop-shadow(0 0 20px rgba(79, 70, 229, 0.8))';
    });
    
    logo.addEventListener('mouseleave', () => {
        logo.style.filter = 'none';
    });
}

// ============================================
// 📱 SMOOTH SCROLL
// ============================================

function smoothScrollTo(targetId) {
    const target = document.getElementById(targetId);
    if (!target) return;
    
    target.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// ============================================
// 🎯 CLICK EFFECTS
// ============================================

function addClickEffect(button) {
    button.addEventListener('click', function(e) {
        const ripple = document.createElement('span');
        const rect = this.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = e.clientX - rect.left - size / 2;
        const y = e.clientY - rect.top - size / 2;
        
        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        ripple.classList.add('ripple-effect');
        
        this.appendChild(ripple);
        
        setTimeout(() => ripple.remove(), 600);
    });
}

// ============================================
// 🚀 AUTO-INIT AU CHARGEMENT
// ============================================

window.addEventListener('load', () => {
    animateLogo();
    
    // Appliquer les animations aux éléments existants
    document.querySelectorAll('.btn-primary, .btn-success').forEach(btn => {
        btn.classList.add('btn-gradient-animated');
    });
    
    // Badges avec pulse
    document.querySelectorAll('.badge').forEach(badge => {
        if (badge.textContent.includes('En cours') || badge.textContent.includes('Training')) {
            badge.classList.add('badge-pulse');
        }
    });
});

// ============================================
// 📤 EXPORT DES FONCTIONS GLOBALES
// ============================================

window.AiobiAnimations = {
    showToast,
    celebrateSuccess,
    showModal,
    hideModal,
    animateProgressBar,
    showSkeleton,
    hideSkeleton,
    showLoadingSpinner,
    hideLoadingSpinner,
    smoothScrollTo,
    createConfetti
};
