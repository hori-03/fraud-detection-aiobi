/**
 * 👁️ AÏOBI INTERACTIVE EYES
 * Yeux qui suivent la souris et clignent automatiquement
 */

class AiobiEyes {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container ${containerId} not found`);
            return;
        }
        
        this.leftEye = null;
        this.rightEye = null;
        this.isBlinking = false;
        this.lastMouseX = window.innerWidth / 2;
        this.lastMouseY = window.innerHeight / 2;
        this.blinkInterval = null;
        
        this.init();
    }
    
    init() {
        console.log('🎯 Initializing Aïobi Eyes...');
        console.log('Container:', this.container);
        
        // Créer la structure des yeux
        this.createEyes();
        
        // Ajouter les event listeners
        this.addEventListeners();
        
        // Démarrer les clignements automatiques
        this.startBlinking();
        
        console.log('✅ Eyes initialized!');
        console.log('Left eye:', this.leftEye);
        console.log('Right eye:', this.rightEye);
    }
    
    createEyes() {
        // Container pour les yeux - points simples style cartoon
        const eyesContainer = document.createElement('div');
        eyesContainer.className = 'aiobi-eyes-container';
        eyesContainer.innerHTML = `
            <div class="aiobi-eye left-eye"></div>
            <div class="aiobi-eye right-eye"></div>
        `;
        
        this.container.appendChild(eyesContainer);
        
        // Références
        this.leftEye = this.container.querySelector('.left-eye');
        this.rightEye = this.container.querySelector('.right-eye');
    }
    
    addEventListeners() {
        // Suivre le mouvement de la souris
        document.addEventListener('mousemove', (e) => {
            this.moveEyes(e.clientX, e.clientY);
        });
        
        // Cligner lors du clic
        document.addEventListener('click', () => {
            this.blink();
        });
        
        // Suivre le scroll (garde le regard sur la souris)
        document.addEventListener('scroll', () => {
            const lastMouseX = this.lastMouseX || window.innerWidth / 2;
            const lastMouseY = this.lastMouseY || window.innerHeight / 2;
            this.moveEyes(lastMouseX, lastMouseY);
        });
        
        console.log('👂 Event listeners added');
    }
    
    moveEyes(mouseX, mouseY) {
        if (!this.leftEye || !this.rightEye || !this.container) return;
        
        this.lastMouseX = mouseX;
        this.lastMouseY = mouseY;
        
        // Position du logo (utiliser this.container au lieu de this.logoContainer)
        const rect = this.container.getBoundingClientRect();
        const logoX = rect.left + rect.width / 2;
        const logoY = rect.top + rect.height / 2;
        
        // Calculer l'angle et la distance depuis le centre du logo
        const angle = Math.atan2(mouseY - logoY, mouseX - logoX);
        const distance = Math.hypot(mouseX - logoX, mouseY - logoY);
        
        // Mouvement très subtil - les yeux bougent légèrement
        const maxMove = 4; // pixels - mouvement minimal pour style cartoon
        const moveDistance = Math.min(distance / 60, maxMove);
        const moveX = Math.cos(angle) * moveDistance;
        const moveY = Math.sin(angle) * moveDistance;
        
        // Appliquer le mouvement avec setProperty pour forcer !important
        this.leftEye.style.setProperty('transform', `translate(${moveX}px, ${moveY}px)`, 'important');
        this.rightEye.style.setProperty('transform', `translate(${moveX}px, ${moveY}px)`, 'important');
        
        console.log('👁️ Moving eyes:', moveX, moveY);
    }
    
    blink() {
        if (this.isBlinking) return;
        
        this.isBlinking = true;
        
        // Sauvegarder les transforms actuels
        const leftTransform = this.leftEye.style.transform;
        const rightTransform = this.rightEye.style.transform;
        
        // Ajouter une transition douce
        this.leftEye.style.transition = 'transform 0.1s ease-out, opacity 0.1s ease-out';
        this.rightEye.style.transition = 'transform 0.1s ease-out, opacity 0.1s ease-out';
        
        // Fermer les yeux progressivement (scaleY + garder le translate)
        this.leftEye.style.setProperty('transform', `${leftTransform} scaleY(0.15)`, 'important');
        this.rightEye.style.setProperty('transform', `${rightTransform} scaleY(0.15)`, 'important');
        this.leftEye.style.setProperty('opacity', '0.5', 'important');
        this.rightEye.style.setProperty('opacity', '0.5', 'important');
        
        // Rouvrir après 200ms (plus lent)
        setTimeout(() => {
            this.leftEye.style.setProperty('transform', leftTransform, 'important');
            this.rightEye.style.setProperty('transform', rightTransform, 'important');
            this.leftEye.style.setProperty('opacity', '1', 'important');
            this.rightEye.style.setProperty('opacity', '1', 'important');
            
            // Réinitialiser la transition après
            setTimeout(() => {
                this.leftEye.style.transition = 'transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1)';
                this.rightEye.style.transition = 'transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1)';
                this.isBlinking = false;
            }, 100);
        }, 200);
        
        console.log('😉 Blink!');
    }
    
    startBlinking() {
        // Cligner aléatoirement toutes les 2-5 secondes
        const scheduleNextBlink = () => {
            const delay = 2000 + Math.random() * 3000;
            setTimeout(() => {
                this.blink();
                scheduleNextBlink();
            }, delay);
        };
        
        scheduleNextBlink();
    }
    
    destroy() {
        if (this.blinkInterval) {
            clearInterval(this.blinkInterval);
        }
        const eyesContainer = this.container.querySelector('.aiobi-eyes-container');
        if (eyesContainer) {
            eyesContainer.remove();
        }
    }
}

// Auto-initialisation pour les pages de connexion/inscription
document.addEventListener('DOMContentLoaded', () => {
    const logoContainer = document.getElementById('aiobi-logo-interactive');
    if (logoContainer) {
        new AiobiEyes('aiobi-logo-interactive');
    }
});
