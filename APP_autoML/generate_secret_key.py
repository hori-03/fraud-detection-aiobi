"""
Génère une SECRET_KEY sécurisée pour Flask en production
À exécuter avant le déploiement pour obtenir une clé unique
"""
import secrets

if __name__ == "__main__":
    # Générer une clé sécurisée de 32 bytes (256 bits)
    secret_key = secrets.token_hex(32)
    
    print("=" * 60)
    print("🔐 SECRET KEY pour Production Railway")
    print("=" * 60)
    print()
    print("Copier cette valeur dans Railway → Variables d'environnement:")
    print()
    print(f"SECRET_KEY={secret_key}")
    print()
    print("=" * 60)
    print("⚠️  Ne JAMAIS commiter cette clé dans Git!")
    print("=" * 60)
