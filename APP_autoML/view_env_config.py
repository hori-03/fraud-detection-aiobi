"""
Affiche les variables d'environnement actuelles (masque les secrets)
Utile pour vérifier la configuration avant déploiement
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Charger .env
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

def mask_secret(value, show_chars=4):
    """Masque un secret en ne montrant que les premiers caractères"""
    if not value:
        return "❌ NON DÉFINIE"
    if len(value) <= show_chars:
        return "*" * len(value)
    return value[:show_chars] + "*" * (len(value) - show_chars)

def check_url(url):
    """Vérifie si une URL est correcte"""
    if not url:
        return "❌ NON DÉFINIE"
    if url.startswith("http://127.0.0.1") or url.startswith("http://localhost"):
        return f"⚠️  DEV: {url}"
    if url.startswith("https://"):
        return f"✅ PROD: {url}"
    return f"❓ {url}"

if __name__ == "__main__":
    print("=" * 70)
    print("🔍 VÉRIFICATION DES VARIABLES D'ENVIRONNEMENT")
    print("=" * 70)
    print()
    
    # Flask
    print("📦 FLASK")
    print(f"  FLASK_ENV        : {os.getenv('FLASK_ENV', '❌ NON DÉFINIE')}")
    print(f"  FLASK_DEBUG      : {os.getenv('FLASK_DEBUG', '❌ NON DÉFINIE')}")
    print(f"  SECRET_KEY       : {mask_secret(os.getenv('SECRET_KEY'))}")
    print()
    
    # Database
    print("💾 BASE DE DONNÉES")
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        # Masquer le password dans l'URL
        if '@' in db_url:
            parts = db_url.split('@')
            user_part = parts[0].split('://')[1]
            if ':' in user_part:
                user, pwd = user_part.split(':')
                masked_url = db_url.replace(pwd, mask_secret(pwd))
                print(f"  DATABASE_URL     : {masked_url}")
            else:
                print(f"  DATABASE_URL     : {db_url}")
        else:
            print(f"  DATABASE_URL     : {db_url}")
    else:
        print(f"  DATABASE_URL     : ❌ NON DÉFINIE")
    print()
    
    # AWS S3
    print("☁️  AWS S3")
    print(f"  AWS_ACCESS_KEY_ID     : {mask_secret(os.getenv('AWS_ACCESS_KEY_ID'))}")
    print(f"  AWS_SECRET_ACCESS_KEY : {mask_secret(os.getenv('AWS_SECRET_ACCESS_KEY'))}")
    print(f"  AWS_DEFAULT_REGION    : {os.getenv('AWS_DEFAULT_REGION', '❌ NON DÉFINIE')}")
    print(f"  S3_MODEL_BUCKET       : {os.getenv('S3_MODEL_BUCKET', '❌ NON DÉFINIE')}")
    print(f"  STORAGE_TYPE          : {os.getenv('STORAGE_TYPE', '❌ NON DÉFINIE')}")
    print()
    
    # Google OAuth
    print("🔐 GOOGLE OAUTH")
    print(f"  GOOGLE_CLIENT_ID      : {mask_secret(os.getenv('GOOGLE_CLIENT_ID'))}")
    print(f"  GOOGLE_CLIENT_SECRET  : {mask_secret(os.getenv('GOOGLE_CLIENT_SECRET'))}")
    redirect_uri = os.getenv('GOOGLE_REDIRECT_URI')
    print(f"  GOOGLE_REDIRECT_URI   : {check_url(redirect_uri)}")
    print()
    
    # Validation
    print("=" * 70)
    print("✅ VALIDATION")
    print("=" * 70)
    
    issues = []
    warnings = []
    
    # Vérifications critiques
    if not os.getenv('SECRET_KEY') or os.getenv('SECRET_KEY') == 'dev-secret-key-change-in-production':
        warnings.append("⚠️  SECRET_KEY: Utilise la clé de développement (générer une nouvelle pour prod)")
    
    if not os.getenv('DATABASE_URL'):
        issues.append("❌ DATABASE_URL non définie")
    
    if not os.getenv('AWS_ACCESS_KEY_ID'):
        issues.append("❌ AWS_ACCESS_KEY_ID non définie")
    
    if not os.getenv('GOOGLE_REDIRECT_URI'):
        issues.append("❌ GOOGLE_REDIRECT_URI non définie")
    elif redirect_uri and redirect_uri.startswith("http://127.0.0.1"):
        warnings.append("⚠️  GOOGLE_REDIRECT_URI: Utilise localhost (OK pour dev, changer pour prod)")
    
    env = os.getenv('FLASK_ENV', 'development')
    if env == 'production' and os.getenv('FLASK_DEBUG') == '1':
        warnings.append("⚠️  FLASK_DEBUG=1 en production (dangereux, mettre à 0)")
    
    # Affichage
    if issues:
        print("\n❌ PROBLÈMES CRITIQUES:")
        for issue in issues:
            print(f"  {issue}")
    
    if warnings:
        print("\n⚠️  AVERTISSEMENTS:")
        for warning in warnings:
            print(f"  {warning}")
    
    if not issues and not warnings:
        print("\n✅ Configuration OK!")
    
    print()
    print("=" * 70)
    print("💡 PROCHAINES ÉTAPES:")
    print("=" * 70)
    
    if env == 'development':
        print("\n📍 Mode DÉVELOPPEMENT détecté")
        print("  Pour passer en production:")
        print("  1. Générer SECRET_KEY: python generate_secret_key.py")
        print("  2. Mettre FLASK_ENV=production et FLASK_DEBUG=0")
        print("  3. Changer GOOGLE_REDIRECT_URI vers l'URL Railway")
    else:
        print("\n🚀 Mode PRODUCTION détecté")
        print("  Vérifier:")
        print("  1. SECRET_KEY est unique et sécurisée")
        print("  2. GOOGLE_REDIRECT_URI pointe vers Railway")
        print("  3. DATABASE_URL est l'URL interne Railway")
    
    print()
