"""Afficher les données de la base PostgreSQL Railway."""

import psycopg

# Connexion
conn = psycopg.connect(
    'postgresql://postgres:rWrQsGaGlBUqQLtXFUVRMRgBrudpIPJX@switchyard.proxy.rlwy.net:45478/railway'
)
cursor = conn.cursor()

print("\n" + "=" * 70)
print("📊 DONNÉES DANS POSTGRESQL RAILWAY")
print("=" * 70)

# Utilisateurs
cursor.execute("SELECT id, email, username, first_name, last_name, company, is_active, created_at FROM users;")
users = cursor.fetchall()
print("\n👤 UTILISATEURS ({} enregistré(s)):".format(len(users)))
print("-" * 70)
for u in users:
    print(f"  ID: {u[0]}")
    print(f"  Email: {u[1]}")
    print(f"  Username: @{u[2]}")
    print(f"  Nom: {u[3]} {u[4]}")
    print(f"  Entreprise: {u[5] or 'N/A'}")
    print(f"  Actif: {'✅' if u[6] else '❌'}")
    print(f"  Créé le: {u[7]}")
    print()

# Licences
cursor.execute("SELECT id, user_id, license_type, max_models, max_datasets_size_mb, is_active, expires_at, license_key FROM licenses;")
licenses = cursor.fetchall()
print("\n🎫 LICENCES ({} enregistrée(s)):".format(len(licenses)))
print("-" * 70)
for l in licenses:
    print(f"  ID: {l[0]}")
    print(f"  User ID: {l[1]}")
    print(f"  Type: {l[2].upper()}")
    print(f"  Max modèles: {l[3]}")
    print(f"  Max dataset: {l[4]} MB")
    print(f"  Active: {'✅' if l[5] else '❌'}")
    print(f"  Expire: {l[6] or 'Jamais'}")
    print(f"  Clé: {l[7]}")
    print()

# Historique d'entraînement
cursor.execute("SELECT COUNT(*) FROM training_history;")
count = cursor.fetchone()[0]
print(f"\n📈 HISTORIQUE D'ENTRAÎNEMENT: {count} enregistrement(s)")

# Modèles de référence (BACKOFFICE)
cursor.execute("""
    SELECT model_name, dataset_size, num_features, roc_auc, domain, fraud_rate, 
           has_amount, has_timestamp, has_country, is_active 
    FROM reference_models 
    ORDER BY domain, model_name
    LIMIT 10;
""")
ref_models = cursor.fetchall()
cursor.execute("SELECT COUNT(*) FROM reference_models;")
total_ref = cursor.fetchone()[0]

print(f"\n🤖 MODÈLES DE RÉFÉRENCE (BACKOFFICE): {total_ref} modèle(s)")
print("-" * 70)
if ref_models:
    for rm in ref_models:
        print(f"  📦 {rm[0]}")
        dataset_size = f"{rm[1]:,}" if rm[1] else "N/A"
        features = rm[2] if rm[2] else "N/A"
        roc_auc = f"{rm[3]:.4f}" if rm[3] else "N/A"
        fraud_rate = f"{rm[5]*100:.2f}%" if rm[5] else "N/A"
        print(f"     Dataset: {dataset_size} samples | Features: {features} | ROC-AUC: {roc_auc}")
        print(f"     Domain: {rm[4]} | Fraud: {fraud_rate}")
        flags = []
        if rm[6]: flags.append("amount")
        if rm[7]: flags.append("timestamp")
        if rm[8]: flags.append("country")
        print(f"     Has: {', '.join(flags) if flags else 'N/A'} | Active: {'✅' if rm[9] else '❌'}")
        print()
    if total_ref > 10:
        print(f"  ... et {total_ref - 10} autres modèles")

conn.close()

print("\n" + "=" * 70)
print("✅ Base PostgreSQL Railway accessible et opérationnelle!")
print("🌐 Tu peux aussi accéder via Railway.app → PostgreSQL → Data")
print("=" * 70 + "\n")
