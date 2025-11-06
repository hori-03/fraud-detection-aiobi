# 🔌 Se connecter à PostgreSQL Railway

## Informations de connexion

```
Host: switchyard.proxy.rlwy.net
Port: 45478
Database: railway
User: postgres
Password: rWrQsGaGlBUqQLtXFUVRMRgBrudpIPJX
```

## Option 1 : Avec pgAdmin (Interface graphique)

1. **Télécharge pgAdmin** : https://www.pgadmin.org/download/
2. **Créer une nouvelle connexion** :
   - Right click "Servers" → Create → Server
   - **General tab** :
     - Name: `Railway Fraud Detection`
   - **Connection tab** :
     - Host: `switchyard.proxy.rlwy.net`
     - Port: `45478`
     - Maintenance database: `railway`
     - Username: `postgres`
     - Password: `rWrQsGaGlBUqQLtXFUVRMRgBrudpIPJX`
     - ✅ Save password
3. **Connect** → Tu verras tes tables dans `railway → Schemas → public → Tables`

## Option 2 : Avec DBeaver (Gratuit et plus léger)

1. **Télécharge DBeaver** : https://dbeaver.io/download/
2. **New Database Connection** → PostgreSQL
3. Entre les mêmes infos que ci-dessus
4. **Test Connection** → **Finish**

## Option 3 : En ligne de commande (psql)

```bash
psql postgresql://postgres:rWrQsGaGlBUqQLtXFUVRMRgBrudpIPJX@switchyard.proxy.rlwy.net:45478/railway
```

## Option 4 : Avec Python (depuis ton code)

```python
# Déjà configuré dans ton app Flask !
# Regarde APP_autoML/.env
DATABASE_URL=postgresql://postgres:rWrQsGaGlBUqQLtXFUVRMRgBrudpIPJX@switchyard.proxy.rlwy.net:45478/railway
```

## 📊 Tes tables actuelles

- **users** : 1 utilisateur (demo@example.com / demo123)
- **licenses** : 1 licence trial (14 jours, 3 modèles max)
- **training_history** : Vide (se remplira quand tu entraîneras des modèles)

## 🧪 Tester la connexion Python

```python
import psycopg
conn = psycopg.connect("postgresql://postgres:rWrQsGaGlBUqQLtXFUVRMRgBrudpIPJX@switchyard.proxy.rlwy.net:45478/railway")
cursor = conn.cursor()
cursor.execute("SELECT email, username FROM users;")
print(cursor.fetchall())  # [('demo@example.com', 'demo')]
conn.close()
```
