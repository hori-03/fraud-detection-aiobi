# 🚀 GUIDE COMPLET: Créer un Bucket AWS S3 (Débutant)

## 📋 PRÉREQUIS
- Une adresse email
- Une carte bancaire (pour vérification, mais service **GRATUIT** la première année)
- 30 minutes de temps

---

## ÉTAPE 1: CRÉER UN COMPTE AWS (15 minutes)

### 1.1 Aller sur AWS
1. Ouvrez votre navigateur
2. Allez sur: https://aws.amazon.com/fr/
3. Cliquez sur **"Créer un compte AWS"** (en haut à droite)

### 1.2 Remplir les informations
```
Email: votre_email@gmail.com
Nom du compte: fraud-detection-ml (ou autre nom)
```

### 1.3 Vérification email
- AWS vous envoie un code à 6 chiffres
- Entrez le code reçu par email

### 1.4 Créer un mot de passe root
```
Mot de passe: (minimum 8 caractères, majuscule, minuscule, chiffre)
Exemple: MyAws2024!Pass
```

### 1.5 Informations de contact
```
Type de compte: Particulier (Personal)
Nom complet: Votre Nom
Téléphone: +33 6 XX XX XX XX (ou votre pays)
Adresse: Votre adresse complète
```

### 1.6 Informations de paiement
⚠️ **IMPORTANT:** AWS demande une carte bancaire pour vérification, mais:
- ✅ Vous ne serez **PAS facturé** si vous restez dans les limites gratuites
- ✅ **Offre gratuite:** 5 GB S3 gratuit pendant 12 mois
- ✅ Votre projet (~1.8 GB) reste **GRATUIT** la première année

```
Numéro de carte: XXXX XXXX XXXX XXXX
Date d'expiration: MM/AA
CVV: XXX
```

AWS va faire une **pré-autorisation de 1€** (remboursé immédiatement) pour vérifier la carte.

### 1.7 Vérification d'identité (téléphone)
AWS vous appelle ou vous envoie un SMS avec un code à 4 chiffres.

### 1.8 Choisir le plan de support
```
✅ Sélectionnez: "Basic Support - Free"
```
(Le plan gratuit suffit amplement)

### 1.9 Félicitations ! 🎉
Vous recevez un email: "Welcome to Amazon Web Services"

---

## ÉTAPE 2: SE CONNECTER À AWS CONSOLE (2 minutes)

### 2.1 Aller sur la console AWS
1. Allez sur: https://console.aws.amazon.com/
2. Cliquez sur **"Root user"** (utilisateur racine)
3. Entrez votre **email**
4. Entrez votre **mot de passe**
5. Cliquez sur **"Sign in"**

### 2.2 Vous êtes sur le Dashboard AWS
Vous voyez:
- Services (en haut)
- Région (en haut à droite, ex: US East (Ohio))
- Votre nom de compte (en haut à droite)

---

## ÉTAPE 3: CRÉER UN BUCKET S3 (5 minutes)

### 3.1 Accéder au service S3
```
Méthode 1 (Barre de recherche):
1. Cliquez sur la barre de recherche en haut
2. Tapez "S3"
3. Cliquez sur "S3" (Scalable Storage in the Cloud)

Méthode 2 (Menu Services):
1. Cliquez sur "Services" (en haut à gauche)
2. Sous "Storage", cliquez sur "S3"
```

### 3.2 Créer le bucket
1. Cliquez sur le bouton orange **"Create bucket"** (Créer un compartiment)

### 3.3 Configuration du bucket

#### A) General configuration
```
Bucket name: fraud-detection-models
⚠️ IMPORTANT: Le nom doit être UNIQUE au monde (comme un nom de domaine)

Si "fraud-detection-models" est pris, essayez:
- fraud-detection-models-2024
- fraud-detection-ml-models
- automl-fraud-models-yourname
```

#### B) AWS Region (Région)
```
✅ Choisissez: US East (N. Virginia) us-east-1
OU
✅ US East (Ohio) us-east-2

💡 Pourquoi? Moins cher et plus rapide depuis l'Europe/Afrique
```

#### C) Object Ownership (Propriété des objets)
```
✅ Laissez: "ACLs disabled (recommended)"
```

#### D) Block Public Access settings (Bloquer l'accès public)
```
✅ LAISSEZ TOUT COCHÉ (sécurité importante!)

[✓] Block all public access
    [✓] Block public access to buckets and objects granted through new ACLs
    [✓] Block public access to buckets and objects granted through any ACLs
    [✓] Block public access to buckets and objects granted through new public bucket policies
    [✓] Block public and cross-account access to buckets and objects through any public bucket policies

💡 Vos modèles seront accessibles via credentials uniquement (sécurisé)
```

#### E) Bucket Versioning (Gestion des versions)
```
⚪ Disable (Désactiver)
💡 Pas nécessaire pour les modèles ML (économise de l'espace)
```

#### F) Tags (Étiquettes) - OPTIONNEL
```
Key: Project
Value: fraud-detection

Key: Environment
Value: production

💡 Utile pour organiser vos buckets si vous en avez plusieurs
```

#### G) Default encryption (Chiffrement par défaut)
```
✅ Server-side encryption: Enabled
✅ Encryption type: Amazon S3 managed keys (SSE-S3)

💡 Vos modèles seront automatiquement chiffrés (sécurité)
```

#### H) Advanced settings (Paramètres avancés)
```
✅ Object Lock: Disabled
💡 Pas nécessaire
```

### 3.4 Créer !
1. Cliquez sur le bouton orange **"Create bucket"** (en bas)
2. ✅ Vous voyez: "Successfully created bucket fraud-detection-models"

---

## ÉTAPE 4: CRÉER UN UTILISATEUR IAM (Sécurité) (8 minutes)

⚠️ **NE PAS utiliser les credentials root!** Créons un utilisateur IAM dédié.

### 4.1 Accéder à IAM
```
1. Barre de recherche (en haut): Tapez "IAM"
2. Cliquez sur "IAM" (Identity and Access Management)
```

### 4.2 Créer un utilisateur
```
1. Dans le menu gauche, cliquez sur "Users" (Utilisateurs)
2. Cliquez sur le bouton "Create user" (Créer un utilisateur)
```

### 4.3 User details (Détails de l'utilisateur)
```
User name: s3-fraud-detection-app
✅ Provide user access to the AWS Management Console: NON COCHÉ
(On veut juste des credentials programmatiques)
```

### 4.4 Set permissions (Définir les autorisations)
```
✅ Sélectionnez: "Attach policies directly" (Attacher des stratégies directement)

Dans la barre de recherche des policies:
1. Tapez "S3"
2. ✅ Cochez: "AmazonS3FullAccess"
   (Permet à l'app de lire/écrire dans tous vos buckets S3)

💡 Plus tard, vous pourrez restreindre à un seul bucket (sécurité avancée)
```

### 4.5 Review and create
```
1. Vérifiez les infos
2. Cliquez sur "Create user"
```

### 4.6 Créer les Access Keys
```
1. Cliquez sur le nom de l'utilisateur créé: "s3-fraud-detection-app"
2. Cliquez sur l'onglet "Security credentials"
3. Descendez à la section "Access keys"
4. Cliquez sur "Create access key"
5. Sélectionnez le use case: "Application running outside AWS"
6. Cochez "I understand the above recommendation"
7. Cliquez "Next"
8. Description tag (optionnel): "Flask fraud detection app"
9. Cliquez "Create access key"
```

### 4.7 ⚠️ SAUVEGARDER VOS CREDENTIALS
```
✅ Access key ID: AKIAIOSFODNN7EXAMPLE
✅ Secret access key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

⚠️ IMPORTANT: Vous ne pourrez PLUS voir la secret key après avoir fermé cette page!

Options pour sauvegarder:
1. Cliquez sur "Download .csv file" → Télécharge un fichier CSV
2. Copiez-collez dans un fichier texte sécurisé
3. Utilisez un gestionnaire de mots de passe (LastPass, 1Password, etc.)

🚨 NE JAMAIS committer ces credentials dans Git !
```

---

## ÉTAPE 5: INSTALLER ET CONFIGURER AWS CLI (5 minutes)

### 5.1 Installer AWS CLI

#### Windows:
```bash
# Télécharger l'installeur
https://awscli.amazonaws.com/AWSCLIV2.msi

# Exécuter l'installeur
# Cliquez sur "Next" → "Next" → "Install"

# Vérifier l'installation
aws --version
# Output: aws-cli/2.15.10 Python/3.11.6 Windows/10 exe/AMD64 prompt/off
```

#### macOS:
```bash
# Via Homebrew
brew install awscli

# Vérifier
aws --version
```

#### Linux:
```bash
# Ubuntu/Debian
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Vérifier
aws --version
```

### 5.2 Configurer AWS CLI
```bash
aws configure

# Répondez aux questions:
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: us-east-1
Default output format [None]: json
```

### 5.3 Tester la connexion
```bash
# Lister vos buckets
aws s3 ls

# Output attendu:
# 2024-11-04 10:30:00 fraud-detection-models

# Créer un fichier test
echo "Test S3" > test.txt

# Uploader dans S3
aws s3 cp test.txt s3://fraud-detection-ml-models/test.txt

# Output: upload: ./test.txt to s3://fraud-detection-models/test.txt

# Lister le contenu du bucket
aws s3 ls s3://fraud-detection-ml-models/

# Output: 2024-11-04 10:35:00         8 test.txt

# Télécharger
aws s3 cp s3://fraud-detection-ml-models/test.txt test_downloaded.txt

# Supprimer le test
aws s3 rm s3://fraud-detection-ml-models/test.txt
```

✅ **Félicitations ! Votre bucket S3 fonctionne !** 🎉

---

## ÉTAPE 6: CONFIGURER VOTRE APPLICATION PYTHON

### 6.1 Installer boto3
```bash
cd C:\Users\HP\Desktop\fraud-project\APP_autoML
pip install boto3
```

### 6.2 Créer un fichier .env (LOCAL UNIQUEMENT)
```bash
# Dans APP_autoML/.env
AWS_ACCESS_KEY_ID=VOTRE_ACCESS_KEY_ID_ICI
AWS_SECRET_ACCESS_KEY=VOTRE_SECRET_ACCESS_KEY_ICI
AWS_DEFAULT_REGION=eu-north-1
S3_MODEL_BUCKET=fraud-detection-ml-models
STORAGE_TYPE=s3
```

⚠️ **IMPORTANT:** Ajoutez `.env` à votre `.gitignore` !
```bash
echo ".env" >> .gitignore
```

### 6.3 Tester la migration (dry-run)
```bash
python migrate_models_to_s3.py --estimate
# Affiche les coûts estimés

python migrate_models_to_s3.py --bucket fraud-detection-models --dry-run
# Simule sans uploader
```

### 6.4 Migration réelle
```bash
python migrate_models_to_s3.py --bucket fraud-detection-models
# Upload tous les modèles (40 modèles × ~50 MB = ~2 GB)
# Temps: ~5-10 minutes selon connexion
```

### 6.5 Vérifier
```bash
# Via AWS CLI
aws s3 ls s3://fraud-detection-models/automl_models/

# Output attendu:
#                            PRE dataset1/
#                            PRE dataset2/
#                            ...
#                            PRE dataset40/

# Via Python
python migrate_models_to_s3.py --bucket fraud-detection-models --verify
```

---

## ÉTAPE 7: CONFIGURER RAILWAY (PRODUCTION)

### 7.1 Ajouter les variables d'environnement
Dans Railway Dashboard → Votre projet → Variables:

```
AWS_ACCESS_KEY_ID = AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION = us-east-1
S3_MODEL_BUCKET = fraud-detection-models
STORAGE_TYPE = s3
```

### 7.2 Déployer
```bash
git add .
git commit -m "feat: Add S3 storage support for production"
git push railway main
```

### 7.3 Tester en production
```bash
# Via curl
curl -X POST https://your-app.railway.app/api/apply_unlabeled \
  -H "Content-Type: application/json" \
  -d '{"filepath": "test.csv"}'
```

---

## 💰 COÛTS RÉELS

### Votre utilisation (~1.8 GB, 40 modèles)

#### Année 1 (Offre Gratuite)
```
Stockage: 1.8 GB / 5 GB gratuits = GRATUIT ✅
Requêtes GET: ~100/jour × 30 = 3000/mois / 20,000 gratuits = GRATUIT ✅
Requêtes PUT: 200 (migration) / 2,000 gratuits = GRATUIT ✅

TOTAL ANNÉE 1: $0.00 🎉
```

#### Après 12 mois
```
Stockage: 1.8 GB × $0.023/GB/mois = $0.041/mois = $0.50/an
Requêtes GET: 3,000 × $0.0004/1000 = $0.0012/mois = $0.014/an
Requêtes PUT: 0 (déjà uploadés) = $0.00

TOTAL APRÈS: $0.51/an (~€0.50/an) ☕
```

**Moins cher qu'un café par an !** ☕

---

## 🔒 SÉCURITÉ - BONNES PRATIQUES

### ✅ À FAIRE:
1. **Ne jamais** committer vos credentials AWS dans Git
2. Utiliser des **IAM users** (pas le root account)
3. Activer **MFA** (Multi-Factor Authentication) sur votre compte root
4. Créer des **policies restreintes** (accès uniquement à votre bucket)
5. Surveiller les **coûts** via AWS Cost Explorer

### ❌ À NE PAS FAIRE:
1. ❌ Utiliser les credentials root pour l'application
2. ❌ Rendre le bucket **public** (sauf si nécessaire)
3. ❌ Oublier de configurer **lifecycle policies** (nettoyage automatique)
4. ❌ Stocker des **données sensibles non chiffrées**

---

## 🆘 TROUBLESHOOTING

### Erreur: "Access Denied"
```bash
# Vérifier les credentials
aws s3 ls --debug

# Solutions:
1. Vérifiez AWS_ACCESS_KEY_ID et AWS_SECRET_ACCESS_KEY
2. Vérifiez que l'utilisateur IAM a la policy "AmazonS3FullAccess"
3. Vérifiez la région (us-east-1, us-east-2, etc.)
```

### Erreur: "Bucket name already exists"
```bash
# Le nom est déjà pris globalement
# Solutions:
1. Ajoutez un suffixe unique: fraud-detection-models-yourname
2. Ajoutez la date: fraud-detection-models-2024
3. Ajoutez un UUID: fraud-detection-models-abc123
```

### Erreur: "Invalid credentials"
```bash
# Les credentials sont incorrects ou expirés
# Solutions:
1. Recréez des Access Keys dans IAM
2. Vérifiez qu'il n'y a pas d'espaces dans les credentials
3. Exécutez: aws configure (réentrez les credentials)
```

---

## 📚 RESSOURCES UTILES

### Documentation officielle
- AWS S3: https://docs.aws.amazon.com/s3/
- AWS CLI: https://docs.aws.amazon.com/cli/
- Boto3 (Python): https://boto3.amazonaws.com/v1/documentation/api/latest/index.html

### Outils de monitoring
- AWS Cost Explorer: https://console.aws.amazon.com/cost-management/home
- AWS Billing Dashboard: https://console.aws.amazon.com/billing/home

### Calculateur de coûts
- AWS Pricing Calculator: https://calculator.aws/

---

## ✅ CHECKLIST FINALE

Avant de passer en production, vérifiez:

- [ ] Compte AWS créé et vérifié
- [ ] Bucket S3 créé avec nom unique
- [ ] Utilisateur IAM créé avec permissions S3
- [ ] Access Keys créées et sauvegardées en lieu sûr
- [ ] AWS CLI installé et configuré
- [ ] boto3 installé (`pip install boto3`)
- [ ] Fichier `.env` créé avec credentials
- [ ] `.env` ajouté à `.gitignore`
- [ ] Migration dry-run testée
- [ ] Migration réelle effectuée (40 modèles)
- [ ] Migration vérifiée (`--verify`)
- [ ] Variables Railway configurées
- [ ] Code déployé sur Railway
- [ ] Test en production réussi
- [ ] Monitoring des coûts activé

---

## 🎉 FÉLICITATIONS !

Vous avez maintenant:
- ✅ Un bucket S3 professionnel
- ✅ 40 modèles ML stockés dans le cloud
- ✅ Une application production-ready
- ✅ Des coûts quasi-nuls ($0.50/an)
- ✅ Une architecture scalable

**Votre système peut maintenant servir des milliers d'utilisateurs !** 🚀

---

**Créé:** 4 novembre 2025  
**Auteur:** Fraud Detection AutoML System v2.0  
**Pour:** Déploiement Production AWS S3