# 🎯 Aide-Mémoire: Configuration GOOGLE_REDIRECT_URI

## ❓ Pourquoi 2 valeurs différentes?

### En Développement (Local)
```bash
GOOGLE_REDIRECT_URI=http://127.0.0.1:5000/auth/google/callback
```
- Ton application tourne sur **ton ordinateur**
- Accessible via `http://127.0.0.1:5000`
- Google OAuth redirige vers **localhost**

### En Production (Railway)
```bash
GOOGLE_REDIRECT_URI=https://fraud-detection-production.railway.app/auth/google/callback
```
- Ton application tourne sur **les serveurs Railway**
- Accessible via une URL publique (ex: `https://fraud-detection-production.railway.app`)
- Google OAuth redirige vers **l'URL Railway**

---

## 📋 Workflow de Déploiement

### Phase 1: Développement Local ✅

**Fichier**: `APP_autoML/.env`
```bash
GOOGLE_REDIRECT_URI=http://127.0.0.1:5000/auth/google/callback
```

**Google Cloud Console**:
- Authorized redirect URIs: `http://127.0.0.1:5000/auth/google/callback`

**Test**: Login Google fonctionne en local ✅

---

### Phase 2: Premier Déploiement Railway

**Railway Variables** (provisoire):
```bash
GOOGLE_REDIRECT_URI=http://127.0.0.1:5000/auth/google/callback
```

⚠️ **Le login Google NE MARCHERA PAS** mais c'est normal!
- Le but est juste de déployer l'app
- Tu vas obtenir l'URL Railway

**Railway Deploy**: L'app est en ligne, tu obtiens l'URL: `https://fraud-detection-production.railway.app`

---

### Phase 3: Configuration Production

#### Étape 1: Mettre à jour Railway
**Railway Dashboard → Variables → Edit**:
```bash
GOOGLE_REDIRECT_URI=https://fraud-detection-production.railway.app/auth/google/callback
```
- Remplacer `fraud-detection-production` par le nom de TON app
- Sauvegarder → Railway redéploie automatiquement

#### Étape 2: Mettre à jour Google Cloud Console
1. Aller sur: https://console.cloud.google.com/apis/credentials
2. Cliquer sur ton OAuth 2.0 Client ID
3. Section "Authorized redirect URIs"
4. **Ajouter** (ne pas remplacer!): `https://fraud-detection-production.railway.app/auth/google/callback`
5. Sauvegarder

**Google Cloud Console** devrait maintenant avoir:
```
Authorized redirect URIs:
✅ http://127.0.0.1:5000/auth/google/callback  (dev local)
✅ https://fraud-detection-production.railway.app/auth/google/callback  (prod Railway)
```

#### Étape 3: Tester
- Aller sur: `https://fraud-detection-production.railway.app`
- Cliquer sur "Login with Google"
- ✅ Devrait fonctionner!

---

## 🔍 Vérification

### Local (Dev)
```bash
echo %GOOGLE_REDIRECT_URI%
# Devrait afficher: http://127.0.0.1:5000/auth/google/callback
```

### Railway (Prod)
```bash
railway variables --service <ton-service>
# Devrait afficher: https://<ton-app>.railway.app/auth/google/callback
```

### Google Cloud Console
Authorized redirect URIs devrait contenir **les 2**:
- ✅ `http://127.0.0.1:5000/auth/google/callback`
- ✅ `https://<ton-app>.railway.app/auth/google/callback`

---

## 🆘 Erreurs Courantes

### Erreur: "redirect_uri_mismatch"

**Message complet**:
```
Error 400: redirect_uri_mismatch
The redirect URI in the request, https://mon-app.railway.app/auth/google/callback,
does not match the ones authorized for the OAuth client.
```

**Cause**: L'URI n'est pas dans Google Cloud Console

**Solution**:
1. Copier l'URI exacte du message d'erreur
2. Aller dans Google Cloud Console
3. L'ajouter dans "Authorized redirect URIs"
4. ⚠️ Vérifier qu'il n'y a PAS d'espace ou de slash `/` en trop

---

### Erreur: "Invalid redirect_uri"

**Cause**: Faute de frappe dans `GOOGLE_REDIRECT_URI`

**Vérifications**:
- ✅ Commence par `https://` (pas `http://` en prod)
- ✅ Se termine par `/auth/google/callback` (avec le `/`)
- ✅ Pas d'espace avant ou après
- ✅ Correspond EXACTEMENT à l'URL Railway

---

### Login Google fonctionne en local mais pas en prod

**Cause**: `GOOGLE_REDIRECT_URI` pas mise à jour dans Railway

**Solution**:
1. Railway Dashboard → Variables
2. Vérifier `GOOGLE_REDIRECT_URI`
3. Doit être: `https://<ton-app>.railway.app/auth/google/callback`
4. Si c'est `http://127.0.0.1:5000/...` → CHANGER!

---

## 📝 Résumé Visuel

```
┌─────────────────────────────────────────────────────────────┐
│  GOOGLE CLOUD CONSOLE                                       │
│  Authorized redirect URIs:                                  │
│  ✅ http://127.0.0.1:5000/auth/google/callback             │
│  ✅ https://fraud-detection-prod.railway.app/auth/...      │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Autorise les 2 URLs
                            ▼
┌────────────────────────┐        ┌──────────────────────────┐
│  LOCAL (.env)          │        │  RAILWAY (Variables)     │
│  GOOGLE_REDIRECT_URI=  │        │  GOOGLE_REDIRECT_URI=    │
│  http://127.0.0.1:5000 │        │  https://<app>.railway   │
│  /auth/google/callback │        │  .app/auth/google/       │
│                        │        │  callback                │
└────────────────────────┘        └──────────────────────────┘
        ▲                                     ▲
        │                                     │
        │ Utilisé en dev                      │ Utilisé en prod
        │                                     │
┌───────────────────┐            ┌──────────────────────────┐
│  Ton ordinateur   │            │  Serveurs Railway        │
│  localhost:5000   │            │  https://app.railway.app │
└───────────────────┘            └──────────────────────────┘
```

---

## ✅ Checklist Finale

Avant de considérer le déploiement terminé:

- [ ] App déployée sur Railway
- [ ] URL Railway notée (ex: `https://fraud-detection-production.railway.app`)
- [ ] `GOOGLE_REDIRECT_URI` mise à jour dans Railway avec l'URL complète
- [ ] URI ajoutée dans Google Cloud Console
- [ ] Login Google testé et fonctionnel sur l'URL Railway
- [ ] Login Google fonctionne toujours en local
- [ ] Les 2 URIs sont dans Google Cloud Console

**Temps estimé**: 5-10 minutes après le premier déploiement
