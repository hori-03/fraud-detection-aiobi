# 🎯 Guide Simple: Comment Fonctionne apply_automl_production.py

## 📖 Concept de Base

**apply_automl_production.py** = Appliquer un modèle déjà entraîné sur de nouvelles données non-étiquetées

```
┌─────────────────────────────────────────────────────────────┐
│  DONNÉES PRODUCTION (nouvelles transactions)                │
│  ❓ Pas d'étiquette is_fraud                                │
│  📊 On veut savoir: lesquelles sont frauduleuses?          │
└─────────────────────────────────────────────────────────────┘
                           ⬇
┌─────────────────────────────────────────────────────────────┐
│  MODÈLE ENTRAÎNÉ (déjà sauvegardé)                         │
│  ✅ Déjà appris sur Dataset1, Dataset2, etc.               │
│  🧠 Connaît les patterns de fraude                          │
└─────────────────────────────────────────────────────────────┘
                           ⬇
┌─────────────────────────────────────────────────────────────┐
│  PRÉDICTIONS                                                 │
│  Transaction 1: 95% fraude ⚠️                               │
│  Transaction 2: 2% fraude ✅                                │
│  Transaction 3: 78% fraude ⚠️                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 Exemple Concret

### Situation:
Vous êtes une banque. Chaque jour, vous recevez **100,000 nouvelles transactions**. Vous voulez identifier lesquelles sont frauduleuses **avant** de les approuver.

### Solution:

```bash
# Appliquer le modèle entraîné sur les nouvelles transactions
python apply_automl_production.py \
  --dataset nouvelles_transactions.csv \
  --auto_match \
  --output predictions_aujourdhui
```

### Résultat:
```
📊 Résultats:
   100,000 transactions analysées
   ⚠️  147 fraudes détectées (HIGH RISK >70%)
   ⚡ 2,345 suspects (MEDIUM RISK 50-70%)
   ✅ 97,508 normales (LOW RISK <50%)
```

---

## 🎬 Les Étapes (Mode Simple)

### Étape 1: Chargement des Données

```
Fichier: nouvelles_transactions.csv
┌──────────────────────────────────────────────────┐
│ tx_id | amount | merchant | country | time       │
├──────────────────────────────────────────────────┤
│ TX001 | 150.00 | Amazon   | FR      | 14:30     │
│ TX002 | 5000.00| Casino   | RU      | 03:15     │ ← Suspect!
│ TX003 | 25.50  | Carrefour| FR      | 12:00     │
└──────────────────────────────────────────────────┘
```

### Étape 2: Sélection du Modèle

```
Option A: Auto-match (recommandé)
┌────────────────────────────────────────┐
│ Script analyse les colonnes:          │
│   - amount ✓                           │
│   - merchant ✓                         │
│   - country ✓                          │
│   - time ✓                             │
│                                        │
│ Trouve le meilleur modèle:            │
│ → Dataset27 (similarité: 92%)         │
└────────────────────────────────────────┘

Option B: Manuel
--model dataset27
```

### Étape 3: Application du Modèle

```
┌─────────────────────────────────────┐
│  Modèle XGBoost (Dataset27)        │
│  Entraîné sur 50,000 transactions  │
│  ROC-AUC: 99.84%                   │
└─────────────────────────────────────┘
              ⬇
      Analyse chaque transaction
              ⬇
┌─────────────────────────────────────┐
│ TX001: amount=150, merchant=Amazon  │
│ → Pattern: Normal                   │
│ → Probabilité fraude: 3%            │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│ TX002: amount=5000, country=RU      │
│ → Pattern: SUSPECT                  │
│ → Probabilité fraude: 94%           │
└─────────────────────────────────────┘
```

### Étape 4: Export des Résultats

```
Fichier: predictions_aujourdhui.csv
┌───────────────────────────────────────────────────────┐
│ tx_id | amount | fraud_probability | fraud_prediction │
├───────────────────────────────────────────────────────┤
│ TX001 | 150    | 0.03             | 0                │
│ TX002 | 5000   | 0.94             | 1 ⚠️             │
│ TX003 | 25.50  | 0.01             | 0                │
└───────────────────────────────────────────────────────┘
```

---

## 🆕 Mode Avancé: Isolation Forest

### Pourquoi Isolation Forest?

**XGBoost seul** = Détecte ce qu'il a **déjà vu** pendant l'entraînement

**Problème:** Et si une **nouvelle technique de fraude** apparaît?

```
Exemple:
┌────────────────────────────────────────────────────┐
│ Nouvelle fraude: Achats massifs crypto à 4h du    │
│ matin depuis un nouveau pays                       │
│                                                     │
│ XGBoost: "Je n'ai jamais vu ce pattern" 🤔        │
│ → Probabilité: 45% (pas assez confiant)           │
└────────────────────────────────────────────────────┘
```

### Solution: Isolation Forest

**Isolation Forest** = Détecte les **comportements bizarres** (anomalies)

```bash
python apply_automl_production.py \
  --dataset nouvelles_transactions.csv \
  --auto_match \
  --anomaly_detection  ← Active Isolation Forest
```

### Comment ça marche?

```
┌──────────────────────────────────────────────────────┐
│  ISOLATION FOREST                                    │
│  "Cette transaction est bizarre par rapport          │
│   à toutes les autres"                               │
└──────────────────────────────────────────────────────┘
              +
┌──────────────────────────────────────────────────────┐
│  XGBOOST                                             │
│  "Cette transaction ressemble aux fraudes            │
│   que j'ai vues pendant l'entraînement"              │
└──────────────────────────────────────────────────────┘
              ⬇
┌──────────────────────────────────────────────────────┐
│  SCORE COMBINÉ (70% XGBoost + 30% Anomaly)          │
│                                                       │
│  TX002: XGBoost=45%, Anomaly=85%                     │
│  → Combined = 0.7×45% + 0.3×85% = 57%               │
│  → Classification: SUSPECT ⚠️                        │
└──────────────────────────────────────────────────────┘
```

### Exemple Visuel:

```
Transaction normale:
┌────────────────────────────┐
│ Amount: 50€                │
│ Time: 14:00 (normal)       │
│ Country: FR (habituel)     │
│ Merchant: Carrefour        │
│                            │
│ XGBoost:  5% fraude ✅     │
│ Anomaly:  2% bizarre ✅    │
│ Combined: 4% fraude ✅     │
└────────────────────────────┘

Transaction suspecte (pattern connu):
┌────────────────────────────┐
│ Amount: 5000€              │
│ Time: 03:00 (louche)       │
│ Country: RU (nouveau)      │
│ Merchant: Casino Online    │
│                            │
│ XGBoost:  92% fraude ⚠️    │
│ Anomaly:  45% bizarre      │
│ Combined: 78% fraude ⚠️    │
└────────────────────────────┘

Transaction TRÈS suspecte (pattern nouveau):
┌────────────────────────────┐
│ Amount: 9999€              │
│ Time: 04:12 (bizarre)      │
│ Country: NG (jamais vu)    │
│ Merchant: Crypto Exchange  │
│                            │
│ XGBoost:  55% fraude       │
│ Anomaly:  95% bizarre ⚠️   │
│ Combined: 67% fraude ⚠️    │
└────────────────────────────┘
     ↑
     └─ Sans anomaly detection, aurait été raté! (55% < 70%)
```

---

## 🎯 Mode Ensemble (Plusieurs Modèles)

### Concept:

Au lieu d'utiliser **1 seul modèle**, on utilise les **3 meilleurs** et on fait la moyenne.

```bash
python apply_automl_production.py \
  --dataset nouvelles_transactions.csv \
  --ensemble \
  --top_k 3  ← Utilise les 3 meilleurs modèles
```

### Pourquoi?

**Problème:** 1 seul modèle peut se tromper

**Solution:** Démocratie des modèles!

```
Transaction TX002:
┌────────────────────────────────────────┐
│ Modèle Dataset27: 92% fraude          │
│ Modèle Dataset31: 88% fraude          │
│ Modèle Dataset35: 95% fraude          │
│                                        │
│ Moyenne pondérée: 92% fraude ⚠️       │
│ Variance: 0.001 (très stable ✅)      │
└────────────────────────────────────────┘

Transaction TX003:
┌────────────────────────────────────────┐
│ Modèle Dataset27: 45% fraude          │
│ Modèle Dataset31: 12% fraude          │
│ Modèle Dataset35: 78% fraude          │
│                                        │
│ Moyenne pondérée: 42% fraude          │
│ Variance: 0.25 (instable ⚠️)          │
│ → Prédiction PEU FIABLE                │
└────────────────────────────────────────┘
```

**Avantage:** Colonnes ajoutées:
- `prediction_variance`: 0-1 (plus bas = plus stable)
- `prediction_stability`: 1 - variance (1 = très stable)

---

## 📊 Calibration des Probabilités

### Problème:

Les modèles XGBoost donnent parfois des probabilités "timides"

```
❌ Sans calibration:
Transaction vraiment frauduleuse → 65% (pas assez confiant)
Transaction limite → 52% (trop confiant)
```

### Solution: Calibration

```bash
python apply_automl_production.py \
  --dataset nouvelles_transactions.csv \
  --auto_match \
  --calibrate  ← Active la calibration
```

### Effet:

```
✅ Avec calibration:
Transaction vraiment frauduleuse → 65% → 89% (plus confiant)
Transaction limite → 52% → 54% (peu changé)
Transaction normale → 5% → 0.2% (plus sûr)
```

**Transformation:** Sigmoïde qui "étire" les probabilités

```
Avant calibration:        Après calibration:
0.0  ████                 0.0  ███████
0.1  ████                 0.1  ██
0.2  ████                 0.2  █
0.3  ████                 0.3  █
0.4  ████                 0.4  █
0.5  ████                 0.5  ███
0.6  ████                 0.6  █
0.7  ████                 0.7  █
0.8  ████                 0.8  ██
0.9  ████                 0.9  ███████
1.0  ████                 1.0  ███████

Distribution plate        Distribution étirée
(peu de confiance)        (plus de confiance)
```

---

## 🚀 Mode Combiné (Tout Ensemble)

### Commande Ultime:

```bash
python apply_automl_production.py \
  --dataset nouvelles_transactions.csv \
  --ensemble \           ← 3 modèles au lieu d'1
  --top_k 3 \
  --anomaly_detection \  ← Détecte patterns nouveaux
  --calibrate \          ← Probabilités plus fiables
  --rich_export \        ← Export Excel + JSON
  --output results
```

### Pipeline Complet:

```
1. CHARGEMENT
   nouvelles_transactions.csv (100,000 lignes)
              ⬇
2. AUTO-MATCH
   Trouve 3 meilleurs modèles (Dataset27, 31, 35)
   Similarité: 92%, 89%, 87%
              ⬇
3. ENSEMBLE PREDICTIONS
   Applique les 3 modèles
   Moyenne pondérée + variance
              ⬇
4. ANOMALY DETECTION
   Isolation Forest détecte outliers
   Combine: 70% XGBoost + 30% Anomaly
              ⬇
5. CALIBRATION
   Ajuste les probabilités
   Extrêmes plus confiants
              ⬇
6. EXPORT ENRICHI
   Excel: 3 sheets (All, High Risk, Summary)
   JSON: Métadonnées complètes
```

### Résultat Final:

```
📊 predictions_results.xlsx

Sheet 1: All Predictions (100,000 lignes)
┌────────────────────────────────────────────────────────────────┐
│ tx_id | amount | fraud_prob | fraud_prob_calibrated |         │
│       |        |            | anomaly_score | combined_score | │
│       |        |            | prediction_variance |           │
├────────────────────────────────────────────────────────────────┤
│ TX001 | 150    | 0.03       | 0.01 | 0.02 | 0.02 | 0.001     │
│ TX002 | 5000   | 0.92       | 0.98 | 0.85 | 0.94 | 0.003     │
└────────────────────────────────────────────────────────────────┘

Sheet 2: High Risk (147 lignes)
Trié par combined_score décroissant
Top 147 transactions à investiguer

Sheet 3: Summary
┌─────────────────────────────────────┐
│ Total transactions: 100,000         │
│ Fraudes détectées: 147 (0.15%)     │
│ High risk (>70%): 147              │
│ Medium risk (50-70%): 2,345        │
│ Low risk (<50%): 97,508            │
│                                     │
│ Anomalies détectées: 89            │
│ Prédictions stables: 99,234 (99%)  │
└─────────────────────────────────────┘
```

---

## 💡 Cas d'Usage Pratiques

### Cas 1: Banque - Transactions du Jour

```bash
# Chaque matin à 8h00
python apply_automl_production.py \
  --dataset transactions_yesterday.csv \
  --ensemble --top_k 3 \
  --anomaly_detection \
  --threshold 0.7 \
  --output daily_review

# Résultat:
# → Analyst reçoit liste des 50 transactions à vérifier
# → 95% sont effectivement frauduleuses (précision)
```

### Cas 2: E-commerce - Détection Temps Réel

```bash
# Toutes les 5 minutes
python apply_automl_production.py \
  --dataset last_5min_orders.csv \
  --auto_match \
  --batch_size 10000 \
  --threshold 0.8 \
  --output realtime_alerts

# Résultat:
# → Commandes >80% bloquées automatiquement
# → Email envoyé au client pour vérification
```

### Cas 3: Assurance - Revue Hebdomadaire

```bash
# Tous les lundis
python apply_automl_production.py \
  --dataset claims_last_week.csv \
  --ensemble --top_k 5 \
  --anomaly_detection \
  --calibrate \
  --rich_export \
  --output weekly_review

# Résultat:
# → Excel envoyé aux investigateurs
# → Sheet "High Risk" = priorité 1
# → Anomalies = nouveaux patterns à analyser
```

---

## 🎓 Résumé Simple

### Sans Options (Mode Basique):
```bash
python apply_automl_production.py \
  --dataset data.csv \
  --auto_match
```
→ **1 modèle**, prédictions XGBoost simples

### Avec Ensemble:
```bash
--ensemble --top_k 3
```
→ **3 modèles**, moyenne pondérée, +15% robustesse

### Avec Anomaly Detection:
```bash
--anomaly_detection
```
→ Détecte **patterns nouveaux** jamais vus

### Avec Calibration:
```bash
--calibrate
```
→ Probabilités **plus fiables** (étirées)

### Avec Rich Export:
```bash
--rich_export
```
→ Excel **3 sheets** + JSON complet

### Tout Ensemble (Recommandé Production):
```bash
python apply_automl_production.py \
  --dataset production_data.csv \
  --ensemble --top_k 3 \
  --anomaly_detection \
  --calibrate \
  --rich_export \
  --output results
```
→ **Maximum précision + robustesse**

---

## ❓ FAQ Rapide

**Q: Quelle différence avec full_automl.py?**
- `full_automl.py` = **Entraînement** (apprend sur données étiquetées)
- `apply_automl_production.py` = **Prédiction** (applique sur données non-étiquetées)

**Q: Isolation Forest c'est obligatoire?**
- Non, mais **recommandé** pour détecter nouveaux patterns
- Ajoute +20% détection sur fraudes inédites

**Q: Ça prend combien de temps?**
- Mode simple: ~30 sec pour 100k transactions
- Mode ensemble: ~90 sec pour 100k transactions
- Mode batch: ~5 min pour 1M transactions

**Q: Quel seuil utiliser?**
- **Banking:** 0.3-0.4 (très sensible, minimiser pertes)
- **E-commerce:** 0.6-0.7 (équilibré, éviter faux positifs)
- **Insurance:** 0.5 (standard)

**Q: Je peux automatiser?**
- Oui! Cronjob ou Task Scheduler
- Exemple: `0 8 * * * python apply_automl_production.py ...`

---

## 🎯 En Résumé

**apply_automl_production.py** = Prendre un modèle déjà entraîné et l'appliquer sur de nouvelles données pour détecter les fraudes

**3 niveaux:**
1. **Simple:** XGBoost seul (rapide, basique)
2. **Avancé:** + Ensemble + Anomaly (précis, robuste)
3. **Expert:** + Calibration + Rich Export (production-ready)

**Résultat:** Liste de transactions frauduleuses avec leur probabilité, prête à être investigée! 🎉
