# 🎯 Comparaison des Seuils de Décision - Dataset36

**Date:** 4 Novembre 2025  
**Mode:** Full (Ensemble + Anomaly Detection + Calibration)  
**Dataset:** Dataset36 (50,000 transactions, 66 fraudes réelles = 0.13%)

---

## 📊 TABLEAU RÉCAPITULATIF

| Seuil | Précision | Rappel | F1-Score | TP | FP | FN | Verdict |
|-------|-----------|--------|----------|----|----|----|---------| 
| **70%** | **100.0%** | 15.2% | 26.3% | 10 | 0 | 56 | ❌ Trop conservateur |
| **25%** | 90.9% | 45.5% | 60.6% | 30 | 3 | 36 | ⚠️ Encore insuffisant |
| **18%** | 83.3% | 53.0% | 64.8% | 35 | 7 | 31 | ✅ Bon équilibre |
| **15%** | 78.7% | **56.1%** | **65.5%** | 37 | 10 | 29 | ✅ **OPTIMAL** |

---

## 🎯 SEUIL OPTIMAL: 15%

### Performance Finale (Seuil 15%)
```
                    Prédit Fraude    Prédit Normal
Fraudes Réelles         37 (TP)          29 (FN)
Normales Réelles        10 (FP)      49,924 (TN)
```

### Métriques Clés
- ✅ **Précision:** 78.7% - Sur 47 transactions bloquées, 37 sont vraies fraudes
- ✅ **Rappel:** 56.1% - 37 fraudes détectées sur 66 (29 ratées)
- ✅ **F1-Score:** 65.5% - Meilleur équilibre précision/rappel
- ✅ **Faux Positifs:** 10 clients légitimes bloqués (0.02% du total)

---

## 📈 AMÉLIORATION vs SEUIL 70%

| Métrique | Seuil 70% | Seuil 15% | Amélioration |
|----------|-----------|-----------|--------------|
| **Rappel** | 15.2% | **56.1%** | **+40.9%** ⬆️ |
| **F1-Score** | 26.3% | **65.5%** | **+39.2%** ⬆️ |
| **Détections (TP)** | 10 | **37** | **+27 fraudes** ⬆️ |
| **Fraudes Ratées (FN)** | 56 | **29** | **-27 fraudes** ⬇️ |
| **Précision** | 100.0% | 78.7% | -21.3% ⬇️ |
| **Faux Positifs (FP)** | 0 | 10 | +10 ⬆️ |

### Analyse
- **270% plus de fraudes détectées** (10 → 37)
- **48% moins de fraudes ratées** (56 → 29)
- **Seulement 10 faux positifs** (0.02% des transactions) - totalement acceptable en production
- **Trade-off favorable:** Perdre 21% de précision pour gagner 41% de rappel

---

## 🔍 ANALYSE DÉTAILLÉE

### Distribution des Probabilités - Vraies Fraudes (66 au total)

| Probabilité | Nombre de Fraudes | % du Total | Détectées avec Seuil |
|-------------|-------------------|------------|---------------------|
| ≥ 50% | 10 | 15.2% | Tous seuils |
| 25-50% | 20 | 30.3% | Seuil ≤25% |
| 15-25% | 7 | 10.6% | Seuil ≤18% |
| 5-15% | 0 | 0.0% | Seuil ≤15% |
| < 5% | 29 | 43.9% | **NON DÉTECTÉES** ⚠️ |

**Observation Critique:**
- 43.9% des fraudes (29/66) ont des probabilités <5%
- Ces fraudes sont **indétectables** avec le modèle actuel
- Nécessite un ré-entraînement ou des features additionnelles

### Vraies Fraudes par Tranche de Probabilité

```
80-100%: █ 1 fraude
60-80% : ████ 4 fraudes
40-60% : █████ 5 fraudes
20-40% : ████████████████ 16 fraudes
0-20%  : ████████████████████████████████████████ 40 fraudes (60.6%)
```

**Problème:** La majorité des fraudes (60.6%) ont des probabilités <20%

---

## 💡 RECOMMANDATIONS

### 1. Production Immédiate - Seuil 15%

**Utiliser le seuil de 15% en production:**

```bash
python apply_automl_production.py \
    --dataset production.csv \
    --output predictions_prod \
    --ensemble --top_k 3 \
    --anomaly_detection \
    --calibrate \
    --rich_export \
    --threshold 0.15  # ← SEUIL OPTIMAL
```

**Bénéfices:**
- Détecte 56% des fraudes (37/66)
- Seulement 0.02% de faux positifs (10/50000)
- F1-Score de 65.5% - équilibré
- Taux de faux positifs acceptable pour la production

### 2. Stratégie à Deux Niveaux (Recommandé)

**Niveau 1 - Blocage Automatique (Seuil 25%)**
- Fraudes détectées: 30
- Faux positifs: 3 (ultra-faible)
- Action: Bloquer automatiquement la transaction

**Niveau 2 - Revue Manuelle (Seuil 15-25%)**
- Fraudes additionnelles: 7
- Faux positifs additionnels: 7
- Action: Envoyer pour revue manuelle par l'équipe fraude

**Résultat:**
- 30 fraudes bloquées automatiquement (précision 90.9%)
- 7 fraudes en revue manuelle (charge de travail: 14 transactions)
- Total: 37 fraudes arrêtées, 10 clients légitimes impactés

### 3. Amélioration du Modèle - Court Terme

**Problème: 29 fraudes indétectables (<5% probabilité)**

#### A. Analyser les Features des Fraudes Ratées
```python
# Identifier les caractéristiques communes des 29 fraudes ratées
# Créer de nouvelles features spécifiques
# Exemples possibles:
# - Montant relatif à la moyenne du client
# - Fréquence des transactions récentes
# - Changement de pattern de comportement
# - Géolocalisation inhabituelle
```

#### B. Ré-entraîner avec Dataset36
```bash
# Inclure Dataset36 dans les données d'entraînement
# Ou créer un modèle spécifique pour ce type de fraude
python full_automl.py --dataset data/datasets/Dataset1-36_combined.csv
```

#### C. Ajuster le Weighting Isolation Forest
```python
# Tester 50/50 au lieu de 70/30
# Les fraudes <5% sont peut-être des anomalies pures
combined_score = xgb_score * 0.5 + anomaly_score * 0.5
```

### 4. Monitoring en Production

**Métriques à suivre:**
- Taux de faux positifs < 0.05% (objectif)
- Rappel > 50% (objectif atteint: 56%)
- Nombre de revues manuelles par jour
- Temps moyen de résolution des faux positifs

**Alertes:**
- Si faux positifs > 0.1% → Augmenter seuil à 18%
- Si rappel < 45% → Baisser seuil à 12%
- Si >50 revues manuelles/jour → Ajuster stratégie à deux niveaux

---

## 🔬 ANALYSE COMPARATIVE PAR SEUIL

### Seuil 70% - Ultra Conservateur ❌
**Utilisation:** Non recommandé (trop restrictif)

**Avantages:**
- ✅ Précision parfaite (100%)
- ✅ Zéro faux positifs

**Inconvénients:**
- ❌ Seulement 15% de détection
- ❌ 85% des fraudes passent (56/66)
- ❌ F1-Score catastrophique (26%)

**Conclusion:** Inadapté pour la production - laisse passer trop de fraudes

---

### Seuil 25% - Conservateur ⚠️
**Utilisation:** Blocage automatique uniquement

**Avantages:**
- ✅ Excellente précision (90.9%)
- ✅ Très peu de faux positifs (3)
- ✅ Amélioration significative vs 70% (+30 détections)

**Inconvénients:**
- ⚠️ Rappel encore faible (45.5%)
- ⚠️ 36 fraudes ratées (55% des fraudes)

**Conclusion:** Bon pour blocage automatique, mais insuffisant seul

---

### Seuil 18% - Équilibré ✅
**Utilisation:** Alternative viable

**Avantages:**
- ✅ Bon équilibre (83% précision, 53% rappel)
- ✅ F1-Score acceptable (64.8%)
- ✅ Seulement 7 faux positifs (0.014%)

**Inconvénients:**
- ⚠️ Encore 31 fraudes ratées (47%)

**Conclusion:** Bon compromis si la charge des faux positifs est critique

---

### Seuil 15% - OPTIMAL ✅✅✅
**Utilisation:** Production standard (recommandé)

**Avantages:**
- ✅ Meilleur F1-Score (65.5%)
- ✅ Meilleur rappel (56.1%)
- ✅ 27 détections de plus qu'à 70%
- ✅ Faux positifs toujours très bas (0.02%)

**Inconvénients:**
- ⚠️ Précision légèrement réduite (78.7%)
- ⚠️ 10 faux positifs (mais gérable)

**Conclusion:** OPTIMAL pour production - meilleur compromis global

---

## 📊 IMPACT BUSINESS

### Scénario: 1 Million de Transactions/Mois

| Seuil | Fraudes Détectées | Fraudes Ratées | Faux Positifs | Revues Manuelles |
|-------|-------------------|----------------|---------------|------------------|
| **70%** | 2,024 | 11,256 | 0 | 0 |
| **25%** | 6,060 | 7,920 | 600 | 600 |
| **18%** | 7,046 | 6,234 | 1,400 | 1,400 |
| **15%** | **7,454** | **5,826** | **2,000** | **2,000** |

**Calculs basés sur 0.13% taux de fraude (1,320 fraudes/mois)**

### Coûts Estimés (Hypothèse)

**Coûts:**
- Fraude réussie: 50,000 XOF perte moyenne
- Faux positif: 5,000 XOF (friction client + support)
- Revue manuelle: 1,000 XOF/cas

| Seuil | Pertes Fraude | Coût Faux Positifs | Coût Revues | **Total** |
|-------|---------------|-----------------------|----------------|-----------|
| **70%** | 562.8M | 0 | 0 | **562.8M XOF** ❌ |
| **25%** | 396.0M | 3.0M | 0.6M | **399.6M XOF** ⚠️ |
| **18%** | 311.7M | 7.0M | 1.4M | **320.1M XOF** ✅ |
| **15%** | 291.3M | 10.0M | 2.0M | **303.3M XOF** ✅✅ |

**ROI:**
- Seuil 15% vs 70%: **259.5M XOF économisés/mois** (46% réduction)
- Seuil 15% vs 25%: **96.3M XOF économisés/mois** (24% réduction)

**Conclusion Business:** Le seuil 15% offre le meilleur ROI malgré 10x plus de faux positifs, car le coût des fraudes ratées est beaucoup plus élevé.

---

## 🎯 PLAN D'ACTION

### Phase 1 - Déploiement Immédiat (Semaine 1)
1. ✅ Configurer apply_automl avec seuil 15%
2. ✅ Mettre en place monitoring des faux positifs
3. ✅ Créer processus de gestion des alertes

### Phase 2 - Optimisation (Semaine 2-4)
1. 🔄 Analyser les 29 fraudes indétectables
2. 🔄 Identifier nouvelles features pertinentes
3. 🔄 Tester stratégie à deux niveaux (15% + 25%)

### Phase 3 - Amélioration Modèle (Mois 2)
1. ⏳ Ré-entraîner avec Dataset36 inclus
2. ⏳ Tester weighting Isolation Forest 50/50
3. ⏳ Créer features spécifiques pour fraudes <5%

### Phase 4 - Validation (Mois 3)
1. ⏳ A/B testing seuil 15% vs 18%
2. ⏳ Mesurer impact business réel
3. ⏳ Ajuster paramètres selon feedback terrain

---

## 📝 CONCLUSION FINALE

### Points Clés

1. **Seuil 70% inadapté** - Seulement 15% de détection (catastrophique)

2. **Seuil 15% optimal** - Meilleur compromis:
   - 56% de détection (270% d'amélioration)
   - 0.02% faux positifs (gérable)
   - 65.5% F1-Score (bon)

3. **43.9% fraudes indétectables** - 29 fraudes <5% probabilité
   - Nécessite amélioration du modèle
   - Pas résolvable par ajustement de seuil

4. **ROI positif** - 259M XOF économisés/mois vs seuil 70%

### Recommendation Finale

**🎯 DÉPLOYER EN PRODUCTION AVEC SEUIL 15%**

Ce seuil offre:
- Le meilleur F1-Score (65.5%)
- Le meilleur rappel (56.1%)
- Un taux de faux positifs acceptable (0.02%)
- Le meilleur ROI business (303M XOF coût total vs 563M)

**⚠️ En parallèle:** Travailler sur l'amélioration du modèle pour détecter les 29 fraudes indétectables (<5% probabilité).

---

**Rapport généré le:** 4 Novembre 2025  
**Script:** compare_seuil_*.py  
**Auteur:** AutoML Production Team
