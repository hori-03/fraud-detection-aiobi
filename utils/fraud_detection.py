"""
fraud_detection.py

Objectif :
Utiliser une configuration XGBoost générée par le méta-modèle pour entraîner un modèle
de détection de fraude sur des données de transactions bancaires / mobile money.

Étapes prévues :
1. Charger un dataset de transactions (banque ou mobile money).
2. Charger la config JSON générée par le méta-modèle.
3. Prétraiter les données (nettoyage, encodage, normalisation).
4. Entraîner un modèle XGBoost avec cette configuration.
5. Évaluer les performances (précision, rappel, F1-score, taux de faux positifs).
6. Sauvegarder le modèle entraîné.
"""
