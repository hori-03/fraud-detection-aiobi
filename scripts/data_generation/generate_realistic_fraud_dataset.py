"""
Générateur de Dataset de Fraude Réaliste - Contexte Burkinabè
=============================================================

Génère des transactions avec historique temporel cohérent et patterns de fraude sophistiqués:
- Smurfing (multiples petites transactions)
- Comptes dormants réactivés
- Comportements inhabituels
- Concentration de bénéficiaires
- Bursts internationaux

Basé sur PROMPT_MOSTLY_AI_BURKINA.md

Usage:
    python generate_realistic_fraud_dataset.py --output data/datasets/Dataset9.csv --n_transactions 10000 --fraud_rate 0.08
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid
import argparse
from typing import Dict, List, Tuple
import random
from utils.column_matcher import ColumnMatcher


# ==================== CONFIGURATION BURKINA FASO ====================

BURKINA_CONFIG = {
    'regions': {
        'Centre': 0.25,
        'Hauts-Bassins': 0.18,
        'Sahel': 0.08,
        'Est': 0.07,
        'Nord': 0.07,
        'Centre-Nord': 0.06,
        'Centre-Ouest': 0.06,
        'Boucle du Mouhoun': 0.06,
        'Cascades': 0.05,
        'Centre-Est': 0.04,
        'Centre-Sud': 0.04,
        'Plateau-Central': 0.03,
        'Sud-Ouest': 0.01
    },
    
    'cities': [
        'Ouagadougou', 'Bobo-Dioulasso', 'Koudougou', 'Ouahigouya', 
        'Banfora', 'Dédougou', 'Kaya', 'Tenkodogo', 'Fada N\'gourma', 'Gaoua'
    ],
    
    'countries': {
        'BF': 0.75,   # Burkina Faso (local)
        'CI': 0.10,   # Côte d'Ivoire
        'ML': 0.05,   # Mali
        'GH': 0.03,   # Ghana
        'NG': 0.03,   # Nigeria
        'NE': 0.02,   # Niger
        'TG': 0.01,   # Togo
        'BJ': 0.01    # Bénin
    },
    
    'tx_methods': {
        'mobile_money': 0.40,
        'virement': 0.25,
        'especes': 0.15,
        'carte': 0.15,
        'atm': 0.05
    },
    
    'tx_purposes': {
        'achat': 0.30,
        'paiement': 0.25,
        'transfert_famille': 0.20,
        'retrait': 0.10,
        'salaire': 0.05,
        'autre': 0.10
    },
    
    'merchant_categories': {
        'telecommunication': 0.30,
        'alimentation': 0.15,
        'transport': 0.15,
        'commerce': 0.15,
        'services': 0.15,
        'autre': 0.10
    },
    
    'customer_types': {
        'particulier': 0.70,
        'entreprise': 0.20,
        'association': 0.08,
        'gouvernement': 0.02
    }
}


# ==================== FRAUD PATTERNS ====================

FRAUD_PATTERNS = {
    'smurfing': {
        'description': 'Multiples petites transactions totalisant un gros montant',
        'duration_days': (5, 10),
        'nb_transactions': (15, 30),
        'amount_range': (30000, 150000),  # XOF
        'total_target': (3000000, 8000000),  # 3M-8M XOF
        'timing': 'distributed',  # Étalé sur la journée
        'weight': 0.20  # 20% des fraudes
    },
    
    'dormant_account': {
        'description': 'Compte inactif réveillé avec grosse transaction',
        'dormant_days': (90, 365),
        'wake_amount': (500000, 5000000),
        'international': 0.70,  # 70% vers l'étranger
        'timing': 'night',  # Souvent la nuit
        'weight': 0.15
    },
    
    'unusual_behavior': {
        'description': 'Montant soudainement 10-50x supérieur à la normale',
        'multiplier': (10, 50),
        'normal_avg': (20000, 100000),
        'timing': 'night',
        'international': 0.50,
        'weight': 0.25
    },
    
    'beneficiary_concentration': {
        'description': 'Répétition vers 1-2 bénéficiaires seulement',
        'duration_days': (7, 21),
        'nb_transactions': (10, 25),
        'nb_beneficiaries': (1, 2),
        'total_target': (5000000, 20000000),
        'weight': 0.15
    },
    
    'international_burst': {
        'description': 'Soudain burst de transactions internationales',
        'duration_days': (3, 7),
        'nb_transactions': (5, 15),
        'countries': ['GH', 'NG', 'CI'],
        'amount_range': (200000, 2000000),
        'weight': 0.15
    },
    
    'high_velocity': {
        'description': 'Très haute fréquence de transactions en peu de temps',
        'duration_hours': (2, 6),
        'nb_transactions': (8, 20),
        'amount_range': (50000, 500000),
        'weight': 0.10
    }
}


# ==================== CLIENT PROFILE CLASS ====================

class ClientProfile:
    """Profil d'un client avec comportement normal et historique"""
    
    def __init__(self, client_id: str, distributions: dict = None):
        self.client_id = client_id
        self.distributions = distributions or {}
        
        # Caractéristiques démographiques
        self.type = np.random.choice(
            list(BURKINA_CONFIG['customer_types'].keys()),
            p=list(BURKINA_CONFIG['customer_types'].values())
        )
        self.region = np.random.choice(
            list(BURKINA_CONFIG['regions'].keys()),
            p=list(BURKINA_CONFIG['regions'].values())
        )
        self.age = int(np.random.normal(35, 12))
        self.age = np.clip(self.age, 18, 75)
        
        # Compte
        self.account_created = datetime(2024, 1, 1) - timedelta(days=int(np.random.exponential(365)))
        self.balance = float(np.random.lognormal(np.log(200000), 2.0))
        self.balance = np.clip(self.balance, 10000, 100000000)
        
        # Comportement normal (utiliser distributions personnalisées)
        amount_mean = self.distributions.get('amount_mean', 50000)
        amount_std = self.distributions.get('amount_std', 1.2)
        self.avg_tx_amount = float(np.random.lognormal(np.log(amount_mean), amount_std))
        
        self.tx_frequency = np.random.choice([0.5, 1, 2, 3, 5], p=[0.2, 0.3, 0.25, 0.15, 0.1])  # tx/jour
        self.preferred_method = np.random.choice(
            list(BURKINA_CONFIG['tx_methods'].keys()),
            p=list(BURKINA_CONFIG['tx_methods'].values())
        )
        
        international_rate = self.distributions.get('international_rate', 0.25)
        self.international_rate = np.random.beta(2, 20) * (international_rate / 0.10)  # Ajuster selon config
        self.international_rate = np.clip(self.international_rate, 0.0, 1.0)
        
        # Historique
        self.transactions = []
        self.last_transaction_date = None
        self.is_fraudster = False
        self.fraud_pattern = None
        self.fraud_start_date = None
        
    def generate_normal_transaction(self, date: datetime) -> Dict:
        """Génère une transaction normale basée sur le profil"""
        
        # Montant (variation autour de la moyenne)
        amount = float(np.random.lognormal(np.log(self.avg_tx_amount), 0.5))
        amount = np.clip(amount, 100, min(self.balance * 0.95, 10000000))
        
        # Vérifier solde suffisant
        if amount > self.balance:
            amount = self.balance * np.random.uniform(0.1, 0.5)
        
        # Destination
        is_international = np.random.random() < self.international_rate
        if is_international:
            dest_country = np.random.choice(['CI', 'ML', 'GH', 'NG', 'NE', 'TG', 'BJ'])
        else:
            dest_country = 'BF'
        
        # Heure (distribution bimodale: pics à 10h et 17h)
        night_rate = self.distributions.get('night_tx_rate', 0.05)
        if np.random.random() < night_rate:  # Nuit selon config
            hour = np.random.randint(0, 6)
        else:
            # Bimodal: matin (8-12h) et après-midi (15-19h)
            if np.random.random() < 0.5:
                hour = int(np.random.normal(10, 1.5))
            else:
                hour = int(np.random.normal(17, 1.5))
        hour = np.clip(hour, 0, 23)
        
        # Weekday
        weekday = date.weekday()
        
        # Processing time
        processing_time = int(np.random.lognormal(np.log(500), 0.8))
        processing_time = np.clip(processing_time, 100, 15000)
        
        # Transaction
        tx = {
            'tx_id': f'TX{len(self.transactions):06d}',
            'tx_timestamp': date.replace(hour=hour, minute=np.random.randint(0, 60), second=np.random.randint(0, 60)),
            'cust_id': self.client_id,
            'cust_type': self.type,
            'cust_region': self.region,
            'cust_age': self.age,
            'account_tenure_days': (date - self.account_created).days,
            'tx_amount_xof': amount,
            'tx_method': self.preferred_method if np.random.random() < 0.7 else np.random.choice(list(BURKINA_CONFIG['tx_methods'].keys())),
            'tx_purpose': np.random.choice(list(BURKINA_CONFIG['tx_purposes'].keys()), p=list(BURKINA_CONFIG['tx_purposes'].values())),
            'merchant_category': np.random.choice(list(BURKINA_CONFIG['merchant_categories'].keys()), p=list(BURKINA_CONFIG['merchant_categories'].values())),
            'dest_country': dest_country,
            'hour': hour,
            'weekday': weekday,
            'balance_before': self.balance,
            'balance_after': self.balance - amount,
            'processing_time_ms': processing_time,
            'fraud_flag': 0
        }
        
        # Mise à jour état
        self.balance -= amount
        self.last_transaction_date = date
        self.transactions.append(tx)
        
        return tx
    
    def generate_normal_history(self, start_date: datetime, end_date: datetime):
        """Génère l'historique normal sur une période"""
        current_date = start_date
        
        while current_date < end_date:
            # Probabilité de transaction ce jour
            if np.random.random() < self.tx_frequency / 10:  # Ajusté pour avoir des jours sans transaction
                nb_tx_today = np.random.poisson(self.tx_frequency)
                nb_tx_today = min(nb_tx_today, 5)  # Max 5 tx/jour
                
                for _ in range(nb_tx_today):
                    if self.balance > 1000:  # Solde minimum
                        self.generate_normal_transaction(current_date)
                        
                        # Recharge aléatoire du compte
                        if np.random.random() < 0.2:  # 20% chance de recharge
                            recharge = float(np.random.lognormal(np.log(self.avg_tx_amount * 3), 1.0))
                            self.balance += recharge
            
            current_date += timedelta(days=1)


# ==================== FRAUD PATTERN INJECTOR ====================

class FraudPatternInjector:
    """Injecte des patterns de fraude réalistes dans l'historique d'un client"""
    
    @staticmethod
    def inject_smurfing(client: ClientProfile, fraud_date: datetime):
        """Smurfing: multiples petites transactions"""
        pattern = FRAUD_PATTERNS['smurfing']
        duration = np.random.randint(*pattern['duration_days'])
        nb_tx = np.random.randint(*pattern['nb_transactions'])
        
        # Répartir les transactions sur la période
        for i in range(nb_tx):
            day_offset = np.random.randint(0, duration)
            tx_date = fraud_date + timedelta(days=day_offset)
            
            amount = np.random.uniform(*pattern['amount_range'])
            
            # Vérifier solde
            if amount > client.balance:
                client.balance += amount * 2  # Recharge pour continuer le smurfing
            
            # Transaction frauduleuse
            hour = np.random.randint(8, 20)  # Étalé sur la journée
            
            tx = {
                'tx_id': f'TX{len(client.transactions):06d}',
                'tx_timestamp': tx_date.replace(hour=hour, minute=np.random.randint(0, 60)),
                'cust_id': client.client_id,
                'cust_type': client.type,
                'cust_region': client.region,
                'cust_age': client.age,
                'account_tenure_days': (tx_date - client.account_created).days,
                'tx_amount_xof': amount,
                'tx_method': 'mobile_money',  # Smurfing souvent via mobile money
                'tx_purpose': 'transfert_famille',
                'merchant_category': 'telecommunication',
                'dest_country': np.random.choice(['BF', 'CI', 'ML']),
                'hour': hour,
                'weekday': tx_date.weekday(),
                'balance_before': client.balance,
                'balance_after': client.balance - amount,
                'processing_time_ms': np.random.randint(300, 1500),
                'fraud_flag': 1
            }
            
            client.balance -= amount
            client.transactions.append(tx)
    
    @staticmethod
    def inject_dormant_account(client: ClientProfile, fraud_date: datetime):
        """Compte dormant réveillé"""
        pattern = FRAUD_PATTERNS['dormant_account']
        
        # Créer période dormante (pas de transactions)
        dormant_days = np.random.randint(*pattern['dormant_days'])
        client.last_transaction_date = fraud_date - timedelta(days=dormant_days)
        
        # Transaction de réveil (grosse somme)
        amount = np.random.uniform(*pattern['wake_amount'])
        
        # Recharger le compte si nécessaire
        if amount > client.balance:
            client.balance = amount * 1.5
        
        is_international = np.random.random() < pattern['international']
        dest_country = np.random.choice(['GH', 'NG', 'CI']) if is_international else 'BF'
        
        # Souvent la nuit
        hour = np.random.randint(0, 6) if np.random.random() < 0.6 else np.random.randint(22, 24)
        
        tx = {
            'tx_id': f'TX{len(client.transactions):06d}',
            'tx_timestamp': fraud_date.replace(hour=hour, minute=np.random.randint(0, 60)),
            'cust_id': client.client_id,
            'cust_type': client.type,
            'cust_region': client.region,
            'cust_age': client.age,
            'account_tenure_days': (fraud_date - client.account_created).days,
            'tx_amount_xof': amount,
            'tx_method': 'virement',
            'tx_purpose': 'retrait',
            'merchant_category': 'services',
            'dest_country': dest_country,
            'hour': hour,
            'weekday': fraud_date.weekday(),
            'balance_before': client.balance,
            'balance_after': client.balance - amount,
            'processing_time_ms': np.random.randint(500, 3000),
            'fraud_flag': 1
        }
        
        client.balance -= amount
        client.transactions.append(tx)
    
    @staticmethod
    def inject_unusual_behavior(client: ClientProfile, fraud_date: datetime):
        """Comportement inhabituel (montant anormal)"""
        pattern = FRAUD_PATTERNS['unusual_behavior']
        
        # Montant beaucoup plus élevé que la normale
        multiplier = np.random.uniform(*pattern['multiplier'])
        amount = client.avg_tx_amount * multiplier
        amount = np.clip(amount, 500000, 5000000)
        
        # Recharger si nécessaire
        if amount > client.balance:
            client.balance = amount * 1.5
        
        is_international = np.random.random() < pattern['international']
        dest_country = np.random.choice(['GH', 'NG']) if is_international else 'BF'
        
        # Souvent la nuit
        hour = np.random.randint(0, 6) if np.random.random() < 0.5 else np.random.randint(8, 20)
        
        tx = {
            'tx_id': f'TX{len(client.transactions):06d}',
            'tx_timestamp': fraud_date.replace(hour=hour, minute=np.random.randint(0, 60)),
            'cust_id': client.client_id,
            'cust_type': client.type,
            'cust_region': client.region,
            'cust_age': client.age,
            'account_tenure_days': (fraud_date - client.account_created).days,
            'tx_amount_xof': amount,
            'tx_method': np.random.choice(['virement', 'carte', 'mobile_money']),
            'tx_purpose': 'transfert_famille',
            'merchant_category': 'commerce',
            'dest_country': dest_country,
            'hour': hour,
            'weekday': fraud_date.weekday(),
            'balance_before': client.balance,
            'balance_after': client.balance - amount,
            'processing_time_ms': np.random.randint(800, 5000),
            'fraud_flag': 1
        }
        
        client.balance -= amount
        client.transactions.append(tx)
    
    @staticmethod
    def inject_beneficiary_concentration(client: ClientProfile, fraud_date: datetime):
        """Répétition vers même bénéficiaire"""
        pattern = FRAUD_PATTERNS['beneficiary_concentration']
        
        duration = np.random.randint(*pattern['duration_days'])
        nb_tx = np.random.randint(*pattern['nb_transactions'])
        nb_beneficiaries = np.random.randint(*pattern['nb_beneficiaries'])
        
        beneficiaries = [f'BEN{i:03d}' for i in range(nb_beneficiaries)]
        
        for i in range(nb_tx):
            day_offset = np.random.randint(0, duration)
            tx_date = fraud_date + timedelta(days=day_offset)
            
            amount = np.random.uniform(200000, 1000000)
            
            if amount > client.balance:
                client.balance += amount * 2
            
            tx = {
                'tx_id': f'TX{len(client.transactions):06d}',
                'tx_timestamp': tx_date.replace(hour=np.random.randint(8, 20), minute=np.random.randint(0, 60)),
                'cust_id': client.client_id,
                'cust_type': client.type,
                'cust_region': client.region,
                'cust_age': client.age,
                'account_tenure_days': (tx_date - client.account_created).days,
                'tx_amount_xof': amount,
                'tx_method': 'virement',
                'tx_purpose': 'paiement',
                'merchant_category': 'commerce',
                'dest_country': np.random.choice(['BF', 'CI']),
                'hour': tx_date.hour,
                'weekday': tx_date.weekday(),
                'balance_before': client.balance,
                'balance_after': client.balance - amount,
                'processing_time_ms': np.random.randint(400, 2000),
                'fraud_flag': 1
            }
            
            client.balance -= amount
            client.transactions.append(tx)
    
    @staticmethod
    def inject_international_burst(client: ClientProfile, fraud_date: datetime):
        """Burst de transactions internationales"""
        pattern = FRAUD_PATTERNS['international_burst']
        
        duration = np.random.randint(*pattern['duration_days'])
        nb_tx = np.random.randint(*pattern['nb_transactions'])
        
        for i in range(nb_tx):
            day_offset = np.random.randint(0, duration)
            tx_date = fraud_date + timedelta(days=day_offset)
            
            amount = np.random.uniform(*pattern['amount_range'])
            
            if amount > client.balance:
                client.balance += amount * 2
            
            dest_country = np.random.choice(pattern['countries'])
            
            tx = {
                'tx_id': f'TX{len(client.transactions):06d}',
                'tx_timestamp': tx_date.replace(hour=np.random.randint(0, 24), minute=np.random.randint(0, 60)),
                'cust_id': client.client_id,
                'cust_type': client.type,
                'cust_region': client.region,
                'cust_age': client.age,
                'account_tenure_days': (tx_date - client.account_created).days,
                'tx_amount_xof': amount,
                'tx_method': np.random.choice(['virement', 'carte']),
                'tx_purpose': 'transfert_famille',
                'merchant_category': 'services',
                'dest_country': dest_country,
                'hour': tx_date.hour,
                'weekday': tx_date.weekday(),
                'balance_before': client.balance,
                'balance_after': client.balance - amount,
                'processing_time_ms': np.random.randint(1000, 8000),
                'fraud_flag': 1
            }
            
            client.balance -= amount
            client.transactions.append(tx)
    
    @staticmethod
    def inject_high_velocity(client: ClientProfile, fraud_date: datetime):
        """Haute vélocité (beaucoup de transactions en peu de temps)"""
        pattern = FRAUD_PATTERNS['high_velocity']
        
        duration_hours = np.random.uniform(*pattern['duration_hours'])
        nb_tx = np.random.randint(*pattern['nb_transactions'])
        
        start_hour = np.random.randint(0, 24 - int(duration_hours))
        
        for i in range(nb_tx):
            # Répartir sur les heures
            hour_offset = (i / nb_tx) * duration_hours
            tx_hour = int(start_hour + hour_offset)
            tx_minute = np.random.randint(0, 60)
            
            amount = np.random.uniform(*pattern['amount_range'])
            
            if amount > client.balance:
                client.balance += amount * 3  # Grosse recharge pour continuer
            
            tx = {
                'tx_id': f'TX{len(client.transactions):06d}',
                'tx_timestamp': fraud_date.replace(hour=tx_hour, minute=tx_minute),
                'cust_id': client.client_id,
                'cust_type': client.type,
                'cust_region': client.region,
                'cust_age': client.age,
                'account_tenure_days': (fraud_date - client.account_created).days,
                'tx_amount_xof': amount,
                'tx_method': 'mobile_money',
                'tx_purpose': 'achat',
                'merchant_category': np.random.choice(['telecommunication', 'commerce']),
                'dest_country': 'BF',
                'hour': tx_hour,
                'weekday': fraud_date.weekday(),
                'balance_before': client.balance,
                'balance_after': client.balance - amount,
                'processing_time_ms': np.random.randint(100, 800),  # Très rapide
                'fraud_flag': 1
            }
            
            client.balance -= amount
            client.transactions.append(tx)


# ==================== DATASET GENERATOR ====================

class DatasetGenerator:
    """Générateur principal de dataset"""
    
    def __init__(self, n_transactions: int = 10000, fraud_rate: float = 0.08, seed: int = 42, config: dict = None):
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        self.seed = seed
        self.config = config or {}
        
        np.random.seed(seed)
        random.seed(seed)
        
        self.clients = []
        self.all_transactions = []
        self.column_matcher = ColumnMatcher(fuzzy_threshold=0.7)
        
        # Extraire les configurations spécifiques
        self.fraud_pattern_weights = self.config.get('fraud_pattern_weights', None)
        self.distributions = self.config.get('distributions', {})
        self.fraud_multipliers = self.config.get('fraud_multipliers', {})
    
    def generate(self) -> pd.DataFrame:
        """Génère le dataset complet"""
        
        print(f"🚀 Génération de {self.n_transactions} transactions (fraud_rate={self.fraud_rate*100:.1f}%)")
        
        # Étape 1: Créer les profils clients
        n_clients = max(500, self.n_transactions // 15)  # ~15 tx par client en moyenne
        print(f"📊 Création de {n_clients} profils clients...")
        
        for i in range(n_clients):
            client_id = f'CUST{i:05d}'
            client = ClientProfile(client_id, distributions=self.distributions)
            self.clients.append(client)
        
        # Étape 2: Générer historique normal (120 jours)
        print(f"📅 Génération de l'historique normal (120 jours)...")
        start_date = datetime(2024, 1, 1)
        fraud_injection_date = datetime(2024, 4, 1)  # Fraudes à partir d'avril
        end_date = datetime(2024, 5, 1)  # Dataset final = mai 2024
        
        for client in self.clients:
            client.generate_normal_history(start_date, fraud_injection_date)
        
        # Étape 3: Sélectionner clients fraudeurs
        # CORRECTION: Calculer basé sur le nombre TARGET de transactions frauduleuses
        target_fraud_transactions = int(self.n_transactions * self.fraud_rate)
        # Estimer ~5 transactions de fraude par client fraudeur (moyenne empirique)
        avg_fraud_tx_per_client = 5
        n_fraud_clients = max(1, int(target_fraud_transactions / avg_fraud_tx_per_client))
        # Ne pas dépasser le nombre de clients disponibles
        n_fraud_clients = min(n_fraud_clients, len(self.clients))
        
        print(f"🚨 Sélection de {n_fraud_clients} clients fraudeurs (pour ~{target_fraud_transactions} transactions frauduleuses)...")
        
        fraud_clients = np.random.choice(self.clients, n_fraud_clients, replace=False)
        
        # Étape 4: Injecter patterns de fraude
        print(f"💉 Injection des patterns de fraude...")
        
        # Utiliser poids personnalisés si fournis, sinon défauts
        if self.fraud_pattern_weights:
            pattern_names = list(self.fraud_pattern_weights.keys())
            pattern_weights = [self.fraud_pattern_weights[p] for p in pattern_names]
        else:
            pattern_names = list(FRAUD_PATTERNS.keys())
            pattern_weights = [FRAUD_PATTERNS[p]['weight'] for p in pattern_names]
        
        for client in fraud_clients:
            client.is_fraudster = True
            
            # Choisir un pattern de fraude
            pattern = np.random.choice(pattern_names, p=pattern_weights)
            client.fraud_pattern = pattern
            
            # Date de début de fraude (entre avril et mai)
            fraud_start = fraud_injection_date + timedelta(days=np.random.randint(0, 30))
            client.fraud_start_date = fraud_start
            
            # Injecter le pattern
            if pattern == 'smurfing':
                FraudPatternInjector.inject_smurfing(client, fraud_start)
            elif pattern == 'dormant_account':
                FraudPatternInjector.inject_dormant_account(client, fraud_start)
            elif pattern == 'unusual_behavior':
                FraudPatternInjector.inject_unusual_behavior(client, fraud_start)
            elif pattern == 'beneficiary_concentration':
                FraudPatternInjector.inject_beneficiary_concentration(client, fraud_start)
            elif pattern == 'international_burst':
                FraudPatternInjector.inject_international_burst(client, fraud_start)
            elif pattern == 'high_velocity':
                FraudPatternInjector.inject_high_velocity(client, fraud_start)
        
        # Étape 5: Continuer historique normal après fraudes
        print(f"📆 Continuation de l'historique normal post-fraude...")
        for client in self.clients:
            if client.transactions:  # Si le client a déjà des transactions
                last_tx_date = client.transactions[-1]['tx_timestamp']
                if last_tx_date < end_date:
                    client.generate_normal_history(last_tx_date + timedelta(days=1), end_date)
        
        # Étape 6: Collecter toutes les transactions
        print(f"📦 Collection de toutes les transactions...")
        for client in self.clients:
            self.all_transactions.extend(client.transactions)
        
        # Étape 7: Sélectionner snapshot final
        print(f"🎯 Sélection du snapshot final ({self.n_transactions} transactions)...")
        
        # Trier par date
        self.all_transactions.sort(key=lambda x: x['tx_timestamp'])
        
        # Prendre les dernières transactions
        if len(self.all_transactions) > self.n_transactions:
            selected_transactions = self.all_transactions[-self.n_transactions:]
        else:
            selected_transactions = self.all_transactions
        
        # Créer DataFrame
        df = pd.DataFrame(selected_transactions)
        
        # Réassigner tx_id séquentiels
        df['tx_id'] = [f'TX{i:06d}' for i in range(len(df))]
        
        # Formater timestamp
        df['tx_timestamp'] = df['tx_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Arrondir les floats
        df['tx_amount_xof'] = df['tx_amount_xof'].round(2)
        df['balance_before'] = df['balance_before'].round(2)
        df['balance_after'] = df['balance_after'].round(2)
        
        # Étape 7.5: Appliquer column_overrides (REMPLACER contenu des colonnes)
        column_overrides = self.config.get('column_overrides', {})
        if column_overrides:
            print(f"🔄 Application de {len(column_overrides)} remplacements de colonnes...")
            for new_col_name, override_spec in column_overrides.items():
                old_col_name = override_spec['replaces']
                if old_col_name in df.columns:
                    print(f"   ↪ {old_col_name} → {new_col_name} (valeurs: {len(override_spec['values'])} catégories)")
                    # Créer nouvelle colonne avec distribution différente
                    df[new_col_name] = np.random.choice(
                        override_spec['values'],
                        size=len(df),
                        p=override_spec.get('weights', None)
                    )
                    # Supprimer ancienne colonne
                    df.drop(columns=[old_col_name], inplace=True)
        
        # Étape 7.6: Supprimer colonnes non désirées (si configuré)
        remove_columns = self.config.get('remove_columns', [])
        if remove_columns:
            existing_to_remove = [col for col in remove_columns if col in df.columns]
            if existing_to_remove:
                print(f"🗑️  Suppression de {len(existing_to_remove)} colonnes: {existing_to_remove}")
                df.drop(columns=existing_to_remove, inplace=True)
        
        # Étape 8: Ajouter colonnes extra (si configurées)
        columns_extra = self.config.get('columns_extra', {})
        if columns_extra:
            print(f"📋 Ajout de {len(columns_extra)} colonnes supplémentaires...")
            for col_name, col_spec in columns_extra.items():
                if col_spec['type'] == 'categorical':
                    if 'generator' in col_spec:
                        df[col_name] = [col_spec['generator']() for _ in range(len(df))]
                    else:
                        df[col_name] = np.random.choice(
                            col_spec['values'],
                            size=len(df),
                            p=col_spec.get('weights', None)
                        )
                elif col_spec['type'] == 'numeric':
                    df[col_name] = [col_spec['generator']() for _ in range(len(df))]
        
        # Étape 9: Appliquer column_mapping (renommer colonnes)
        column_mapping = self.config.get('column_mapping', {})
        if column_mapping:
            print(f"🔄 Renommage de {len(column_mapping)} colonnes...")
            df.rename(columns=column_mapping, inplace=True)
        
        # Étape 10: Séparer timestamp si demandé
        if self.config.get('split_timestamp', False):
            timestamp_col = column_mapping.get('tx_timestamp', 'tx_timestamp')
            if timestamp_col in df.columns:
                print(f"📅 Séparation de {timestamp_col} en date + heure...")
                df['date_transaction'] = pd.to_datetime(df[timestamp_col]).dt.date
                df['heure_transaction'] = pd.to_datetime(df[timestamp_col]).dt.time
                df.drop(columns=[timestamp_col], inplace=True)
        
        # Étape 11: Ajouter méta-features (features dérivées utiles pour ML)
        print(f"🧮 Ajout des méta-features...")
        df = self._add_meta_features(df, column_mapping)
        
        # Étape 12: Afficher statistiques détaillées
        self._print_generation_statistics(df, column_mapping, fraud_clients)
        
        return df
    
    def _add_meta_features(self, df, column_mapping):
        """Ajoute des features dérivées utiles pour le ML"""
        
        # 1. Flag devise étrangère (si colonne *_currency existe)
        currency_cols = [c for c in df.columns if 'currency' in c.lower()]
        if currency_cols:
            currency_col = currency_cols[0]
            df['is_foreign_currency'] = (df[currency_col] != 'FCFA').astype(int)
            print(f"   ✓ is_foreign_currency (basé sur {currency_col})")
        
        # 2. Montant Z-score (standardisé)
        amount_cols = [c for c in df.columns if 'amount' in c.lower() and 'fcfa' in c.lower()]
        if not amount_cols:
            amount_cols = [c for c in df.columns if 'amount' in c.lower()]
        
        if amount_cols:
            amount_col = amount_cols[0]
            mean_amount = df[amount_col].mean()
            std_amount = df[amount_col].std()
            
            if std_amount > 0:
                df['amount_zscore'] = (df[amount_col] - mean_amount) / std_amount
                df['is_high_amount'] = (df['amount_zscore'] > 2).astype(int)
                print(f"   ✓ amount_zscore, is_high_amount (basé sur {amount_col})")
        
        # 3. Ratio montant/balance (si balance existe)
        if 'balance_before' in df.columns and amount_cols:
            df['amount_to_balance_ratio'] = df[amount_cols[0]] / (df['balance_before'] + 1)
            df['depletes_balance'] = (df['amount_to_balance_ratio'] > 0.8).astype(int)
            print(f"   ✓ amount_to_balance_ratio, depletes_balance")
        
        # 4. Flag weekend (si weekday existe)
        if 'weekday' in df.columns:
            df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
            print(f"   ✓ is_weekend")
        
        # 5. Flag heures anormales (si hour existe)
        if 'hour' in df.columns:
            df['is_night_tx'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
            print(f"   ✓ is_night_tx, is_business_hours")
        
        # 6. Flag international (si dest_country existe)
        if 'dest_country' in df.columns:
            df['is_international'] = (df['dest_country'] != 'BF').astype(int)
            print(f"   ✓ is_international")
        
        return df
    
    def _print_generation_statistics(self, df, column_mapping, fraud_clients):
        """Affiche statistiques détaillées post-génération"""
        
        target_col = column_mapping.get('fraud_flag', 'fraud_flag')
        fraud_count = df[target_col].sum()
        fraud_rate_final = fraud_count / len(df)
        
        print(f"\n" + "="*80)
        print(f"STATISTIQUES DE GÉNÉRATION")
        print("="*80)
        
        # Stats globales
        print(f"\nTransactions: {len(df):,}")
        print(f"Colonnes: {len(df.columns)}")
        print(f"Fraudes: {fraud_count:,} ({fraud_rate_final*100:.2f}%)")
        
        cust_col = column_mapping.get('cust_id', 'cust_id')
        if cust_col in df.columns:
            print(f"Clients uniques: {df[cust_col].nunique():,}")
        
        # Stats par devise (si colonne currency existe)
        currency_cols = [c for c in df.columns if 'currency' in c.lower()]
        if currency_cols:
            currency_col = currency_cols[0]
            print(f"\nRepartition par devise ({currency_col}):")
            currency_dist = df[currency_col].value_counts(normalize=True).sort_index()
            for curr in currency_dist.index:
                pct = currency_dist[curr]
                fraud_mask = df[currency_col] == curr
                fraud_rate_curr = df[fraud_mask][target_col].mean() if fraud_mask.sum() > 0 else 0
                count = df[fraud_mask].shape[0]
                print(f"  {curr:4s}: {count:7,} tx ({pct*100:5.1f}%) | Fraude: {fraud_rate_curr*100:5.2f}%")
        
        # Stats de montants
        amount_cols = [c for c in df.columns if 'amount' in c.lower() and 'fcfa' in c.lower()]
        if not amount_cols:
            amount_cols = [c for c in df.columns if 'amount' in c.lower()]
        
        if amount_cols:
            amount_col = amount_cols[0]
            print(f"\nDistribution des montants ({amount_col}):")
            print(f"  Min:    {df[amount_col].min():>15,.0f}")
            print(f"  Mean:   {df[amount_col].mean():>15,.0f}")
            print(f"  Median: {df[amount_col].median():>15,.0f}")
            print(f"  P95:    {df[amount_col].quantile(0.95):>15,.0f}")
            print(f"  Max:    {df[amount_col].max():>15,.0f}")
        
        # Patterns de fraude
        if fraud_count > 0 and len(fraud_clients) > 0:
            print(f"\nPatterns de fraude:")
            pattern_counts = {}
            for client in fraud_clients:
                if hasattr(client, 'fraud_pattern') and client.fraud_pattern:
                    pattern_counts[client.fraud_pattern] = pattern_counts.get(client.fraud_pattern, 0) + 1
            
            for pattern in sorted(pattern_counts.keys()):
                count = pattern_counts[pattern]
                print(f"  {pattern:30s}: {count:5d} clients ({count/len(fraud_clients)*100:5.1f}%)")
        
        print("="*80 + "\n")


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='Génère un dataset de fraude réaliste (contexte Burkinabè)')
    parser.add_argument('--dataset', type=str, default='Dataset9', help='Nom du dataset (Dataset9-19) ou custom')
    parser.add_argument('--output', type=str, default=None, help='Fichier de sortie (auto si non spécifié)')
    parser.add_argument('--n_transactions', type=int, default=None, help='Nombre de transactions (override config)')
    parser.add_argument('--fraud_rate', type=float, default=None, help='Taux de fraude (override config)')
    parser.add_argument('--seed', type=int, default=42, help='Seed pour reproductibilité')
    parser.add_argument('--list', action='store_true', help='Lister toutes les configurations')
    
    args = parser.parse_args()
    
    # Lister les configs si demandé
    if args.list:
        from dataset_configs import print_all_configs
        print_all_configs()
        return
    
    # Charger la configuration
    try:
        from dataset_configs import get_config
        config = get_config(args.dataset)
        print(f"\n📋 Configuration chargée: {args.dataset}")
        print(f"   {config['description']}")
    except (ImportError, ValueError) as e:
        print(f"⚠️  Configuration non trouvée, utilisation des paramètres par défaut")
        config = {
            'n_transactions': 10000,
            'fraud_rate': 0.08,
            'difficulty': 'medium',
            'fraud_pattern_weights': None,
            'distributions': None,
            'fraud_multipliers': None
        }
    
    # Override avec les arguments CLI si fournis
    if args.n_transactions is not None:
        config['n_transactions'] = args.n_transactions
    if args.fraud_rate is not None:
        config['fraud_rate'] = args.fraud_rate
    
    # Déterminer le fichier de sortie
    if args.output is None:
        args.output = f'data/datasets/{args.dataset}.csv'
    
    # Déterminer le fichier de sortie
    if args.output is None:
        args.output = f'data/datasets/{args.dataset}.csv'
    
    # Générer le dataset
    generator = DatasetGenerator(
        n_transactions=config['n_transactions'],
        fraud_rate=config['fraud_rate'],
        seed=args.seed,
        config=config  # Passer toute la config
    )
    
    df = generator.generate()
    
    # Sauvegarder
    print(f"\n💾 Sauvegarde dans {args.output}...")
    df.to_csv(args.output, index=False)
    
    print(f"✅ Dataset sauvegardé avec succès!")
    print(f"\n🎯 Prochaines étapes:")
    print(f"   1. Tester avec: python automl_transformer/full_automl.py {args.output} fraud_flag")
    print(f"   2. Vérifier F1 score (difficulté: {config.get('difficulty', 'N/A')})")
    print(f"   3. Analyser les patterns détectés")


if __name__ == '__main__':
    main()
