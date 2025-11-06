"""
Configuration des Datasets 9-30
================================

Définit les caractéristiques de chaque dataset pour créer une diversité maximale
pour l'entraînement du Meta-Transformer.

Datasets 9-19: Orientation Mobile Money Burkina Faso
Datasets 20-30: Orientation Banking (structures très variées)
"""

import numpy as np
import random

DATASET_CONFIGS = {
    'Dataset9': {
        'n_transactions': 10000,
        'fraud_rate': 0.08,
        'difficulty': 'easy',
        'description': 'BF_Small_Easy - Patterns évidents',
        'column_structure': 'structure_A',  # Structure de base
        'columns_extra': {},
        'column_overrides': {},  # Pas de changements pour Dataset9 (baseline)
        'fraud_pattern_weights': {
            'smurfing': 0.20,
            'dormant_account': 0.15,
            'unusual_behavior': 0.25,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.15,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 50000,
            'amount_std': 1.8,
            'international_rate': 0.25,
            'night_tx_rate': 0.05
        }
    },
    
    'Dataset10': {
        'n_transactions': 12000,
        'fraud_rate': 0.15,
        'difficulty': 'very_easy',
        'description': 'BF_Small_Rich - Beaucoup de features, fraud fréquent',
        'column_structure': 'structure_B',  # Structure différente
        'column_mapping': {
            # Noms de colonnes différents de Dataset9
            'tx_id': 'transaction_id',
            'cust_id': 'customer_id',
            'cust_type': 'customer_type',
            'cust_region': 'zone',  # Agrégé en 5 zones au lieu de 13 régions
            'cust_age': 'age',
            'tx_amount_xof': 'amount',
            'tx_method': 'payment_method',
            'tx_purpose': 'transaction_purpose',
            'dest_country': 'destination',
            'fraud_flag': 'is_fraud'
        },
        'column_overrides': {
            # CHANGEMENT DE CONTENU, pas juste renommage
            'zone': {
                'type': 'categorical',
                'values': ['Centre', 'Ouest', 'Est', 'Nord', 'Sud'],  # 5 zones agrégées au lieu de 13 régions
                'weights': [0.40, 0.20, 0.15, 0.15, 0.10],
                'replaces': 'cust_region'
            },
            'payment_method': {
                'type': 'categorical',
                'values': ['Mobile Money', 'Virement Bancaire', 'Espèces', 'Carte Bancaire', 'Distributeur'],
                'weights': [0.45, 0.25, 0.12, 0.13, 0.05],
                'replaces': 'tx_method'
            },
            'transaction_purpose': {
                'type': 'categorical',
                'values': ['Achat Marchandises', 'Paiement Facture', 'Transfert Familial', 'Retrait Espèces', 'Salaire/Revenu', 'Épargne', 'Autre'],
                'weights': [0.28, 0.22, 0.20, 0.12, 0.08, 0.05, 0.05],
                'replaces': 'tx_purpose'
            }
        },
        'columns_extra': {
            'merchant_id': {
                'type': 'categorical',
                'generator': lambda: f'MERCH{np.random.randint(1, 500):04d}',
                'description': 'ID du marchand (disponible au moment T)'
            },
            'device_type': {
                'type': 'categorical',
                'values': ['Smartphone', 'Ordinateur', 'Tablette', 'GAB', 'Agence'],
                'weights': [0.45, 0.20, 0.10, 0.10, 0.15],
                'description': 'Type d\'appareil (disponible au moment T)'
            },
            'ip_country': {
                'type': 'categorical',
                'values': ['BF', 'CI', 'ML', 'FR', 'US', 'INCONNU'],
                'weights': [0.70, 0.10, 0.08, 0.05, 0.02, 0.05],
                'description': 'Pays de l\'IP (disponible au moment T)'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.30,  # Plus de smurfing
            'dormant_account': 0.05,
            'unusual_behavior': 0.35,  # Très inhabituel
            'beneficiary_concentration': 0.10,
            'international_burst': 0.15,
            'high_velocity': 0.05
        },
        'distributions': {
            'amount_mean': 80000,  # Montants plus élevés
            'amount_std': 2.2,
            'international_rate': 0.35,  # Plus d'international
            'night_tx_rate': 0.15  # Plus de nuit (suspect)
        },
        'fraud_multipliers': {
            'amount_high': 8.0,  # Très fort (easy)
            'night': 5.0,
            'ratio_account': 6.0,
            'new_account': 3.0
        }
    },
    
    'Dataset11': {
        'n_transactions': 15000,
        'fraud_rate': 0.10,
        'difficulty': 'medium',
        'description': 'BF_Medium_Balanced - Équilibré, patterns clairs',
        'column_structure': 'structure_C',  # Structure encore plus différente
        'column_mapping': {
            # Structure encore différente
            'tx_timestamp': 'date_heure',
            'tx_id': 'id_transaction',
            'cust_id': 'client_id',
            'cust_type': 'type_client',
            'tx_amount_xof': 'montant_fcfa',
            'tx_method': 'code_methode',  # Codes au lieu de noms
            'tx_purpose': 'motif',
            'merchant_category': 'secteur_activite',
            'fraud_flag': 'fraude'
        },
        'column_overrides': {
            # Codes courts au lieu de noms longs
            'code_methode': {
                'type': 'categorical',
                'values': ['MM', 'VB', 'ES', 'CB', 'DA'],  # Mobile Money, Virement, Espèces, Carte, Distributeur
                'weights': [0.40, 0.25, 0.15, 0.15, 0.05],
                'replaces': 'tx_method'
            },
            'secteur_activite': {
                'type': 'categorical',
                'values': ['TEL', 'ALI', 'TRA', 'COM', 'SAN', 'EDU', 'AUT'],  # Codes secteurs
                'weights': [0.30, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05],
                'replaces': 'merchant_category'
            },
            'motif': {
                'type': 'categorical',
                'values': ['ACH', 'PAI', 'TRF', 'RET', 'SAL', 'AUT'],
                'weights': [0.30, 0.25, 0.20, 0.10, 0.10, 0.05],
                'replaces': 'tx_purpose'
            }
        },
        'remove_columns': ['cust_region', 'hour'],  # Supprimer certaines colonnes
        'split_timestamp': True,  # Séparer en date + heure
        'columns_extra': {
            'ville_origine': {
                'type': 'categorical',
                'values': ['Ouagadougou', 'Bobo-Dioulasso', 'Koudougou', 'Ouahigouya', 'Autres'],
                'weights': [0.40, 0.25, 0.10, 0.10, 0.15],
                'description': 'Ville origine transaction (disponible au moment T)'
            },
            'client_segment': {
                'type': 'categorical',
                'values': ['VIP', 'Premium', 'Standard', 'Nouveau'],
                'weights': [0.05, 0.15, 0.60, 0.20],
                'description': 'Segment client statique (disponible au moment T)'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.20,
            'dormant_account': 0.20,
            'unusual_behavior': 0.20,
            'beneficiary_concentration': 0.20,
            'international_burst': 0.10,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 60000,
            'amount_std': 1.9,
            'international_rate': 0.20,
            'night_tx_rate': 0.05
        },
        'fraud_multipliers': {
            'amount_high': 4.0,  # Medium
            'night': 3.0,
            'ratio_account': 3.5,
            'new_account': 2.0
        }
    },
    
    'Dataset12': {
        'n_transactions': 20000,
        'fraud_rate': 0.03,
        'difficulty': 'hard',
        'description': 'BF_Medium_Sparse - Peu de fraud, beaucoup de bruit',
        'column_mapping': {
            # Retour à noms courts mais différents
            'cust_type': 'type_client',
            'cust_region': 'region_bf',
            'tx_method': 'methode',
            'fraud_flag': 'label_fraude'
        },
        'columns_extra': {},  # Pas de colonnes extra pour ce dataset (hard, minimal)
        'fraud_pattern_weights': {
            'smurfing': 0.15,
            'dormant_account': 0.25,  # Plus dormant (hard to detect)
            'unusual_behavior': 0.20,
            'beneficiary_concentration': 0.20,
            'international_burst': 0.15,
            'high_velocity': 0.05
        },
        'distributions': {
            'amount_mean': 40000,  # Montants plus bas
            'amount_std': 1.6,
            'international_rate': 0.15,
            'night_tx_rate': 0.03
        },
        'fraud_multipliers': {
            'amount_high': 2.0,  # Faible (hard)
            'night': 1.8,
            'ratio_account': 2.0,
            'new_account': 1.5
        }
    },
    
    'Dataset13': {
        'n_transactions': 18000,
        'fraud_rate': 0.06,
        'difficulty': 'medium',
        'description': 'BF_Medium_Standard - Standard, cas typique',
        'column_structure': 'structure_E',
        'column_mapping': {
            'tx_id': 'ref_transaction',
            'cust_id': 'numero_client',
            'fraud_flag': 'fraude'
        },
        'columns_extra': {
            'merchant_id': {
                'type': 'categorical',
                'generator': lambda: f'M{np.random.randint(1, 800):04d}',
                'description': 'ID marchand (cardinalité différente)'
            },
            'hour_category': {
                'type': 'categorical',
                'values': ['Matin', 'Apres-midi', 'Soir', 'Nuit'],
                'weights': [0.30, 0.40, 0.20, 0.10],
                'description': 'Catégorie horaire'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.18,
            'dormant_account': 0.18,
            'unusual_behavior': 0.22,
            'beneficiary_concentration': 0.18,
            'international_burst': 0.14,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 55000,
            'amount_std': 1.8,
            'international_rate': 0.22,
            'night_tx_rate': 0.05
        },
        'fraud_multipliers': {
            'amount_high': 3.5,
            'night': 3.0,
            'ratio_account': 3.0,
            'new_account': 2.0
        }
    },
    
    'Dataset14': {
        'n_transactions': 30000,
        'fraud_rate': 0.05,
        'difficulty': 'medium',
        'description': 'BF_Large_Standard - Grand dataset standard',
        'column_structure': 'structure_F',
        'remove_columns': ['weekday', 'processing_time_ms'],  # Retirer certaines colonnes
        'columns_extra': {},  # Minimal pour ce dataset
        'fraud_pattern_weights': {
            'smurfing': 0.20,
            'dormant_account': 0.15,
            'unusual_behavior': 0.25,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.15,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 65000,
            'amount_std': 2.0,
            'international_rate': 0.25,
            'night_tx_rate': 0.06
        },
        'fraud_multipliers': {
            'amount_high': 3.5,
            'night': 3.0,
            'ratio_account': 3.5,
            'new_account': 2.0
        }
    },
    
    'Dataset15': {
        'n_transactions': 40000,
        'fraud_rate': 0.12,
        'difficulty': 'easy',
        'description': 'BF_Large_Balanced - Beaucoup de données, patterns nets',
        'column_structure': 'structure_G',
        'split_timestamp': True,  # Date + heure séparées
        'column_mapping': {
            'cust_id': 'client',
            'tx_amount_xof': 'montant',
            'fraud_flag': 'flag_fraude'
        },
        'columns_extra': {
            'is_weekend': {
                'type': 'categorical',
                'values': ['Oui', 'Non'],
                'weights': [0.28, 0.72],
                'description': 'Transaction weekend'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.25,
            'dormant_account': 0.10,
            'unusual_behavior': 0.30,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.15,
            'high_velocity': 0.05
        },
        'distributions': {
            'amount_mean': 70000,
            'amount_std': 2.1,
            'international_rate': 0.30,
            'night_tx_rate': 0.10
        },
        'fraud_multipliers': {
            'amount_high': 5.0,
            'night': 4.0,
            'ratio_account': 4.5,
            'new_account': 2.5
        }
    },
    
    'Dataset16': {
        'n_transactions': 50000,
        'fraud_rate': 0.02,
        'difficulty': 'very_hard',
        'description': 'BF_Large_Rare - Énorme dataset, fraud rare',
        'column_structure': 'structure_H',
        'remove_columns': ['balance_before', 'balance_after', 'hour'],  # Features minimales
        'columns_extra': {},  # Aucune colonne extra (hard mode)
        'fraud_pattern_weights': {
            'smurfing': 0.10,
            'dormant_account': 0.30,  # Beaucoup dormant (hard)
            'unusual_behavior': 0.20,
            'beneficiary_concentration': 0.25,
            'international_burst': 0.10,
            'high_velocity': 0.05
        },
        'distributions': {
            'amount_mean': 45000,
            'amount_std': 1.7,
            'international_rate': 0.18,
            'night_tx_rate': 0.04
        },
        'fraud_multipliers': {
            'amount_high': 1.8,  # Très faible (very hard)
            'night': 1.5,
            'ratio_account': 1.8,
            'new_account': 1.3
        }
    },
    
    'Dataset17': {
        'n_transactions': 25000,
        'fraud_rate': 0.08,
        'difficulty': 'medium',
        'description': 'BF_Large_Varied - Structure très différente',
        'column_structure': 'structure_I',
        'column_mapping': {
            'tx_timestamp': 'datetime_tx',
            'cust_type': 'segment',
            'tx_method': 'canal',
            'fraud_flag': 'is_fraud'
        },
        'column_overrides': {
            'segment': {
                'type': 'categorical',
                'values': ['Particulier', 'Pro', 'Entreprise'],
                'weights': [0.70, 0.20, 0.10],
                'replaces': 'cust_type'
            },
            'canal': {
                'type': 'categorical',
                'values': ['APP', 'WEB', 'USSD', 'AGENT', 'GAB'],
                'weights': [0.40, 0.25, 0.15, 0.15, 0.05],
                'replaces': 'tx_method'
            }
        },
        'columns_extra': {
            'device_os': {
                'type': 'categorical',
                'values': ['Android', 'iOS', 'Web', 'USSD', 'Autre'],
                'weights': [0.50, 0.20, 0.15, 0.10, 0.05],
                'description': 'Système exploitation'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.15,
            'dormant_account': 0.20,
            'unusual_behavior': 0.25,
            'beneficiary_concentration': 0.25,
            'international_burst': 0.10,
            'high_velocity': 0.05
        },
        'distributions': {
            'amount_mean': 75000,
            'amount_std': 2.0,
            'international_rate': 0.28,
            'night_tx_rate': 0.07
        },
        'fraud_multipliers': {
            'amount_high': 2.5,
            'night': 2.0,
            'ratio_account': 2.5,
            'new_account': 1.8
        }
    },
    
    'Dataset18': {
        'n_transactions': 35000,
        'fraud_rate': 0.09,
        'difficulty': 'easy',
        'description': 'BF_Large_Easy - Large et facile',
        'column_structure': 'structure_J',
        'column_mapping': {
            'tx_id': 'id',
            'cust_id': 'client_id',
            'cust_region': 'province',
            'fraud_flag': 'fraud'
        },
        'columns_extra': {
            'transaction_channel': {
                'type': 'categorical',
                'values': ['Mobile', 'Internet', 'Agence', 'Partenaire'],
                'weights': [0.50, 0.25, 0.15, 0.10],
                'description': 'Canal de transaction'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.20,
            'dormant_account': 0.15,
            'unusual_behavior': 0.25,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.15,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 58000,
            'amount_std': 1.9,
            'international_rate': 0.23,
            'night_tx_rate': 0.05
        },
        'fraud_multipliers': {
            'amount_high': 3.5,
            'night': 3.0,
            'ratio_account': 3.5,
            'new_account': 2.0
        }
    },
    
    'Dataset19': {
        'n_transactions': 22000,
        'fraud_rate': 0.07,
        'difficulty': 'medium',
        'description': 'BF_Medium_Complex - Moyennement complexe',
        'column_structure': 'structure_K',
        'remove_columns': ['processing_time_ms', 'weekday'],
        'split_timestamp': True,
        'column_mapping': {
            'cust_id': 'customer_ref',
            'tx_amount_xof': 'amount_fcfa',
            'merchant_category': 'business_type',
            'fraud_flag': 'is_fraudulent'
        },
        'columns_extra': {
            'network_provider': {
                'type': 'categorical',
                'values': ['Orange', 'Moov', 'Telecel', 'Coris', 'Autre'],
                'weights': [0.35, 0.30, 0.20, 0.10, 0.05],
                'description': 'Opérateur réseau'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.18,
            'dormant_account': 0.17,
            'unusual_behavior': 0.23,
            'beneficiary_concentration': 0.17,
            'international_burst': 0.15,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 62000,
            'amount_std': 1.95,
            'international_rate': 0.24,
            'night_tx_rate': 0.06
        },
        'fraud_multipliers': {
            'amount_high': 3.5,
            'night': 3.0,
            'ratio_account': 3.5,
            'new_account': 2.0
        }
    },
    
    # ==================================================================================
    # DATASETS 20-30: BANKING ORIENTATION (STRUCTURES TRÈS VARIÉES)
    # ==================================================================================
    
    'Dataset20': {
        'n_transactions': 28000,
        'fraud_rate': 0.04,
        'difficulty': 'medium',
        'description': 'Bank_Card_Fraud - Fraude par carte bancaire',
        'currency': 'FCFA',
        'column_structure': 'structure_banking_A',
        'column_mapping': {
            'tx_id': 'card_transaction_id',
            'cust_id': 'card_number_hash',
            'cust_type': 'card_type',
            'tx_amount_xof': 'transaction_amount_fcfa',
            'tx_method': 'authorization_method',
            'tx_purpose': 'merchant_category_code',
            'fraud_flag': 'is_fraudulent_transaction'
        },
        'column_overrides': {
            'card_type': {
                'type': 'categorical',
                'values': ['Debit', 'Credit', 'Prepaid', 'Corporate'],
                'weights': [0.55, 0.30, 0.10, 0.05],
                'replaces': 'cust_type'
            },
            'authorization_method': {
                'type': 'categorical',
                'values': ['PIN', 'Signature', 'Contactless', 'Online', 'Chip'],
                'weights': [0.35, 0.10, 0.25, 0.20, 0.10],
                'replaces': 'tx_method'
            },
            'merchant_category_code': {
                'type': 'categorical',
                'values': ['Retail', 'Gas_Station', 'Restaurant', 'Online_Shopping', 'Travel', 'Entertainment', 'Healthcare', 'Utilities'],
                'weights': [0.25, 0.15, 0.18, 0.20, 0.08, 0.06, 0.05, 0.03],
                'replaces': 'tx_purpose'
            }
        },
        'remove_columns': ['cust_region', 'balance_before', 'balance_after', 'processing_time_ms'],
        'columns_extra': {
            'card_present': {
                'type': 'binary',
                'true_rate': 0.65,
                'description': 'Card physically present'
            },
            'cvv_verified': {
                'type': 'binary',
                'true_rate': 0.85,
                'description': 'CVV code verified'
            },
            'distance_from_home': {
                'type': 'numeric',
                'generator': lambda: np.random.lognormal(3, 1.5),
                'description': 'Distance en km du lieu habituel'
            },
            'transaction_velocity_24h': {
                'type': 'numeric',
                'generator': lambda: max(0, int(np.random.gamma(2, 2))),
                'description': 'Nombre de transactions dernières 24h'
            },
            'transaction_currency': {
                'type': 'categorical',
                'values': ['FCFA', 'EUR', 'USD'],
                'weights': [0.70, 0.20, 0.10],
                'description': 'Devise transaction (majoritairement FCFA, parfois EUR/USD pour international)'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.10,
            'dormant_account': 0.05,
            'unusual_behavior': 0.40,
            'beneficiary_concentration': 0.10,
            'international_burst': 0.25,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 556750,  # 850 EUR × 655 = 556,750 FCFA
            'amount_std': 2.5,
            'international_rate': 0.15,
            'night_tx_rate': 0.08
        },
        'fraud_multipliers': {
            'amount_high': 4.0,
            'night': 3.5,
            'ratio_account': 5.0,
            'new_account': 2.5
        }
    },
    
    'Dataset21': {
        'n_transactions': 45000,
        'fraud_rate': 0.015,
        'difficulty': 'very_hard',
        'description': 'Bank_Wire_Transfer - Virements bancaires (fraude rare)',
        'currency': 'FCFA',
        'column_structure': 'structure_banking_B',
        'column_mapping': {
            'tx_id': 'wire_reference',
            'cust_id': 'account_iban_hash',
            'cust_type': 'account_category',
            'tx_amount_xof': 'wire_amount_fcfa',
            'tx_method': 'wire_type',
            'tx_purpose': 'transfer_reason',
            'fraud_flag': 'flagged_suspicious'
        },
        'column_overrides': {
            'account_category': {
                'type': 'categorical',
                'values': ['Personal', 'Business', 'Savings', 'Investment', 'Joint'],
                'weights': [0.50, 0.25, 0.15, 0.07, 0.03],
                'replaces': 'cust_type'
            },
            'wire_type': {
                'type': 'categorical',
                'values': ['SEPA', 'SWIFT', 'Internal', 'Express', 'Scheduled'],
                'weights': [0.40, 0.15, 0.30, 0.10, 0.05],
                'replaces': 'tx_method'
            },
            'transfer_reason': {
                'type': 'categorical',
                'values': ['Invoice_Payment', 'Salary', 'Personal_Transfer', 'Investment', 'Loan_Repayment', 'Tax_Payment', 'Other'],
                'weights': [0.30, 0.20, 0.25, 0.08, 0.07, 0.05, 0.05],
                'replaces': 'tx_purpose'
            }
        },
        'remove_columns': ['hour', 'weekday', 'merchant_category'],
        'split_timestamp': True,
        'columns_extra': {
            'originator_bank_country': {
                'type': 'categorical',
                'values': ['FR', 'DE', 'UK', 'US', 'CH', 'LU', 'BE', 'NL', 'Other'],
                'weights': [0.35, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.05],
                'description': 'Pays banque émettrice'
            },
            'beneficiary_bank_country': {
                'type': 'categorical',
                'values': ['FR', 'DE', 'UK', 'US', 'CH', 'CN', 'AE', 'SG', 'Other'],
                'weights': [0.30, 0.12, 0.10, 0.12, 0.05, 0.08, 0.06, 0.05, 0.12],
                'description': 'Pays banque bénéficiaire'
            },
            'relationship_length_months': {
                'type': 'numeric',
                'generator': lambda: int(np.random.gamma(10, 5)),
                'description': 'Ancienneté relation bénéficiaire (mois)'
            },
            'swift_message_type': {
                'type': 'categorical',
                'values': ['MT103', 'MT202', 'MT400', 'MT760', 'Other'],
                'weights': [0.70, 0.15, 0.08, 0.03, 0.04],
                'description': 'Type message SWIFT'
            },
            'wire_currency': {
                'type': 'categorical',
                'values': ['FCFA', 'USD', 'EUR', 'GBP'],
                'weights': [0.60, 0.20, 0.15, 0.05],
                'description': 'Devise virement (majoritairement FCFA, USD/EUR pour international)'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.20,
            'dormant_account': 0.25,
            'unusual_behavior': 0.20,
            'beneficiary_concentration': 0.20,
            'international_burst': 0.10,
            'high_velocity': 0.05
        },
        'distributions': {
            'amount_mean': 15000000,  # 25,000 USD × 600 = 15,000,000 FCFA
            'amount_std': 2.8,
            'international_rate': 0.40,
            'night_tx_rate': 0.03
        },
        'fraud_multipliers': {
            'amount_high': 2.0,
            'night': 1.5,
            'ratio_account': 2.5,
            'new_account': 2.0
        }
    },
    
    'Dataset22': {
        'n_transactions': 18000,
        'fraud_rate': 0.11,
        'difficulty': 'easy',
        'description': 'Bank_ATM_Skimming - Fraude distributeur automatique',
        'currency': 'FCFA',
        'column_structure': 'structure_banking_C',
        'column_mapping': {
            'tx_id': 'atm_transaction_ref',
            'cust_id': 'card_hash',
            'cust_type': 'card_issuer_type',
            'cust_region': 'atm_region',
            'tx_amount_xof': 'withdrawal_amount_fcfa',
            'tx_method': 'atm_brand',
            'fraud_flag': 'skimming_detected'
        },
        'column_overrides': {
            'card_issuer_type': {
                'type': 'categorical',
                'values': ['Major_Bank', 'Regional_Bank', 'Credit_Union', 'Online_Bank', 'Neobank'],
                'weights': [0.50, 0.25, 0.12, 0.08, 0.05],
                'replaces': 'cust_type'
            },
            'atm_region': {
                'type': 'categorical',
                'values': ['City_Center', 'Suburb', 'Highway', 'Mall', 'Airport', 'Rural'],
                'weights': [0.30, 0.35, 0.10, 0.15, 0.05, 0.05],
                'replaces': 'cust_region'
            },
            'atm_brand': {
                'type': 'categorical',
                'values': ['BankOwned', 'Diebold', 'NCR', 'Wincor', 'Generic'],
                'weights': [0.45, 0.20, 0.18, 0.12, 0.05],
                'replaces': 'tx_method'
            }
        },
        'remove_columns': ['tx_purpose', 'merchant_category', 'dest_country', 'balance_before', 'balance_after'],
        'columns_extra': {
            'atm_indoor': {
                'type': 'binary',
                'true_rate': 0.60,
                'description': 'ATM en intérieur vs extérieur'
            },
            'camera_operational': {
                'type': 'binary',
                'true_rate': 0.85,
                'description': 'Caméra de surveillance fonctionnelle'
            },
            'last_maintenance_days': {
                'type': 'numeric',
                'generator': lambda: int(np.random.uniform(1, 90)),
                'description': 'Jours depuis dernière maintenance'
            },
            'consecutive_withdrawals': {
                'type': 'numeric',
                'generator': lambda: max(1, int(np.random.geometric(0.7))),
                'description': 'Retraits consécutifs même carte'
            },
            'card_currency': {
                'type': 'categorical',
                'values': ['FCFA', 'EUR', 'USD'],
                'weights': [0.85, 0.10, 0.05],
                'description': 'Devise carte (majoritairement FCFA local)'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.05,
            'dormant_account': 0.10,
            'unusual_behavior': 0.50,
            'beneficiary_concentration': 0.05,
            'international_burst': 0.10,
            'high_velocity': 0.20
        },
        'distributions': {
            'amount_mean': 140400,  # 180 GBP × 780 = 140,400 FCFA
            'amount_std': 1.6,
            'international_rate': 0.05,
            'night_tx_rate': 0.20
        },
        'fraud_multipliers': {
            'amount_high': 6.0,
            'night': 5.0,
            'ratio_account': 4.0,
            'new_account': 3.0
        }
    },
    
    'Dataset23': {
        'n_transactions': 60000,
        'fraud_rate': 0.02,
        'difficulty': 'hard',
        'description': 'Bank_Corporate_Large - Transactions corporates (gros volumes)',
        'currency': 'FCFA',
        'column_structure': 'structure_banking_D',
        'column_mapping': {
            'tx_id': 'payment_order_id',
            'cust_id': 'corporate_client_id',
            'cust_type': 'company_size',
            'cust_region': 'headquarters_country',
            'tx_amount_xof': 'amount_fcfa',
            'tx_method': 'payment_system',
            'tx_purpose': 'business_category',
            'fraud_flag': 'aml_flagged'
        },
        'column_overrides': {
            'company_size': {
                'type': 'categorical',
                'values': ['SME', 'MidCap', 'Large_Corp', 'Multinational'],
                'weights': [0.40, 0.35, 0.18, 0.07],
                'replaces': 'cust_type'
            },
            'headquarters_country': {
                'type': 'categorical',
                'values': ['France', 'Germany', 'UK', 'Netherlands', 'Belgium', 'Italy', 'Spain', 'Switzerland', 'Other_EU', 'Non_EU'],
                'weights': [0.25, 0.18, 0.15, 0.10, 0.08, 0.07, 0.06, 0.04, 0.05, 0.02],
                'replaces': 'cust_region'
            },
            'payment_system': {
                'type': 'categorical',
                'values': ['SEPA_SCT', 'SEPA_SDD', 'TARGET2', 'BankInternal', 'Check', 'BillOfExchange'],
                'weights': [0.45, 0.25, 0.15, 0.08, 0.05, 0.02],
                'replaces': 'tx_method'
            },
            'business_category': {
                'type': 'categorical',
                'values': ['Manufacturing', 'Services', 'Trade', 'Construction', 'Technology', 'Finance', 'Healthcare', 'Energy', 'Other'],
                'weights': [0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.04],
                'replaces': 'tx_purpose'
            }
        },
        'remove_columns': ['merchant_category', 'hour', 'weekday'],
        'columns_extra': {
            'employee_count': {
                'type': 'numeric',
                'generator': lambda: int(np.random.lognormal(4, 2)),
                'description': 'Nombre employés entreprise'
            },
            'annual_revenue_m_eur': {
                'type': 'numeric',
                'generator': lambda: max(0.1, np.random.lognormal(2, 1.8)),
                'description': 'Chiffre affaires annuel (millions EUR)'
            },
            'years_as_client': {
                'type': 'numeric',
                'generator': lambda: int(np.random.gamma(5, 2)),
                'description': 'Ancienneté client banque (années)'
            },
            'authorized_signatories': {
                'type': 'numeric',
                'generator': lambda: np.random.choice([1, 2, 3, 4, 5], p=[0.40, 0.30, 0.15, 0.10, 0.05]),
                'description': 'Nombre signataires autorisés'
            },
            'payment_currency': {
                'type': 'categorical',
                'values': ['FCFA', 'EUR', 'USD'],
                'weights': [0.50, 0.30, 0.20],
                'description': 'Devise paiement (FCFA local, EUR/USD pour import/export)'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.15,
            'dormant_account': 0.20,
            'unusual_behavior': 0.25,
            'beneficiary_concentration': 0.25,
            'international_burst': 0.10,
            'high_velocity': 0.05
        },
        'distributions': {
            'amount_mean': 55675000,  # 85,000 EUR × 655 = 55,675,000 FCFA
            'amount_std': 3.2,
            'international_rate': 0.30,
            'night_tx_rate': 0.02
        },
        'fraud_multipliers': {
            'amount_high': 2.5,
            'night': 2.0,
            'ratio_account': 3.0,
            'new_account': 2.5
        }
    },
    
    'Dataset24': {
        'n_transactions': 32000,
        'fraud_rate': 0.08,
        'difficulty': 'medium',
        'description': 'Bank_Mobile_Banking - App bancaire mobile',
        'currency': 'FCFA',
        'column_structure': 'structure_banking_E',
        'column_mapping': {
            'tx_id': 'session_transaction_id',
            'cust_id': 'user_id_hash',
            'cust_type': 'user_segment',
            'cust_region': 'login_country',
            'tx_amount_xof': 'amount_fcfa',
            'tx_method': 'operation_type',
            'tx_purpose': 'transaction_category',
            'fraud_flag': 'fraud_alert'
        },
        'column_overrides': {
            'user_segment': {
                'type': 'categorical',
                'values': ['Millennial', 'GenZ', 'GenX', 'Boomer', 'Premium'],
                'weights': [0.35, 0.25, 0.20, 0.15, 0.05],
                'replaces': 'cust_type'
            },
            'login_country': {
                'type': 'categorical',
                'values': ['Home_Country', 'EU', 'US', 'Asia', 'Other'],
                'weights': [0.75, 0.12, 0.05, 0.05, 0.03],
                'replaces': 'cust_region'
            },
            'operation_type': {
                'type': 'categorical',
                'values': ['P2P_Transfer', 'Bill_Payment', 'TopUp', 'QR_Payment', 'CardPayment'],
                'weights': [0.35, 0.25, 0.20, 0.12, 0.08],
                'replaces': 'tx_method'
            },
            'transaction_category': {
                'type': 'categorical',
                'values': ['Shopping', 'Transport', 'Food', 'Utilities', 'Entertainment', 'Health', 'Education', 'Other'],
                'weights': [0.28, 0.15, 0.18, 0.12, 0.10, 0.07, 0.05, 0.05],
                'replaces': 'tx_purpose'
            }
        },
        'remove_columns': ['balance_before', 'balance_after', 'processing_time_ms'],
        'split_timestamp': True,
        'columns_extra': {
            'device_os': {
                'type': 'categorical',
                'values': ['iOS', 'Android', 'HarmonyOS', 'Other'],
                'weights': [0.40, 0.55, 0.03, 0.02],
                'description': 'Système exploitation mobile'
            },
            'biometric_auth': {
                'type': 'binary',
                'true_rate': 0.70,
                'description': 'Authentification biométrique utilisée'
            },
            'app_version_outdated': {
                'type': 'binary',
                'true_rate': 0.15,
                'description': 'Version app obsolète'
            },
            'session_duration_seconds': {
                'type': 'numeric',
                'generator': lambda: int(np.random.gamma(3, 60)),
                'description': 'Durée session (secondes)'
            },
            'app_transaction_currency': {
                'type': 'categorical',
                'values': ['FCFA', 'USD', 'EUR'],
                'weights': [0.90, 0.06, 0.04],
                'description': 'Devise transaction app (quasi-exclusivement FCFA)'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.20,
            'dormant_account': 0.10,
            'unusual_behavior': 0.35,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.10,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 72000,  # 120 USD × 600 = 72,000 FCFA
            'amount_std': 2.0,
            'international_rate': 0.08,
            'night_tx_rate': 0.12
        },
        'fraud_multipliers': {
            'amount_high': 4.5,
            'night': 3.0,
            'ratio_account': 4.0,
            'new_account': 3.0
        }
    },
    
    'Dataset25': {
        'n_transactions': 15000,
        'fraud_rate': 0.13,
        'difficulty': 'easy',
        'description': 'Bank_Crypto_Gateway - Passerelle crypto-monnaie',
        'currency': 'FCFA',
        'column_structure': 'structure_banking_F',
        'column_mapping': {
            'tx_id': 'crypto_tx_hash',
            'cust_id': 'wallet_address_hash',
            'cust_type': 'wallet_type',
            'tx_amount_xof': 'amount_crypto_fcfa_equiv',
            'tx_method': 'crypto_currency',
            'tx_purpose': 'exchange_direction',
            'fraud_flag': 'suspicious_activity'
        },
        'column_overrides': {
            'wallet_type': {
                'type': 'categorical',
                'values': ['Custodial', 'Non_Custodial', 'Hardware', 'Exchange'],
                'weights': [0.50, 0.25, 0.15, 0.10],
                'replaces': 'cust_type'
            },
            'crypto_currency': {
                'type': 'categorical',
                'values': ['Bitcoin', 'Ethereum', 'USDT', 'USDC', 'Litecoin', 'Ripple', 'Other'],
                'weights': [0.35, 0.25, 0.15, 0.10, 0.05, 0.05, 0.05],
                'replaces': 'tx_method'
            },
            'exchange_direction': {
                'type': 'categorical',
                'values': ['Crypto_to_Fiat', 'Fiat_to_Crypto', 'Crypto_to_Crypto'],
                'weights': [0.45, 0.40, 0.15],
                'replaces': 'tx_purpose'
            }
        },
        'remove_columns': ['cust_region', 'merchant_category', 'dest_country', 'balance_before', 'balance_after'],
        'columns_extra': {
            'kyc_level': {
                'type': 'categorical',
                'values': ['Basic', 'Intermediate', 'Advanced', 'Institutional'],
                'weights': [0.35, 0.40, 0.20, 0.05],
                'description': 'Niveau vérification KYC'
            },
            'blockchain_confirmations': {
                'type': 'numeric',
                'generator': lambda: int(np.random.choice([1, 3, 6, 12], p=[0.10, 0.50, 0.30, 0.10])),
                'description': 'Nombre confirmations blockchain'
            },
            'volatility_24h_pct': {
                'type': 'numeric',
                'generator': lambda: abs(np.random.normal(2, 3)),
                'description': 'Volatilité crypto 24h (%)'
            },
            'previous_tx_count': {
                'type': 'numeric',
                'generator': lambda: int(np.random.gamma(3, 5)),
                'description': 'Nombre transactions historiques'
            },
            'crypto_exchange_currency': {
                'type': 'categorical',
                'values': ['FCFA', 'EUR', 'USD'],
                'weights': [0.65, 0.25, 0.10],
                'description': 'Devise échange crypto (FCFA/EUR/USD)'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.30,
            'dormant_account': 0.05,
            'unusual_behavior': 0.30,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.15,
            'high_velocity': 0.05
        },
        'distributions': {
            'amount_mean': 982500,  # 1,500 EUR × 655 = 982,500 FCFA
            'amount_std': 2.8,
            'international_rate': 0.60,
            'night_tx_rate': 0.25
        },
        'fraud_multipliers': {
            'amount_high': 5.0,
            'night': 2.5,
            'ratio_account': 4.5,
            'new_account': 4.0
        }
    },
    
    'Dataset26': {
        'n_transactions': 52000,
        'fraud_rate': 0.025,
        'difficulty': 'hard',
        'description': 'Bank_Mortgage_Payments - Paiements hypothécaires',
        'currency': 'FCFA',
        'column_structure': 'structure_banking_G',
        'column_mapping': {
            'tx_id': 'payment_reference',
            'cust_id': 'loan_account_id',
            'cust_type': 'property_type',
            'cust_region': 'property_region',
            'tx_amount_xof': 'monthly_payment_fcfa',
            'tx_method': 'payment_channel',
            'tx_purpose': 'payment_component',
            'fraud_flag': 'payment_irregularity'
        },
        'column_overrides': {
            'property_type': {
                'type': 'categorical',
                'values': ['Primary_Residence', 'Second_Home', 'Investment', 'Condo', 'Commercial'],
                'weights': [0.65, 0.12, 0.15, 0.05, 0.03],
                'replaces': 'cust_type'
            },
            'property_region': {
                'type': 'categorical',
                'values': ['Urban', 'Suburban', 'Rural', 'Coastal', 'Mountain'],
                'weights': [0.35, 0.40, 0.15, 0.07, 0.03],
                'replaces': 'cust_region'
            },
            'payment_channel': {
                'type': 'categorical',
                'values': ['Auto_Debit', 'Online_Banking', 'Check', 'Wire_Transfer', 'Branch'],
                'weights': [0.70, 0.18, 0.05, 0.04, 0.03],
                'replaces': 'tx_method'
            },
            'payment_component': {
                'type': 'categorical',
                'values': ['Principal_Interest', 'Principal_Only', 'Interest_Only', 'Escrow', 'Late_Fee'],
                'weights': [0.85, 0.05, 0.03, 0.05, 0.02],
                'replaces': 'tx_purpose'
            }
        },
        'remove_columns': ['merchant_category', 'dest_country', 'hour', 'weekday'],
        'columns_extra': {
            'loan_to_value_ratio': {
                'type': 'numeric',
                'generator': lambda: min(95, max(50, np.random.normal(75, 12))),
                'description': 'Ratio prêt/valeur (%)'
            },
            'months_since_origination': {
                'type': 'numeric',
                'generator': lambda: int(np.random.uniform(1, 360)),
                'description': 'Mois depuis origine prêt'
            },
            'payment_history_30d': {
                'type': 'categorical',
                'values': ['OnTime', 'Late_1_15_Days', 'Late_16_30_Days', 'Late_Over_30_Days'],
                'weights': [0.90, 0.06, 0.03, 0.01],
                'description': 'Historique paiements 30j'
            },
            'interest_rate_pct': {
                'type': 'numeric',
                'generator': lambda: max(2.5, min(8.0, np.random.normal(4.5, 1.2))),
                'description': 'Taux intérêt (%)'
            },
            'loan_currency': {
                'type': 'categorical',
                'values': ['FCFA', 'USD', 'EUR'],
                'weights': [0.80, 0.12, 0.08],
                'description': 'Devise prêt (majoritairement FCFA local)'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.05,
            'dormant_account': 0.30,
            'unusual_behavior': 0.25,
            'beneficiary_concentration': 0.20,
            'international_burst': 0.05,
            'high_velocity': 0.15
        },
        'distributions': {
            'amount_mean': 1080000,  # 1,800 USD × 600 = 1,080,000 FCFA
            'amount_std': 1.8,
            'international_rate': 0.02,
            'night_tx_rate': 0.01
        },
        'fraud_multipliers': {
            'amount_high': 2.5,
            'night': 2.0,
            'ratio_account': 3.5,
            'new_account': 2.0
        }
    },
    
    'Dataset27': {
        'n_transactions': 38000,
        'fraud_rate': 0.06,
        'difficulty': 'medium',
        'description': 'Bank_Investment_Trading - Trading actions/obligations',
        'currency': 'FCFA',
        'column_structure': 'structure_banking_H',
        'column_mapping': {
            'tx_id': 'trade_order_id',
            'cust_id': 'brokerage_account_id',
            'cust_type': 'investor_profile',
            'cust_region': 'market_region',
            'tx_amount_xof': 'trade_value_fcfa',
            'tx_method': 'order_type',
            'tx_purpose': 'asset_class',
            'fraud_flag': 'market_manipulation_flag'
        },
        'column_overrides': {
            'investor_profile': {
                'type': 'categorical',
                'values': ['Conservative', 'Moderate', 'Aggressive', 'Professional', 'Institutional'],
                'weights': [0.25, 0.35, 0.25, 0.10, 0.05],
                'replaces': 'cust_type'
            },
            'market_region': {
                'type': 'categorical',
                'values': ['North_America', 'Europe', 'Asia_Pacific', 'Emerging_Markets', 'Global'],
                'weights': [0.40, 0.25, 0.20, 0.10, 0.05],
                'replaces': 'cust_region'
            },
            'order_type': {
                'type': 'categorical',
                'values': ['Market_Order', 'Limit_Order', 'Stop_Loss', 'Stop_Limit', 'Trailing_Stop'],
                'weights': [0.40, 0.35, 0.12, 0.08, 0.05],
                'replaces': 'tx_method'
            },
            'asset_class': {
                'type': 'categorical',
                'values': ['Equity', 'Fixed_Income', 'ETF', 'Mutual_Fund', 'Options', 'Futures', 'Forex'],
                'weights': [0.35, 0.20, 0.20, 0.12, 0.06, 0.04, 0.03],
                'replaces': 'tx_purpose'
            }
        },
        'remove_columns': ['merchant_category', 'dest_country', 'balance_before', 'balance_after'],
        'columns_extra': {
            'portfolio_value_k_usd': {
                'type': 'numeric',
                'generator': lambda: max(1, np.random.lognormal(4, 2)),
                'description': 'Valeur portefeuille (milliers USD)'
            },
            'trade_execution_speed_ms': {
                'type': 'numeric',
                'generator': lambda: int(np.random.gamma(2, 50)),
                'description': 'Vitesse exécution (ms)'
            },
            'leverage_ratio': {
                'type': 'numeric',
                'generator': lambda: np.random.choice([1.0, 2.0, 5.0, 10.0], p=[0.70, 0.15, 0.10, 0.05]),
                'description': 'Ratio effet levier'
            },
            'volatility_index': {
                'type': 'numeric',
                'generator': lambda: max(10, min(80, np.random.normal(20, 15))),
                'description': 'Indice volatilité marché'
            },
            'trade_currency': {
                'type': 'categorical',
                'values': ['FCFA', 'USD', 'EUR'],
                'weights': [0.40, 0.40, 0.20],
                'description': 'Devise trading (FCFA local, USD/EUR pour marchés internationaux)'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.10,
            'dormant_account': 0.15,
            'unusual_behavior': 0.35,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.15,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 9000000,  # 15,000 USD × 600 = 9,000,000 FCFA
            'amount_std': 3.0,
            'international_rate': 0.35,
            'night_tx_rate': 0.05
        },
        'fraud_multipliers': {
            'amount_high': 3.5,
            'night': 2.5,
            'ratio_account': 3.5,
            'new_account': 2.5
        }
    },
    
    'Dataset28': {
        'n_transactions': 25000,
        'fraud_rate': 0.09,
        'difficulty': 'medium',
        'description': 'Bank_Insurance_Claims - Réclamations assurance',
        'currency': 'FCFA',
        'column_structure': 'structure_banking_I',
        'column_mapping': {
            'tx_id': 'claim_number',
            'cust_id': 'policy_holder_id',
            'cust_type': 'policy_type',
            'cust_region': 'claim_region',
            'tx_amount_xof': 'claim_amount_fcfa',
            'tx_method': 'claim_channel',
            'tx_purpose': 'claim_category',
            'fraud_flag': 'fraud_indicator'
        },
        'column_overrides': {
            'policy_type': {
                'type': 'categorical',
                'values': ['Auto', 'Home', 'Life', 'Health', 'Travel', 'Business'],
                'weights': [0.35, 0.25, 0.15, 0.15, 0.06, 0.04],
                'replaces': 'cust_type'
            },
            'claim_region': {
                'type': 'categorical',
                'values': ['Urban', 'Suburban', 'Rural', 'High_Risk_Zone', 'Disaster_Area'],
                'weights': [0.40, 0.35, 0.18, 0.05, 0.02],
                'replaces': 'cust_region'
            },
            'claim_channel': {
                'type': 'categorical',
                'values': ['Online_Portal', 'Phone', 'Agent', 'Mobile_App', 'Email'],
                'weights': [0.40, 0.30, 0.15, 0.10, 0.05],
                'replaces': 'tx_method'
            },
            'claim_category': {
                'type': 'categorical',
                'values': ['Property_Damage', 'Theft', 'Accident', 'Medical', 'Natural_Disaster', 'Liability', 'Other'],
                'weights': [0.25, 0.18, 0.20, 0.15, 0.08, 0.10, 0.04],
                'replaces': 'tx_purpose'
            }
        },
        'remove_columns': ['merchant_category', 'dest_country', 'hour', 'weekday'],
        'columns_extra': {
            'policy_age_months': {
                'type': 'numeric',
                'generator': lambda: int(np.random.gamma(8, 6)),
                'description': 'Âge police (mois)'
            },
            'previous_claims_count': {
                'type': 'numeric',
                'generator': lambda: int(np.random.choice([0, 1, 2, 3, 4, 5], p=[0.60, 0.20, 0.10, 0.05, 0.03, 0.02])),
                'description': 'Nombre réclamations antérieures'
            },
            'claim_complexity_score': {
                'type': 'numeric',
                'generator': lambda: int(np.random.choice([1, 2, 3, 4, 5], p=[0.30, 0.30, 0.20, 0.15, 0.05])),
                'description': 'Score complexité réclamation (1-5)'
            },
            'adjuster_experience_years': {
                'type': 'numeric',
                'generator': lambda: int(np.random.gamma(5, 2)),
                'description': 'Expérience expert sinistre (années)'
            },
            'claim_currency': {
                'type': 'categorical',
                'values': ['FCFA', 'EUR'],
                'weights': [0.75, 0.25],
                'description': 'Devise réclamation (FCFA local, EUR pour assurances internationales)'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.10,
            'dormant_account': 0.15,
            'unusual_behavior': 0.40,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.05,
            'high_velocity': 0.15
        },
        'distributions': {
            'amount_mean': 2947500,  # 4,500 EUR × 655 = 2,947,500 FCFA
            'amount_std': 2.5,
            'international_rate': 0.08,
            'night_tx_rate': 0.03
        },
        'fraud_multipliers': {
            'amount_high': 4.5,
            'night': 2.0,
            'ratio_account': 4.0,
            'new_account': 3.5
        }
    },
    
    'Dataset29': {
        'n_transactions': 70000,
        'fraud_rate': 0.018,
        'difficulty': 'very_hard',
        'description': 'Bank_Retail_POS - Terminaux point de vente retail (fraude rare)',
        'currency': 'FCFA',
        'column_structure': 'structure_banking_J',
        'column_mapping': {
            'tx_id': 'pos_transaction_id',
            'cust_id': 'merchant_id',
            'cust_type': 'merchant_size',
            'cust_region': 'merchant_location',
            'tx_amount_xof': 'transaction_amount_fcfa',
            'tx_method': 'payment_method',
            'tx_purpose': 'product_category',
            'fraud_flag': 'chargeback_fraud'
        },
        'column_overrides': {
            'merchant_size': {
                'type': 'categorical',
                'values': ['Micro', 'Small', 'Medium', 'Large', 'Enterprise'],
                'weights': [0.30, 0.35, 0.20, 0.10, 0.05],
                'replaces': 'cust_type'
            },
            'merchant_location': {
                'type': 'categorical',
                'values': ['City_Center', 'Shopping_Mall', 'Residential', 'Industrial', 'Airport', 'Station', 'Online_Only'],
                'weights': [0.25, 0.20, 0.25, 0.08, 0.05, 0.07, 0.10],
                'replaces': 'cust_region'
            },
            'payment_method': {
                'type': 'categorical',
                'values': ['Chip_PIN', 'Contactless', 'Swipe', 'Manual_Entry', 'Mobile_Wallet', 'QR_Code'],
                'weights': [0.40, 0.30, 0.10, 0.05, 0.12, 0.03],
                'replaces': 'tx_method'
            },
            'product_category': {
                'type': 'categorical',
                'values': ['Groceries', 'Electronics', 'Clothing', 'Fuel', 'Pharmacy', 'Restaurant', 'Services', 'Other'],
                'weights': [0.25, 0.12, 0.15, 0.10, 0.08, 0.15, 0.10, 0.05],
                'replaces': 'tx_purpose'
            }
        },
        'remove_columns': ['balance_before', 'balance_after', 'processing_time_ms'],
        'columns_extra': {
            'terminal_id': {
                'type': 'categorical',
                'generator': lambda: f'TERM{np.random.randint(1, 5000):05d}',
                'description': 'ID terminal POS'
            },
            'tip_included': {
                'type': 'binary',
                'true_rate': 0.25,
                'description': 'Pourboire inclus'
            },
            'cashback_amount': {
                'type': 'numeric',
                'generator': lambda: np.random.choice([0, 20, 40, 60, 100], p=[0.85, 0.06, 0.05, 0.03, 0.01]),
                'description': 'Montant cashback'
            },
            'merchant_risk_score': {
                'type': 'numeric',
                'generator': lambda: int(np.random.beta(2, 5) * 100),
                'description': 'Score risque commerçant (0-100)'
            },
            'pos_currency': {
                'type': 'categorical',
                'values': ['FCFA', 'EUR', 'USD'],
                'weights': [0.92, 0.05, 0.03],
                'description': 'Devise POS (quasi-exclusivement FCFA local)'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.08,
            'dormant_account': 0.10,
            'unusual_behavior': 0.30,
            'beneficiary_concentration': 0.25,
            'international_burst': 0.12,
            'high_velocity': 0.15
        },
        'distributions': {
            'amount_mean': 39000,  # 65 USD × 600 = 39,000 FCFA
            'amount_std': 2.2,
            'international_rate': 0.03,
            'night_tx_rate': 0.06
        },
        'fraud_multipliers': {
            'amount_high': 2.0,
            'night': 1.8,
            'ratio_account': 2.5,
            'new_account': 2.0
        }
    },
    
    'Dataset30': {
        'n_transactions': 42000,
        'fraud_rate': 0.055,
        'difficulty': 'medium',
        'description': 'Bank_P2P_Lending - Prêts peer-to-peer',
        'currency': 'FCFA',
        'column_structure': 'structure_banking_K',
        'column_mapping': {
            'tx_id': 'loan_transaction_id',
            'cust_id': 'borrower_id',
            'cust_type': 'credit_grade',
            'cust_region': 'borrower_state',
            'tx_amount_xof': 'loan_amount_fcfa',
            'tx_method': 'loan_purpose',
            'tx_purpose': 'repayment_status',
            'fraud_flag': 'default_fraud_flag'
        },
        'column_overrides': {
            'credit_grade': {
                'type': 'categorical',
                'values': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                'weights': [0.10, 0.18, 0.25, 0.22, 0.15, 0.07, 0.03],
                'replaces': 'cust_type'
            },
            'borrower_state': {
                'type': 'categorical',
                'values': ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI', 'Other'],
                'weights': [0.15, 0.10, 0.09, 0.08, 0.06, 0.06, 0.05, 0.05, 0.05, 0.04, 0.27],
                'replaces': 'cust_region'
            },
            'loan_purpose': {
                'type': 'categorical',
                'values': ['Debt_Consolidation', 'Credit_Card_Payoff', 'Home_Improvement', 'Business', 'Major_Purchase', 'Medical', 'Other'],
                'weights': [0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.03],
                'replaces': 'tx_method'
            },
            'repayment_status': {
                'type': 'categorical',
                'values': ['Current', 'Late_16_30_Days', 'Late_31_120_Days', 'Default', 'Paid_Off'],
                'weights': [0.70, 0.08, 0.05, 0.03, 0.14],
                'replaces': 'tx_purpose'
            }
        },
        'remove_columns': ['merchant_category', 'dest_country', 'hour', 'weekday', 'balance_before', 'balance_after'],
        'columns_extra': {
            'debt_to_income_ratio': {
                'type': 'numeric',
                'generator': lambda: min(50, max(5, np.random.normal(20, 10))),
                'description': 'Ratio dette/revenu (%)'
            },
            'annual_income_k_usd': {
                'type': 'numeric',
                'generator': lambda: max(15, np.random.lognormal(4, 0.8)),
                'description': 'Revenu annuel (milliers USD)'
            },
            'employment_length_years': {
                'type': 'numeric',
                'generator': lambda: int(np.random.gamma(5, 2)),
                'description': 'Ancienneté emploi (années)'
            },
            'number_of_open_accounts': {
                'type': 'numeric',
                'generator': lambda: int(np.random.gamma(3, 3)),
                'description': 'Nombre comptes ouverts'
            },
            'number_of_delinquencies': {
                'type': 'numeric',
                'generator': lambda: int(np.random.choice([0, 1, 2, 3, 4], p=[0.70, 0.15, 0.08, 0.05, 0.02])),
                'description': 'Nombre impayés historiques'
            },
            'lending_currency': {
                'type': 'categorical',
                'values': ['FCFA', 'USD'],
                'weights': [0.85, 0.15],
                'description': 'Devise prêt (majoritairement FCFA, USD pour P2P international)'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.10,
            'dormant_account': 0.20,
            'unusual_behavior': 0.30,
            'beneficiary_concentration': 0.20,
            'international_burst': 0.05,
            'high_velocity': 0.15
        },
        'distributions': {
            'amount_mean': 7200000,  # 12,000 USD × 600 = 7,200,000 FCFA
            'amount_std': 2.4,
            'international_rate': 0.02,
            'night_tx_rate': 0.01
        },
        'fraud_multipliers': {
            'amount_high': 3.5,
            'night': 1.5,
            'ratio_account': 4.0,
            'new_account': 3.5
        }
    },
    
    # ==================== DATASETS 31-35: TRÈS GROS (100K-500K) ====================
    
    'Dataset31': {
        'n_transactions': 100000,
        'fraud_rate': 0.045,
        'difficulty': 'hard',
        'description': 'BIG_100K - Grand volume, patterns dilués',
        'currency': 'FCFA',
        'column_structure': 'structure_A',
        'column_mapping': {
            'tx_id': 'tx_id',
            'cust_id': 'cust_id',
            'cust_type': 'cust_type',
            'cust_region': 'cust_region',
            'tx_amount_xof': 'tx_amount_xof',
            'tx_method': 'tx_method',
            'tx_purpose': 'tx_purpose',
            'merchant_category': 'merchant_category',
            'dest_country': 'dest_country',
            'fraud_flag': 'fraud_flag'
        },
        'columns_extra': {
            'transaction_category': {
                'type': 'categorical',
                'values': ['Retail', 'Online', 'Service', 'Transfer', 'Bill_Payment'],
                'weights': [0.30, 0.25, 0.20, 0.15, 0.10],
                'description': 'Catégorie transaction'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.25,
            'dormant_account': 0.15,
            'unusual_behavior': 0.20,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.15,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 45000,
            'amount_std': 1.9,
            'international_rate': 0.20,
            'night_tx_rate': 0.08
        },
        'fraud_multipliers': {
            'amount_high': 3.0,
            'night': 2.5,
            'ratio_account': 3.5,
            'new_account': 2.8
        }
    },
    
    'Dataset32': {
        'n_transactions': 200000,
        'fraud_rate': 0.035,
        'difficulty': 'very_hard',
        'description': 'BIG_200K - Très grand volume, grande variété',
        'currency': 'FCFA',
        'column_structure': 'structure_B',
        'column_mapping': {
            'tx_id': 'transaction_ref',
            'cust_id': 'client_number',
            'cust_type': 'customer_segment',
            'cust_region': 'region',
            'tx_amount_xof': 'amount_fcfa',
            'tx_method': 'payment_channel',
            'tx_purpose': 'transaction_type',
            'merchant_category': 'merchant_cat',
            'dest_country': 'destination',
            'fraud_flag': 'is_fraudulent'
        },
        'columns_extra': {
            'device_id': {
                'type': 'categorical',
                'generator': lambda: f'DEV{np.random.randint(1, 20000):06d}',
                'description': 'Identifiant appareil'
            },
            'merchant_risk_level': {
                'type': 'categorical',
                'values': ['Low', 'Medium', 'High', 'Unknown'],
                'weights': [0.60, 0.25, 0.10, 0.05],
                'description': 'Niveau risque marchand'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.30,
            'dormant_account': 0.20,
            'unusual_behavior': 0.15,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.10,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 55000,
            'amount_std': 2.0,
            'international_rate': 0.25,
            'night_tx_rate': 0.10
        },
        'fraud_multipliers': {
            'amount_high': 3.2,
            'night': 2.8,
            'ratio_account': 3.8,
            'new_account': 3.0
        }
    },
    
    'Dataset33': {
        'n_transactions': 300000,
        'fraud_rate': 0.05,
        'difficulty': 'very_hard',
        'description': 'BIG_300K - Énorme volume, complexité maximale',
        'currency': 'FCFA',
        'column_structure': 'structure_A',
        'column_mapping': {
            'tx_id': 'tx_id',
            'cust_id': 'cust_id',
            'cust_type': 'cust_type',
            'cust_region': 'cust_region',
            'tx_amount_xof': 'tx_amount_xof',
            'tx_method': 'tx_method',
            'tx_purpose': 'tx_purpose',
            'merchant_category': 'merchant_category',
            'dest_country': 'dest_country',
            'fraud_flag': 'fraud_flag'
        },
        'columns_extra': {
            'merchant_type': {
                'type': 'categorical',
                'values': ['E-commerce', 'Retail', 'Restaurant', 'Service', 'Transport'],
                'weights': [0.30, 0.25, 0.20, 0.15, 0.10],
                'description': 'Type marchand'
            },
            'device_fingerprint': {
                'type': 'categorical',
                'generator': lambda: f'FP{np.random.randint(1, 10000):05d}',
                'description': 'Empreinte appareil'
            },
            'ip_reputation_score': {
                'type': 'numeric',
                'generator': lambda: round(np.random.beta(8, 2) * 100, 1),
                'description': 'Score réputation IP (0-100)'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.20,
            'dormant_account': 0.20,
            'unusual_behavior': 0.20,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.15,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 48000,
            'amount_std': 2.1,
            'international_rate': 0.22,
            'night_tx_rate': 0.09
        },
        'fraud_multipliers': {
            'amount_high': 3.5,
            'night': 3.0,
            'ratio_account': 4.0,
            'new_account': 3.2
        }
    },
    
    'Dataset34': {
        'n_transactions': 400000,
        'fraud_rate': 0.03,
        'difficulty': 'extreme',
        'description': 'BIG_400K - Massif, patterns sophistiqués',
        'currency': 'FCFA',
        'column_structure': 'structure_B',
        'column_mapping': {
            'tx_id': 'txn_id',
            'cust_id': 'account_id',
            'cust_type': 'account_type',
            'cust_region': 'geographic_zone',
            'tx_amount_xof': 'transaction_amount_fcfa',
            'tx_method': 'channel',
            'tx_purpose': 'purpose_code',
            'merchant_category': 'business_sector',
            'dest_country': 'dest_country',
            'fraud_flag': 'fraud_indicator'
        },
        'columns_extra': {
            'velocity_1h': {
                'type': 'numeric',
                'generator': lambda: int(np.random.gamma(2, 1.5)),
                'description': 'Nombre transactions dernière heure'
            },
            'velocity_24h': {
                'type': 'numeric',
                'generator': lambda: int(np.random.gamma(3, 3)),
                'description': 'Nombre transactions dernières 24h'
            },
            'avg_amount_30d': {
                'type': 'numeric',
                'generator': lambda: int(np.random.lognormal(10.5, 0.8)),
                'description': 'Montant moyen 30 derniers jours'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.28,
            'dormant_account': 0.18,
            'unusual_behavior': 0.18,
            'beneficiary_concentration': 0.12,
            'international_burst': 0.12,
            'high_velocity': 0.12
        },
        'distributions': {
            'amount_mean': 52000,
            'amount_std': 2.2,
            'international_rate': 0.28,
            'night_tx_rate': 0.12
        },
        'fraud_multipliers': {
            'amount_high': 3.5,
            'night': 3.0,
            'ratio_account': 4.0,
            'new_account': 3.5,
            'velocity': 4.5
        }
    },
    
    'Dataset35': {
        'n_transactions': 500000,
        'fraud_rate': 0.04,
        'difficulty': 'extreme',
        'description': 'BIG_500K - Géant, aiguille dans botte de foin',
        'currency': 'FCFA',
        'column_structure': 'structure_A',
        'column_mapping': {
            'tx_id': 'tx_id',
            'cust_id': 'cust_id',
            'cust_type': 'cust_type',
            'cust_region': 'cust_region',
            'tx_amount_xof': 'tx_amount_xof',
            'tx_method': 'tx_method',
            'tx_purpose': 'tx_purpose',
            'merchant_category': 'merchant_category',
            'dest_country': 'dest_country',
            'fraud_flag': 'fraud_flag'
        },
        'columns_extra': {
            'session_id': {
                'type': 'categorical',
                'generator': lambda: f'SESS{np.random.randint(1, 50000):06d}'
            },
            'browser_type': {
                'type': 'categorical',
                'values': ['Chrome', 'Firefox', 'Safari', 'Edge', 'Mobile'],
                'weights': [0.40, 0.20, 0.15, 0.15, 0.10]
            },
            'connection_quality': {
                'type': 'categorical',
                'values': ['4G', 'WiFi', '3G', 'Ethernet', '2G'],
                'weights': [0.35, 0.30, 0.20, 0.10, 0.05],
                'description': 'Type de connexion utilisé'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.22,
            'dormant_account': 0.18,
            'unusual_behavior': 0.20,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.15,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 47000,
            'amount_std': 1.95,
            'international_rate': 0.18,
            'night_tx_rate': 0.07
        },
        'fraud_multipliers': {
            'amount_high': 3.8,
            'night': 3.5,
            'ratio_account': 4.5,
            'new_account': 4.0
        }
    },
    
    # ==================== DATASETS 36-40: TAUX DE FRAUDE TRÈS BAS (0.1%-0.5%) ====================
    
    'Dataset36': {
        'n_transactions': 50000,
        'fraud_rate': 0.001,  # 0.1% - ~50 cas de fraude
        'difficulty': 'extreme',
        'description': 'LOW_FRAUD_0.1% - Fraude ultra-rare (50 cas)',
        'currency': 'FCFA',
        'column_structure': 'structure_A',
        'column_mapping': {
            'tx_id': 'tx_id',
            'cust_id': 'cust_id',
            'cust_type': 'cust_type',
            'cust_region': 'cust_region',
            'tx_amount_xof': 'tx_amount_xof',
            'tx_method': 'tx_method',
            'tx_purpose': 'tx_purpose',
            'merchant_category': 'merchant_category',
            'dest_country': 'dest_country',
            'fraud_flag': 'fraud_flag'
        },
        'columns_extra': {
            'kyc_verification_level': {
                'type': 'categorical',
                'values': ['basic', 'standard', 'enhanced', 'premium'],
                'weights': [0.15, 0.40, 0.30, 0.15],
                'description': 'Niveau vérification KYC'
            },
            'account_balance_range': {
                'type': 'categorical',
                'values': ['<10K', '10K-50K', '50K-200K', '>200K'],
                'weights': [0.25, 0.40, 0.25, 0.10],
                'description': 'Tranche solde compte'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.30,
            'dormant_account': 0.20,
            'unusual_behavior': 0.15,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.10,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 40000,
            'amount_std': 1.7,
            'international_rate': 0.15,
            'night_tx_rate': 0.05
        },
        'fraud_multipliers': {
            'amount_high': 5.0,  # Fraude très évidente quand elle apparaît
            'night': 4.0,
            'ratio_account': 5.0,
            'new_account': 5.0
        }
    },
    
    'Dataset37': {
        'n_transactions': 75000,
        'fraud_rate': 0.002,  # 0.2% - ~150 cas de fraude
        'difficulty': 'extreme',
        'description': 'LOW_FRAUD_0.2% - Fraude très rare (150 cas)',
        'currency': 'FCFA',
        'column_structure': 'structure_B',
        'column_mapping': {
            'tx_id': 'operation_id',
            'cust_id': 'customer_num',
            'cust_type': 'customer_segment',
            'cust_region': 'region',
            'tx_amount_xof': 'amount',
            'tx_method': 'method',
            'tx_purpose': 'purpose',
            'merchant_category': 'merchant_cat',
            'dest_country': 'destination',
            'fraud_flag': 'is_anomaly'
        },
        'columns_extra': {
            'geolocation_verified': {
                'type': 'categorical',
                'values': ['verified', 'not_verified', 'suspicious'],
                'weights': [0.80, 0.15, 0.05],
                'description': 'Statut vérification géolocalisation'
            },
            'time_since_last_tx_min': {
                'type': 'numeric',
                'generator': lambda: int(np.random.exponential(120)),
                'description': 'Minutes depuis dernière transaction'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.25,
            'dormant_account': 0.20,
            'unusual_behavior': 0.20,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.10,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 42000,
            'amount_std': 1.8,
            'international_rate': 0.16,
            'night_tx_rate': 0.06
        },
        'fraud_multipliers': {
            'amount_high': 4.5,
            'night': 3.5,
            'ratio_account': 4.5,
            'new_account': 4.5
        }
    },
    
    'Dataset38': {
        'n_transactions': 60000,
        'fraud_rate': 0.003,  # 0.3% - ~180 cas de fraude
        'difficulty': 'very_hard',
        'description': 'LOW_FRAUD_0.3% - Fraude rare (180 cas)',
        'currency': 'FCFA',
        'column_structure': 'structure_A',
        'column_mapping': {
            'tx_id': 'tx_id',
            'cust_id': 'cust_id',
            'cust_type': 'cust_type',
            'cust_region': 'cust_region',
            'tx_amount_xof': 'tx_amount_xof',
            'tx_method': 'tx_method',
            'tx_purpose': 'tx_purpose',
            'merchant_category': 'merchant_category',
            'dest_country': 'dest_country',
            'fraud_flag': 'fraud_flag'
        },
        'columns_extra': {
            'risk_score': {
                'type': 'numeric',
                'generator': lambda: round(np.random.uniform(0, 100), 2),
                'description': 'Score de risque calculé'
            },
            'previous_fraud_reports': {
                'type': 'numeric',
                'generator': lambda: np.random.choice([0, 0, 0, 1, 2], p=[0.92, 0.05, 0.02, 0.007, 0.003]),
                'description': 'Nombre signalements fraude antérieurs'
            },
            'merchant_reputation': {
                'type': 'categorical',
                'values': ['excellent', 'good', 'average', 'poor', 'unknown'],
                'weights': [0.30, 0.35, 0.20, 0.10, 0.05],
                'description': 'Réputation du marchand'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.25,
            'dormant_account': 0.18,
            'unusual_behavior': 0.20,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.12,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 43000,
            'amount_std': 1.85,
            'international_rate': 0.17,
            'night_tx_rate': 0.06
        },
        'fraud_multipliers': {
            'amount_high': 4.0,
            'night': 3.0,
            'ratio_account': 4.0,
            'new_account': 4.0
        }
    },
    
    'Dataset39': {
        'n_transactions': 80000,
        'fraud_rate': 0.004,  # 0.4% - ~320 cas de fraude
        'difficulty': 'very_hard',
        'description': 'LOW_FRAUD_0.4% - Fraude peu fréquente (320 cas)',
        'currency': 'FCFA',
        'column_structure': 'structure_B',
        'column_mapping': {
            'tx_id': 'trans_ref',
            'cust_id': 'user_id',
            'cust_type': 'user_type',
            'cust_region': 'location',
            'tx_amount_xof': 'transaction_value',
            'tx_method': 'channel',
            'tx_purpose': 'reason',
            'merchant_category': 'business_type',
            'dest_country': 'target_country',
            'fraud_flag': 'suspicious'
        },
        'columns_extra': {
            'recurring_transaction': {
                'type': 'categorical',
                'values': ['yes', 'no'],
                'weights': [0.25, 0.75],
                'description': 'Transaction récurrente'
            },
            'beneficiary_trust_score': {
                'type': 'numeric',
                'generator': lambda: round(np.random.beta(5, 2) * 100, 2),
                'description': 'Score confiance bénéficiaire'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.23,
            'dormant_account': 0.18,
            'unusual_behavior': 0.20,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.14,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 44000,
            'amount_std': 1.9,
            'international_rate': 0.18,
            'night_tx_rate': 0.07
        },
        'fraud_multipliers': {
            'amount_high': 3.8,
            'night': 3.0,
            'ratio_account': 3.8,
            'new_account': 3.8
        }
    },
    
    'Dataset40': {
        'n_transactions': 100000,
        'fraud_rate': 0.005,  # 0.5% - ~500 cas de fraude
        'difficulty': 'hard',
        'description': 'LOW_FRAUD_0.5% - Fraude occasionnelle (500 cas)',
        'currency': 'FCFA',
        'column_structure': 'structure_A',
        'column_mapping': {
            'tx_id': 'tx_id',
            'cust_id': 'cust_id',
            'cust_type': 'cust_type',
            'cust_region': 'cust_region',
            'tx_amount_xof': 'tx_amount_xof',
            'tx_method': 'tx_method',
            'tx_purpose': 'tx_purpose',
            'merchant_category': 'merchant_category',
            'dest_country': 'dest_country',
            'fraud_flag': 'fraud_flag'
        },
        'columns_extra': {
            'transaction_score': {
                'type': 'numeric',
                'generator': lambda: round(np.random.uniform(0, 1), 4),
                'description': 'Score de confiance de la transaction'
            },
            'merchant_reputation': {
                'type': 'categorical',
                'values': ['Excellent', 'Good', 'Average', 'Poor', 'New'],
                'weights': [0.30, 0.35, 0.20, 0.10, 0.05]
            },
            'payment_gateway': {
                'type': 'categorical',
                'values': ['gateway_A', 'gateway_B', 'gateway_C', 'direct', 'aggregator'],
                'weights': [0.25, 0.20, 0.15, 0.30, 0.10],
                'description': 'Passerelle de paiement utilisée'
            }
        },
        'fraud_pattern_weights': {
            'smurfing': 0.22,
            'dormant_account': 0.18,
            'unusual_behavior': 0.20,
            'beneficiary_concentration': 0.15,
            'international_burst': 0.15,
            'high_velocity': 0.10
        },
        'distributions': {
            'amount_mean': 46000,
            'amount_std': 1.92,
            'international_rate': 0.19,
            'night_tx_rate': 0.08
        },
        'fraud_multipliers': {
            'amount_high': 3.5,
            'night': 2.8,
            'ratio_account': 3.5,
            'new_account': 3.5
        }
    }
}


def get_config(dataset_name: str) -> dict:
    """Récupère la configuration pour un dataset"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not found. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]


def validate_config(config_name: str) -> list:
    """
    Valide qu'une configuration est complète et cohérente
    
    Retourne une liste d'erreurs (vide si tout est OK)
    """
    if config_name not in DATASET_CONFIGS:
        return [f"Dataset {config_name} not found"]
    
    config = DATASET_CONFIGS[config_name]
    errors = []
    warnings = []
    
    # 1. Vérifier champs obligatoires
    required_fields = [
        'n_transactions', 'fraud_rate', 'difficulty', 'description',
        'currency', 'column_structure', 'column_mapping', 
        'distributions', 'fraud_pattern_weights'
    ]
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # 2. Vérifier cohérence currency
    if 'currency' in config and 'column_mapping' in config:
        currency = config['currency']
        amount_col = config['column_mapping'].get('tx_amount_xof', '')
        
        if currency == 'FCFA' and 'fcfa' not in amount_col.lower():
            errors.append(f"currency='FCFA' but amount column '{amount_col}' doesn't contain 'fcfa'")
        
        # Vérifier qu'il y a une colonne *_currency dans columns_extra
        has_currency_col = any('currency' in k.lower() for k in config.get('columns_extra', {}).keys())
        if not has_currency_col:
            warnings.append(f"No *_currency column found in columns_extra (recommended for multi-currency tracking)")
    
    # 3. Vérifier amount_mean cohérent avec FCFA
    if 'distributions' in config and 'currency' in config:
        amount_mean = config['distributions'].get('amount_mean', 0)
        
        if config['currency'] == 'FCFA':
            if amount_mean < 1000:
                errors.append(f"amount_mean={amount_mean:,.0f} FCFA is too small (should be >= 1,000)")
            elif amount_mean > 100_000_000:
                warnings.append(f"amount_mean={amount_mean:,.0f} FCFA is very large (> 100M)")
    
    # 4. Vérifier weights des colonnes_extra somment à 1.0
    for col_name, col_config in config.get('columns_extra', {}).items():
        if 'weights' in col_config:
            weights = col_config['weights']
            total = sum(weights)
            
            if abs(total - 1.0) > 0.01:
                errors.append(f"Column '{col_name}': weights sum to {total:.3f} instead of 1.0")
            
            # Vérifier que len(values) == len(weights)
            if 'values' in col_config:
                if len(col_config['values']) != len(weights):
                    errors.append(f"Column '{col_name}': {len(col_config['values'])} values but {len(weights)} weights")
    
    # 5. Vérifier fraud_pattern_weights somment à 1.0
    if 'fraud_pattern_weights' in config:
        total = sum(config['fraud_pattern_weights'].values())
        if abs(total - 1.0) > 0.01:
            errors.append(f"fraud_pattern_weights sum to {total:.3f} instead of 1.0")
    
    # 6. Vérifier fraud_rate est valide
    if 'fraud_rate' in config:
        fraud_rate = config['fraud_rate']
        if not (0.0 < fraud_rate < 1.0):
            errors.append(f"fraud_rate={fraud_rate} should be between 0 and 1")
    
    # 7. Vérifier n_transactions est raisonnable
    if 'n_transactions' in config:
        n = config['n_transactions']
        if n < 1000:
            warnings.append(f"n_transactions={n:,} is very small (< 1,000)")
        elif n > 100_000:
            warnings.append(f"n_transactions={n:,} is very large (> 100,000) - generation may be slow")
    
    # 8. Vérifier difficulty est valide
    if 'difficulty' in config:
        valid_difficulties = ['very_easy', 'easy', 'medium', 'hard', 'very_hard']
        if config['difficulty'] not in valid_difficulties:
            errors.append(f"difficulty='{config['difficulty']}' not in {valid_difficulties}")
    
    # Combiner errors et warnings
    all_messages = []
    if errors:
        all_messages.extend([f"ERROR: {e}" for e in errors])
    if warnings:
        all_messages.extend([f"WARNING: {w}" for w in warnings])
    
    return all_messages


def validate_all_configs(verbose=True) -> dict:
    """
    Valide toutes les configurations
    
    Retourne un dict avec les résultats de validation
    """
    results = {}
    
    for config_name in sorted(DATASET_CONFIGS.keys()):
        messages = validate_config(config_name)
        results[config_name] = {
            'valid': len([m for m in messages if m.startswith('ERROR')]) == 0,
            'messages': messages
        }
    
    if verbose:
        print("\n" + "="*80)
        print("VALIDATION DES CONFIGURATIONS")
        print("="*80)
        
        valid_count = sum(1 for r in results.values() if r['valid'])
        total_count = len(results)
        
        for config_name in sorted(results.keys()):
            result = results[config_name]
            status = "OK " if result['valid'] else "ERR"
            
            config = DATASET_CONFIGS[config_name]
            desc = config['description'][:50]
            
            if result['valid'] and not result['messages']:
                print(f"{status} {config_name}: {desc:50s} OK")
            else:
                print(f"{status} {config_name}: {desc:50s}")
                for msg in result['messages']:
                    print(f"     {msg}")
        
        print("\n" + "="*80)
        print(f"Resume: {valid_count}/{total_count} configurations valides")
        
        if valid_count == total_count:
            print("Toutes les configurations sont valides!")
        else:
            print(f"ATTENTION: {total_count - valid_count} configuration(s) avec erreurs")
        print("="*80 + "\n")
    
    return results


def print_all_configs():
    """Affiche un résumé de toutes les configurations"""
    print("\n" + "="*80)
    print("CONFIGURATIONS DES DATASETS 9-40")
    print("="*80)
    
    for name, config in DATASET_CONFIGS.items():
        print(f"\n{name}: {config['description']}")
        print(f"  • Transactions: {config['n_transactions']:,}")
        print(f"  • Fraud rate: {config['fraud_rate']*100:.1f}%")
        print(f"  • Difficulty: {config['difficulty']}")
        print(f"  • Currency: {config.get('currency', 'N/A')}")
        print(f"  • Extra columns: {len(config.get('columns_extra', {}))}")
        print(f"  • Total columns: {18 + len(config.get('columns_extra', {}))}")
    
    print("\n" + "="*80)
    print(f"Total datasets: {len(DATASET_CONFIGS)}")
    print(f"Total transactions: {sum(c['n_transactions'] for c in DATASET_CONFIGS.values()):,}")
    print("="*80 + "\n")


if __name__ == '__main__':
    # Valider toutes les configs
    results = validate_all_configs(verbose=True)
    
    # Afficher résumé si tout est valide
    if all(r['valid'] for r in results.values()):
        print_all_configs()

