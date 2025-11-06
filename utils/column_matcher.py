"""
column_matcher.py v2.0

Matching sémantique avancé de colonnes entre datasets.
Résout le problème des noms de colonnes différents pour des concepts identiques.

Nouvelles fonctionnalités v2.0:
- Cache de similarité pour accélérer les comparaisons répétées
- Support multi-langues étendu (FR, EN, ES, DE)
- Détection automatique de patterns numériques (_1, _2, ratio, squared)
- Synonymes enrichis par domaine (banking, e-commerce, telco)
- Scoring amélioré avec pondération intelligente
- Support des abréviations courantes (amt, tx, cust, etc.)

Exemples:
    transaction_amount = amount = montant = transaction_value = tx_amount = amt
    customer_id = client_id = user_id = cust_id = custid
    timestamp = tx_time = transaction_date = date_transaction = datetime
    amount_ratio = ratio_amount = amount_normalized
"""

from typing import List, Dict, Set, Tuple, Optional
from difflib import SequenceMatcher
import re
from functools import lru_cache


class ColumnMatcher:
    """
    Match des colonnes basé sur la sémantique plutôt que le nom exact.
    
    Version 2.0: Support multi-langues étendu, cache de similarité, 
    détection de patterns numériques, synonymes enrichis.
    """
    
    # Groupes sémantiques de noms de colonnes (v2.0 - ENRICHIS)
    SEMANTIC_GROUPS = {
        'amount': [
            # Anglais
            'amount', 'value', 'price', 'sum', 'total', 'balance', 'payment',
            'transaction_amount', 'tx_amount', 'trans_amount', 'payment_amount',
            'amt', 'val', 'txamt', 'transamt', 'payamt',
            # Français
            'montant', 'valeur', 'prix', 'solde', 'paiement', 'somme', 'total',
            # Espagnol
            'monto', 'valor', 'precio', 'saldo', 'pago',
            # Allemand
            'betrag', 'wert', 'preis', 'saldo', 'zahlung',
            # Patterns
            'volume', 'cash', 'money', 'fund', 'debit', 'credit'
        ],
        'timestamp': [
            # Anglais
            'timestamp', 'time', 'date', 'datetime', 'tx_time', 'trans_time',
            'transaction_time', 'transaction_date', 'tx_date', 'trans_date',
            'created_at', 'updated_at', 'processed_at', 'posted_at',
            # Français
            'heure', 'temps', 'horodatage', 'horodate', 'date_transaction',
            # Autres
            'when', 'at', 'on', 'dt', 'ts', 'epoch'
        ],
        'customer_id': [
            # Anglais
            'customer_id', 'client_id', 'user_id', 'cust_id', 'userid',
            'customer', 'client', 'user', 'account_id', 'sender_id',
            'custid', 'clientid', 'uid', 'accountid',
            # Français
            'id_client', 'id_utilisateur', 'identifiant_client',
            # Patterns
            'payer', 'buyer', 'owner', 'holder', 'cardholder'
        ],
        'merchant': [
            # Anglais
            'merchant', 'merchant_id', 'merchant_name', 'merchantid',
            'shop', 'store', 'vendor', 'receiver', 'beneficiary', 'payee',
            'seller', 'supplier', 'provider',
            # Français
            'commercant', 'vendeur', 'fournisseur', 'destinataire',
            # Patterns
            'to', 'recipient', 'acceptor'
        ],
        'transaction_id': [
            # Anglais
            'transaction_id', 'tx_id', 'trans_id', 'transid', 'txid',
            'transaction_number', 'tx_number', 'trans_number',
            'reference', 'ref', 'trace', 'trace_id', 'traceid',
            # Français
            'id_transaction', 'numero_transaction', 'reference',
            # Patterns
            'uuid', 'guid', 'key', 'identifier'
        ],
        'country': [
            # Anglais
            'country', 'nation', 'country_code', 'country_name',
            'destination_country', 'dest_country', 'origin_country',
            'src_country', 'source_country',
            # Français
            'pays', 'nation', 'pays_destination', 'pays_origine',
            # Patterns
            'iso', 'iso_code', 'nationality'
        ],
        'currency': [
            # Anglais
            'currency', 'currency_code', 'curr', 'money_type', 'ccy',
            # Français
            'devise', 'monnaie', 'code_devise',
            # Patterns
            'iso_currency', 'curr_code'
        ],
        'card': [
            # Anglais
            'card', 'card_number', 'card_type', 'card_id', 'cardid',
            'pan', 'card_hash', 'bin', 'card_bin', 'issuer',
            # Français
            'carte', 'numero_carte', 'type_carte',
            # Patterns
            'payment_card', 'debit_card', 'credit_card'
        ],
        'status': [
            # Anglais
            'status', 'state', 'transaction_status', 'payment_status',
            'approval_status', 'auth_status', 'authorization_status',
            # Français
            'statut', 'etat', 'statut_transaction',
            # Patterns
            'approved', 'declined', 'pending', 'result', 'outcome'
        ],
        'channel': [
            # Anglais
            'channel', 'channel_type', 'payment_method', 'method',
            'transaction_type', 'tx_type', 'trans_type', 'entry_mode',
            # Français
            'canal', 'type_canal', 'methode_paiement', 'type_transaction',
            # Patterns
            'mode', 'interface', 'platform', 'device_type'
        ],
        'fraud': [
            # Anglais
            'fraud', 'is_fraud', 'fraud_flag', 'fraudulent', 'is_fraudulent',
            'suspicious', 'suspect', 'risk', 'risk_score', 'risk_level',
            # Français
            'fraude', 'frauduleux', 'suspect', 'risque',
            # Patterns
            'alert', 'anomaly', 'threat', 'score'
        ],
        'age': [
            # Anglais
            'age', 'customer_age', 'client_age', 'user_age', 'cust_age',
            # Français
            'age', 'age_client',
            # Patterns
            'years', 'years_old', 'birth'
        ],
        'location': [
            # Anglais
            'location', 'city', 'region', 'province', 'state', 'area',
            'address', 'zip', 'zipcode', 'postal', 'postal_code',
            # Français
            'ville', 'region', 'province', 'adresse', 'code_postal',
            # Patterns
            'geo', 'geography', 'place'
        ],
        'phone': [
            # Anglais
            'phone', 'mobile', 'phone_number', 'tel', 'telephone',
            'cell', 'cellphone', 'mobile_number',
            # Français
            'telephone', 'mobile', 'numero_telephone',
            # Patterns
            'contact', 'msisdn'
        ],
        'email': [
            # Anglais
            'email', 'mail', 'email_address', 'e_mail',
            # Français
            'courriel', 'adresse_email',
            # Patterns
            'contact_email'
        ],
        'account': [
            # Anglais
            'account', 'account_number', 'account_id', 'accountid',
            'iban', 'bban', 'routing', 'routing_number',
            # Français
            'compte', 'numero_compte', 'id_compte',
            # Patterns
            'acc', 'acct', 'acctnbr'
        ],
        'ip_address': [
            # Anglais
            'ip', 'ip_address', 'ipaddress', 'ip_addr',
            'source_ip', 'dest_ip', 'destination_ip',
            # Patterns
            'ipv4', 'ipv6', 'host'
        ],
        'device': [
            # Anglais
            'device', 'device_id', 'deviceid', 'device_type',
            'device_fingerprint', 'fingerprint',
            # Patterns
            'terminal', 'pos', 'atm', 'mobile_device'
        ],
        'browser': [
            # Anglais
            'browser', 'user_agent', 'useragent', 'ua',
            # Patterns
            'client', 'browser_type'
        ],
        'distance': [
            # Anglais
            'distance', 'dist', 'km', 'miles',
            'geo_distance', 'location_distance',
            # Patterns
            'proximity', 'separation'
        ],
        'ratio': [
            # Patterns numériques
            'ratio', 'rate', 'percentage', 'pct', 'percent',
            'proportion', 'fraction', 'normalized',
            # Français
            'taux', 'pourcentage'
        ],
        'count': [
            # Anglais
            'count', 'number', 'num', 'nbr', 'quantity', 'qty',
            'frequency', 'freq', 'occurrences',
            # Français
            'nombre', 'quantite', 'frequence',
            # Patterns
            'total', 'sum'
        ]
    }
    
    # Abréviations courantes (v2.0 - NOUVEAU)
    ABBREVIATIONS = {
        'tx': 'transaction',
        'trans': 'transaction',
        'amt': 'amount',
        'val': 'value',
        'cust': 'customer',
        'acct': 'account',
        'acc': 'account',
        'num': 'number',
        'nbr': 'number',
        'qty': 'quantity',
        'freq': 'frequency',
        'pct': 'percent',
        'avg': 'average',
        'std': 'standard',
        'dev': 'deviation',
        'min': 'minimum',
        'max': 'maximum',
        'dst': 'distance',
        'dist': 'distance',
        'src': 'source',
        'dest': 'destination',
        'orig': 'origin',
        'recv': 'receiver',
        'snd': 'sender'
    }
    
    def __init__(self, fuzzy_threshold: float = 0.7, use_cache: bool = True):
        """
        Args:
            fuzzy_threshold: Seuil de similarité pour fuzzy matching (0-1)
            use_cache: Active le cache de similarité (recommandé pour performances)
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.use_cache = use_cache
        self._similarity_cache = {} if use_cache else None
        
        # Créer un index inversé : nom -> groupe sémantique
        self.name_to_group = {}
        for group, names in self.SEMANTIC_GROUPS.items():
            for name in names:
                self.name_to_group[name.lower()] = group
    
    def is_temporal_column(self, col: str) -> bool:
        """
        Détecte si une colonne est temporelle (date, time, timestamp).
        Ces colonnes ne devraient JAMAIS être des targets de classification.
        
        Args:
            col: Nom de la colonne
            
        Returns:
            True si la colonne est temporelle
        """
        col_lower = col.lower()
        
        # Patterns temporels explicites
        temporal_exact = {'weekday', 'day', 'month', 'year', 'hour', 'minute', 'second',
                         'day_of_week', 'day_of_month', 'day_of_year', 'week', 'quarter',
                         'date', 'time', 'datetime', 'timestamp'}
        
        if col_lower in temporal_exact:
            return True
        
        # Patterns temporels partiels
        temporal_patterns = ['date', 'time', 'timestamp', 'datetime', 'heure', 'temps',
                            'horodat', 'epoch', 'created', 'updated', 'modified']
        
        if any(pattern in col_lower for pattern in temporal_patterns):
            return True
        
        # Vérifier via groupe sémantique
        group = self.get_semantic_group(col)
        return group == 'timestamp'
    
    def is_financial_continuous_column(self, col: str) -> bool:
        """
        Détecte si une colonne est financière continue (montants, balances, prix).
        Ces colonnes ne devraient JAMAIS être des targets binaires de classification.
        
        Args:
            col: Nom de la colonne
            
        Returns:
            True si la colonne est financière continue
        """
        col_lower = col.lower()
        
        # Patterns financiers
        financial_patterns = ['amount', 'balance', 'price', 'cost', 'fee', 'value',
                             'montant', 'solde', 'prix', 'valeur', 'cash', 'money',
                             'payment', 'paiement', 'debit', 'credit', 'fund']
        
        if any(pattern in col_lower for pattern in financial_patterns):
            return True
        
        # Vérifier via groupe sémantique
        group = self.get_semantic_group(col)
        return group == 'amount'
    
    def is_identifier_column(self, col: str) -> bool:
        """
        Détecte si une colonne est un identifiant (ID, reference, UUID).
        Ces colonnes ne devraient JAMAIS être des targets de classification.
        
        Args:
            col: Nom de la colonne
            
        Returns:
            True si la colonne est un identifiant
        """
        col_lower = col.lower()
        
        # Patterns d'identifiants
        id_patterns = ['_id', 'id_', 'identifier', 'reference', '_ref', 'ref_',
                      'uuid', 'guid', 'key', 'index']
        
        if any(pattern in col_lower for pattern in id_patterns):
            return True
        
        # Vérifier via groupes sémantiques
        group = self.get_semantic_group(col)
        return group in ['transaction_id', 'customer_id', 'account']
    
    def is_non_target_column(self, col: str) -> bool:
        """
        Détecte si une colonne ne devrait JAMAIS être une target de classification.
        Combine tous les critères : temporel, financier continu, identifiant.
        
        Args:
            col: Nom de la colonne
            
        Returns:
            True si la colonne ne doit pas être une target
        """
        return (self.is_temporal_column(col) or 
                self.is_financial_continuous_column(col) or 
                self.is_identifier_column(col))
    
    def is_potential_target_column(self, col: str) -> bool:
        """
        Détecte si une colonne pourrait potentiellement être une target de classification.
        
        Args:
            col: Nom de la colonne
            
        Returns:
            True si la colonne pourrait être une target (fraud, label, status, etc.)
        """
        col_lower = col.lower()
        
        # Patterns de target typiques
        target_patterns = ['fraud', 'label', 'target', 'class', 'is_', 'flag',
                          'status', 'result', 'outcome', 'alert', 'risk',
                          'suspicious', 'anomaly', 'threat']
        
        if any(pattern in col_lower for pattern in target_patterns):
            return True
        
        # Vérifier via groupe sémantique
        group = self.get_semantic_group(col)
        return group in ['fraud', 'status']
    
    def expand_abbreviations(self, col: str) -> str:
        """Expand les abréviations courantes (v2.0 - NOUVEAU)"""
        col_lower = col.lower()
        
        # Remplacer les abréviations en préservant les séparateurs
        for abbr, full in self.ABBREVIATIONS.items():
            # Match abbr en tant que mot complet
            pattern = r'\b' + re.escape(abbr) + r'\b'
            col_lower = re.sub(pattern, full, col_lower)
        
        return col_lower
    
    def normalize_column_name(self, col: str) -> str:
        """
        Normalise un nom de colonne (v2.0 - AMÉLIORÉ).
        
        1. Expand les abréviations
        2. Lowercase
        3. Retire underscores/espaces/tirets
        4. Détecte patterns numériques (_1, _2, squared, ratio)
        """
        # Expand abréviations d'abord
        col = self.expand_abbreviations(col)
        
        col = col.lower()
        
        # Retirer les suffixes numériques mais les capturer (pour patterns)
        # amount_1, amount_2 → amount
        # amount_ratio, amount_squared → amount
        col = re.sub(r'_(squared|cubed|sqrt|log|exp|\d+)$', '', col)
        
        # Retirer séparateurs
        col = re.sub(r'[_\s-]+', '', col)
        
        return col
    
    @lru_cache(maxsize=1024)
    def get_semantic_group_cached(self, col: str) -> str:
        """Version cachée de get_semantic_group pour performances"""
        return self._get_semantic_group_impl(col)
    
    def get_semantic_group(self, col: str) -> str:
        """
        Trouve le groupe sémantique d'une colonne (v2.0 - OPTIMISÉ).
        Utilise un cache LRU pour accélérer les lookups répétés.
        """
        if self.use_cache:
            return self.get_semantic_group_cached(col)
        else:
            return self._get_semantic_group_impl(col)
    
    def _get_semantic_group_impl(self, col: str) -> str:
        """Implémentation réelle de get_semantic_group"""
        col_norm = self.normalize_column_name(col)
        
        # Match exact
        if col_norm in self.name_to_group:
            return self.name_to_group[col_norm]
        
        # Match partiel (contains) - plus robuste
        for group, names in self.SEMANTIC_GROUPS.items():
            for name in names:
                name_norm = self.normalize_column_name(name)
                
                # Bidirectionnel: name in col OU col in name
                if name_norm in col_norm or col_norm in name_norm:
                    return group
        
        return 'unknown'
    
    def fuzzy_similarity(self, str1: str, str2: str) -> float:
        """
        Calcule la similarité fuzzy entre deux strings (v2.0 - OPTIMISÉ).
        Utilise un cache si activé.
        """
        # Créer une clé de cache canonique (ordre alphabétique)
        cache_key = tuple(sorted([str1.lower(), str2.lower()]))
        
        if self.use_cache and cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        # Calcul de similarité
        similarity = SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
        
        # Stocker dans le cache
        if self.use_cache:
            self._similarity_cache[cache_key] = similarity
        
        return similarity
    
    def clear_cache(self):
        """Vide le cache de similarité (v2.0 - NOUVEAU)"""
        if self.use_cache:
            self._similarity_cache.clear()
            # Clear LRU cache aussi
            self.get_semantic_group_cached.cache_clear()
    
    def match_columns(self, cols1: List[str], cols2: List[str], 
                     smart_threshold: bool = True) -> Dict[str, List[str]]:
        """
        Match les colonnes de deux datasets basé sur la sémantique (v2.0 - AMÉLIORÉ).
        
        Args:
            cols1: Colonnes du dataset 1
            cols2: Colonnes du dataset 2
            smart_threshold: Ajuste le seuil fuzzy selon le contexte (v2.0)
        
        Returns:
            dict: {
                'exact_matches': [(col1, col2), ...],
                'semantic_matches': [(col1, col2), ...],
                'fuzzy_matches': [(col1, col2, score), ...],
                'unmatched_cols1': [col1, ...],
                'unmatched_cols2': [col2, ...]
            }
        """
        result = {
            'exact_matches': [],
            'semantic_matches': [],
            'fuzzy_matches': [],
            'unmatched_cols1': [],
            'unmatched_cols2': []
        }
        
        cols2_remaining = set(cols2)
        
        for col1 in cols1:
            matched = False
            
            # 1. Match exact (case-insensitive)
            for col2 in list(cols2_remaining):
                if col1.lower() == col2.lower():
                    result['exact_matches'].append((col1, col2))
                    cols2_remaining.remove(col2)
                    matched = True
                    break
            
            if matched:
                continue
            
            # 2. Match sémantique (même groupe)
            group1 = self.get_semantic_group(col1)
            if group1 != 'unknown':
                for col2 in list(cols2_remaining):
                    group2 = self.get_semantic_group(col2)
                    if group1 == group2:
                        result['semantic_matches'].append((col1, col2))
                        cols2_remaining.remove(col2)
                        matched = True
                        break
            
            if matched:
                continue
            
            # 3. Fuzzy matching (v2.0 - AMÉLIORÉ avec smart threshold)
            best_match = None
            best_score = 0
            
            # Ajuster le seuil selon la longueur des noms
            threshold = self.fuzzy_threshold
            if smart_threshold:
                # Colonnes courtes (<5 chars) nécessitent un seuil plus élevé
                if len(col1) < 5:
                    threshold = min(0.85, self.fuzzy_threshold + 0.15)
            
            for col2 in cols2_remaining:
                score = self.fuzzy_similarity(col1, col2)
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = col2
            
            if best_match:
                result['fuzzy_matches'].append((col1, best_match, best_score))
                cols2_remaining.remove(best_match)
                matched = True
            
            if not matched:
                result['unmatched_cols1'].append(col1)
        
        result['unmatched_cols2'] = list(cols2_remaining)
        
        return result
    
    def calculate_semantic_similarity(self, cols1: List[str], cols2: List[str], 
                                     verbose: bool = False,
                                     smart_threshold: bool = True) -> Dict:
        """
        Calcule la similarité sémantique entre deux listes de colonnes (v2.0 - AMÉLIORÉ).
        
        Args:
            cols1: Colonnes du dataset 1
            cols2: Colonnes du dataset 2
            verbose: Afficher les détails
            smart_threshold: Utiliser le seuil intelligent pour fuzzy matching
        
        Returns:
            dict avec scores détaillés et métriques enrichies
        """
        matches = self.match_columns(cols1, cols2, smart_threshold=smart_threshold)
        
        n_exact = len(matches['exact_matches'])
        n_semantic = len(matches['semantic_matches'])
        n_fuzzy = len(matches['fuzzy_matches'])
        n_total_matches = n_exact + n_semantic + n_fuzzy
        
        max_cols = max(len(cols1), len(cols2))
        min_cols = min(len(cols1), len(cols2))
        
        # Score de similarité pondéré (v2.0 - AMÉLIORÉ)
        # Exact match = 100%, Semantic = 95% (up from 90%), Fuzzy = score * 85% (up from 80%)
        exact_score = n_exact * 1.0
        semantic_score = n_semantic * 0.95
        fuzzy_score = sum(score * 0.85 for _, _, score in matches['fuzzy_matches'])
        
        total_score = exact_score + semantic_score + fuzzy_score
        similarity = total_score / max_cols if max_cols > 0 else 0
        
        # Calcul du overlap (Jaccard-like mais sémantique)
        overlap_ratio = n_total_matches / max_cols if max_cols > 0 else 0
        
        # Précision et Recall (v2.0 - NOUVEAU)
        precision = n_total_matches / len(cols1) if len(cols1) > 0 else 0
        recall = n_total_matches / len(cols2) if len(cols2) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Score de confiance (v2.0 - NOUVEAU)
        # Plus il y a de matches exacts/sémantiques vs fuzzy, plus c'est confiant
        if n_total_matches > 0:
            confidence = (n_exact + n_semantic * 0.9) / n_total_matches
        else:
            confidence = 0.0
        
        result = {
            'similarity': similarity,
            'overlap_ratio': overlap_ratio,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confidence': confidence,
            'exact_matches': n_exact,
            'semantic_matches': n_semantic,
            'fuzzy_matches': n_fuzzy,
            'total_matches': n_total_matches,
            'n_cols1': len(cols1),
            'n_cols2': len(cols2),
            'unmatched_cols1': len(matches['unmatched_cols1']),
            'unmatched_cols2': len(matches['unmatched_cols2']),
            'details': matches
        }
        
        if verbose:
            print(f"\n📊 Similarité Sémantique (v2.0)")
            print(f"{'='*60}")
            print(f"🎯 Matches:")
            print(f"   Exact:      {n_exact} (100%)")
            print(f"   Sémantique: {n_semantic} (95%)")
            print(f"   Fuzzy:      {n_fuzzy} (85%)")
            print(f"   Total:      {n_total_matches}/{max_cols}")
            print(f"\n📈 Scores:")
            print(f"   Similarité:  {similarity:.1%}")
            print(f"   Overlap:     {overlap_ratio:.1%}")
            print(f"   Precision:   {precision:.1%}")
            print(f"   Recall:      {recall:.1%}")
            print(f"   F1-Score:    {f1_score:.1%}")
            print(f"   Confiance:   {confidence:.1%}")
        
        return result
    
    def analyze_column_groups(self, cols: List[str], verbose: bool = False) -> Dict[str, List[str]]:
        """
        Groupe les colonnes par leur sémantique (v2.0 - AMÉLIORÉ avec verbose).
        
        Args:
            cols: Liste de colonnes à analyser
            verbose: Afficher les groupes détectés
        
        Returns:
            dict: {group_name: [col1, col2, ...]}
        """
        groups = {}
        
        for col in cols:
            group = self.get_semantic_group(col)
            if group not in groups:
                groups[group] = []
            groups[group].append(col)
        
        if verbose:
            print(f"\n🔍 Analyse des Groupes Sémantiques")
            print(f"{'='*60}")
            for group, cols_list in sorted(groups.items(), key=lambda x: -len(x[1])):
                print(f"  {group:15s} ({len(cols_list):2d}): {', '.join(cols_list[:3])}")
                if len(cols_list) > 3:
                    print(f"                     ... +{len(cols_list)-3} autres")
        
        return groups
    
    def get_column_fingerprint(self, cols: List[str]) -> Set[str]:
        """
        Crée une empreinte sémantique du dataset (liste des groupes présents).
        Utile pour comparer des datasets indépendamment des noms de colonnes.
        """
        groups = set()
        for col in cols:
            group = self.get_semantic_group(col)
            if group != 'unknown':
                groups.add(group)
        return groups
    
    def get_match_quality(self, cols1: List[str], cols2: List[str]) -> str:
        """
        Évalue la qualité du match entre deux datasets (v2.0 - NOUVEAU).
        
        Returns:
            str: 'excellent' | 'good' | 'fair' | 'poor' | 'incompatible'
        """
        result = self.calculate_semantic_similarity(cols1, cols2, verbose=False)
        
        similarity = result['similarity']
        confidence = result['confidence']
        
        # Score combiné (70% similarity, 30% confidence)
        combined_score = similarity * 0.7 + confidence * 0.3
        
        if combined_score >= 0.85:
            return 'excellent'
        elif combined_score >= 0.70:
            return 'good'
        elif combined_score >= 0.50:
            return 'fair'
        elif combined_score >= 0.30:
            return 'poor'
        else:
            return 'incompatible'
    
    def suggest_column_mapping(self, cols1: List[str], cols2: List[str], 
                               top_n: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Suggère les meilleurs mappings pour chaque colonne (v2.0 - NOUVEAU).
        
        Args:
            cols1: Colonnes source
            cols2: Colonnes cible
            top_n: Nombre de suggestions par colonne
        
        Returns:
            dict: {col1: [(col2_candidate, score), ...]}
        """
        suggestions = {}
        
        for col1 in cols1:
            candidates = []
            
            # Calculer le score pour chaque colonne cible
            for col2 in cols2:
                # Score combiné: semantic + fuzzy
                group1 = self.get_semantic_group(col1)
                group2 = self.get_semantic_group(col2)
                
                if group1 == group2 and group1 != 'unknown':
                    semantic_bonus = 0.5
                else:
                    semantic_bonus = 0.0
                
                fuzzy = self.fuzzy_similarity(col1, col2)
                combined = fuzzy * 0.6 + semantic_bonus
                
                if combined >= 0.3:  # Seuil minimal
                    candidates.append((col2, combined))
            
            # Trier et garder top-N
            candidates.sort(key=lambda x: -x[1])
            suggestions[col1] = candidates[:top_n]
        
        return suggestions


# Fonctions utilitaires
def compare_datasets(cols1: List[str], cols2: List[str], 
                    fuzzy_threshold: float = 0.7, 
                    verbose: bool = True,
                    use_cache: bool = True) -> Dict:
    """
    Compare deux datasets et retourne une analyse détaillée (v2.0 - AMÉLIORÉ).
    
    Args:
        cols1: Colonnes du dataset 1
        cols2: Colonnes du dataset 2
        fuzzy_threshold: Seuil pour fuzzy matching
        verbose: Afficher les détails
        use_cache: Utiliser le cache de similarité
    
    Returns:
        dict avec similarity score et détails enrichis
    """
    matcher = ColumnMatcher(fuzzy_threshold=fuzzy_threshold, use_cache=use_cache)
    result = matcher.calculate_semantic_similarity(cols1, cols2, verbose=False)
    
    # Ajouter la qualité du match
    quality = matcher.get_match_quality(cols1, cols2)
    result['match_quality'] = quality
    
    if verbose:
        print(f"\n📊 Comparaison de Datasets (v2.0)")
        print(f"{'='*60}")
        print(f"Dataset 1: {result['n_cols1']} colonnes")
        print(f"Dataset 2: {result['n_cols2']} colonnes")
        print(f"\n🎯 Matches:")
        print(f"   Exact:      {result['exact_matches']} (100%)")
        print(f"   Sémantique: {result['semantic_matches']} (95%)")
        print(f"   Fuzzy:      {result['fuzzy_matches']} (85%)")
        print(f"   Total:      {result['total_matches']}/{max(result['n_cols1'], result['n_cols2'])}")
        print(f"\n📈 Scores:")
        print(f"   Similarité:  {result['similarity']:.1%}")
        print(f"   Overlap:     {result['overlap_ratio']:.1%}")
        print(f"   Precision:   {result['precision']:.1%}")
        print(f"   Recall:      {result['recall']:.1%}")
        print(f"   F1-Score:    {result['f1_score']:.1%}")
        print(f"   Confiance:   {result['confidence']:.1%}")
        print(f"\n🏆 Qualité du Match: {quality.upper()}")
        
        if result['unmatched_cols1'] > 0:
            print(f"\n⚠️  Non-matchées Dataset 1: {result['unmatched_cols1']}")
            for col in result['details']['unmatched_cols1'][:5]:
                print(f"      - {col}")
            if result['unmatched_cols1'] > 5:
                print(f"      ... +{result['unmatched_cols1']-5} autres")
        
        if result['unmatched_cols2'] > 0:
            print(f"\n⚠️  Non-matchées Dataset 2: {result['unmatched_cols2']}")
            for col in result['details']['unmatched_cols2'][:5]:
                print(f"      - {col}")
            if result['unmatched_cols2'] > 5:
                print(f"      ... +{result['unmatched_cols2']-5} autres")
    
    return result


if __name__ == "__main__":
    # Test du module v2.0
    print("🧪 Test du Column Matcher v2.0\n")
    
    # Exemple 1: Noms différents, même sémantique
    print("="*70)
    print("Test 1: Noms différents, sémantique identique")
    print("="*70)
    cols1 = ['transaction_amount', 'customer_id', 'timestamp', 'merchant_name', 'card_type']
    cols2 = ['amount', 'client_id', 'tx_time', 'commercant', 'card']
    compare_datasets(cols1, cols2)
    
    # Exemple 2: Colonnes partiellement différentes
    print("\n" + "="*70)
    print("Test 2: Colonnes partiellement différentes")
    print("="*70)
    cols3 = ['tx_amount', 'user_id', 'date', 'country', 'currency']
    cols4 = ['montant', 'customer_id', 'timestamp', 'pays', 'fraud_flag']
    compare_datasets(cols3, cols4)
    
    # Test 3: Abréviations (v2.0 - NOUVEAU)
    print("\n" + "="*70)
    print("Test 3: Support des abréviations")
    print("="*70)
    cols5 = ['tx_amt', 'cust_id', 'tx_time', 'acct_num']
    cols6 = ['transaction_amount', 'customer_id', 'timestamp', 'account_number']
    compare_datasets(cols5, cols6)
    
    # Test 4: Features engineered (v2.0 - NOUVEAU)
    print("\n" + "="*70)
    print("Test 4: Features numériques/engineered")
    print("="*70)
    cols7 = ['amount_ratio', 'amount_squared', 'tx_count_1', 'avg_amount_30d']
    cols8 = ['ratio_amount', 'amount_2', 'transaction_count_1h', 'amount_average']
    result = compare_datasets(cols7, cols8)
    
    # Test 5: Analyse de groupes (v2.0 - NOUVEAU)
    print("\n" + "="*70)
    print("Test 5: Analyse des groupes sémantiques")
    print("="*70)
    matcher = ColumnMatcher()
    all_cols = ['tx_id', 'tx_amount', 'tx_time', 'customer_id', 'merchant_id', 
                'card_type', 'country', 'currency', 'is_fraud', 'ip_address',
                'device_id', 'amount_ratio', 'tx_count', 'status']
    matcher.analyze_column_groups(all_cols, verbose=True)
    
    # Test 6: Suggestions de mapping (v2.0 - NOUVEAU)
    print("\n" + "="*70)
    print("Test 6: Suggestions de mapping")
    print("="*70)
    source = ['montant', 'client_id', 'pays']
    target = ['transaction_amount', 'customer_id', 'country_code', 'amount', 'user_id']
    suggestions = matcher.suggest_column_mapping(source, target, top_n=3)
    
    print("Suggestions de mapping:")
    for src, candidates in suggestions.items():
        print(f"\n  {src} →")
        for tgt, score in candidates:
            print(f"    - {tgt:25s} (score: {score:.2f})")
    
    # Test 7: Qualité du match (v2.0 - NOUVEAU)
    print("\n" + "="*70)
    print("Test 7: Évaluation de qualité")
    print("="*70)
    test_cases = [
        (['amount', 'time', 'customer'], ['amount', 'time', 'customer'], "Identique"),
        (['tx_amt', 'cust_id'], ['amount', 'customer_id'], "Abréviations"),
        (['montant', 'client'], ['amount', 'customer'], "Multi-langue"),
        (['a', 'b', 'c'], ['x', 'y', 'z'], "Incompatible")
    ]
    
    for cols_a, cols_b, description in test_cases:
        quality = matcher.get_match_quality(cols_a, cols_b)
        print(f"  {description:20s} → {quality.upper()}")
    
    print("\n" + "="*70)
    print("✅ Tests v2.0 terminés!")
    print("="*70)


def detect_column_types(df):
    """
    Détecte automatiquement les types de colonnes dans un DataFrame
    
    Args:
        df: pandas DataFrame
    
    Returns:
        dict: {type: [list of column names]}
            Ex: {'amount': ['tx_amount', 'balance'], 'time': ['timestamp'], ...}
    """
    matcher = ColumnMatcher()
    column_types = {}
    
    for col in df.columns:
        # Détecter le groupe sémantique
        normalized = matcher.normalize_column_name(col)
        
        # Vérifier si c'est un match exact
        if normalized in matcher.name_to_group:
            group = matcher.name_to_group[normalized]
            if group not in column_types:
                column_types[group] = []
            column_types[group].append(col)
            continue
        
        # Vérifier si c'est un match partiel
        matched = False
        for group, keywords in matcher.SEMANTIC_GROUPS.items():
            for keyword in keywords:
                if keyword in normalized or normalized in keyword:
                    if group not in column_types:
                        column_types[group] = []
                    column_types[group].append(col)
                    matched = True
                    break
            if matched:
                break
        
        # Si pas de match, catégoriser par type numpy
        if not matched:
            if df[col].dtype in ['int64', 'float64']:
                if 'numeric' not in column_types:
                    column_types['numeric'] = []
                column_types['numeric'].append(col)
            elif df[col].dtype == 'object':
                if 'categorical' not in column_types:
                    column_types['categorical'] = []
                column_types['categorical'].append(col)
    
    return column_types
