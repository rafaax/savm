import re

class SQLIFeatureExtractor:
    def __init__(self):
        self.patterns = {
            'union': re.compile(r'\bunion\b', re.IGNORECASE),
            'select_all': re.compile(r'select\s+\*', re.IGNORECASE),
            'drop': re.compile(r'\bdrop\b', re.IGNORECASE),
            'exec': re.compile(r'\bexec(?:ute)?\b', re.IGNORECASE),
            'function_exploit': re.compile(r'xp_cmdshell|sp_execute|utl_http', re.IGNORECASE),
            'single_quote': re.compile(r"'"),
            'parentheses': re.compile(r"[()]"),
            'semicolon': re.compile(r";"),
            'concat': re.compile(r"\|\||\+"),
            'hex': re.compile(r"0x[0-9a-f]+", re.IGNORECASE),
            'encoding': re.compile(r"%[0-9a-f]{2}", re.IGNORECASE),
            'system_tables': re.compile(r'information_schema\.|sys(?:columns|\.)|pg_catalog', re.IGNORECASE),
            'union_fragments': re.compile(r'union\s+(?:all\s+)?select|select\s+\*\s+from', re.IGNORECASE),
            'oracle_exploits': re.compile(r'\|\|utl_http\.request|dbms_\w+|utl_inaddr', re.IGNORECASE),
            'char_encoding': re.compile(r'char\s*$$\s*[\d\s,]+\s*$$', re.IGNORECASE),
            'time_delay': re.compile(r'\b(?:sleep|waitfor|pg_sleep)\s*$$\s*\d+\s*$$', re.IGNORECASE),
            'load_file': re.compile(r'\bload_file\s*$$|into\s+(?:out|dump)file', re.IGNORECASE)
        }

    def extract_features(self, df):
        """Extrai todas as features de SQL injection"""
        
        # Pré-processamento básico
        df['query'] = df['query'].str.lower()
        df['query_length'] = df['query'].str.len()
        
        # Features básicas
        df['has_or'] = df['query'].str.contains(r'\bor\b', regex=True).astype(int)
        df['has_and'] = df['query'].str.contains(r'\band\b', regex=True).astype(int)
        df['has_not'] = df['query'].str.contains(r'\bnot\b', regex=True).astype(int)
        df['has_union'] = df['query'].str.contains(self.patterns['union']).astype(int)
        df['has_select_all'] = df['query'].str.contains(self.patterns['select_all']).astype(int)
        df['has_comment'] = df['query'].str.contains(r'--|#|\/\*', regex=True).astype(int)
        df['has_equals'] = df['query'].str.contains('=', regex=False).astype(int)
        
        # Comandos perigosos
        df['has_drop'] = df['query'].str.contains(self.patterns['drop']).astype(int)
        df['has_exec'] = df['query'].str.contains(self.patterns['exec']).astype(int)
        df['has_function_exploit'] = df['query'].str.contains(self.patterns['function_exploit']).astype(int)
        
        # Padrões estruturais
        df['has_single_quote'] = df['query'].str.contains(self.patterns['single_quote']).astype(int)
        df['has_parentheses'] = df['query'].str.contains(self.patterns['parentheses']).astype(int)
        df['has_semicolon'] = df['query'].str.contains(self.patterns['semicolon']).astype(int)
        df['has_concat_symbols'] = df['query'].str.contains(self.patterns['concat']).astype(int)
        df['has_hex'] = df['query'].str.contains(self.patterns['hex']).astype(int)
        df['has_encoding'] = df['query'].str.contains(self.patterns['encoding']).astype(int)
        
        # Contagens
        df['space_count'] = df['query'].str.count(' ')
        df['quote_count'] = df['query'].str.count("'")
        df['special_char_count'] = df['query'].str.count(r"[\"#%&|;=()]")
        
        # Features para falsos negativos
        df['has_union_fragments'] = df['query'].str.contains(self.patterns['union_fragments']).astype(int)
        df['has_oracle_exploits'] = df['query'].str.contains(self.patterns['oracle_exploits']).astype(int)
        df['has_char_encoding'] = df['query'].str.contains(self.patterns['char_encoding']).astype(int)
        df['has_system_tables'] = df['query'].str.contains(self.patterns['system_tables']).astype(int)
        df['has_time_delay_fn'] = df['query'].str.contains(self.patterns['time_delay']).astype(int)
        df['has_load_file_fn'] = df['query'].str.contains(self.patterns['load_file']).astype(int)
        
        return df