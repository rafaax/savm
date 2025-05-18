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
            'load_file': re.compile(r'\bload_file\s*$$|into\s+(?:out|dump)file', re.IGNORECASE),
            'sleep': re.compile(r'\bsleep\(\d+\)', re.IGNORECASE),
            'waitfor': re.compile(r'\bwaitfor delay\b', re.IGNORECASE),
            'benchmark': re.compile(r'benchmark\(\d+,', re.IGNORECASE),
            'information_schema': re.compile(r'information_schema', re.IGNORECASE),
            'stacked_queries': re.compile(r';\s*select\b', re.IGNORECASE),
            'delete': re.compile(r'\bdelete\b', re.IGNORECASE),
            'truncate': re.compile(r'\btruncate\b', re.IGNORECASE),
            'alter': re.compile(r'\balter\b', re.IGNORECASE),
            'update': re.compile(r'\bupdate\b', re.IGNORECASE),
            'insert': re.compile(r'\binsert\b', re.IGNORECASE),
        }

    def extract_features(self, df):
        if 'query' in df.columns:
        
            # Pré-processamento básico
            df['query'] = df['query'].str.lower()
            df['query_length'] = df['query'].str.len()

            # Função para verificar e converter
            def safe_contains(column, pattern):
                result = df['query'].str.contains(pattern, regex=True).fillna(False)
                result = result.infer_objects(copy=False)
                return result.astype(int)

            # Features básicas
            df['has_or'] = safe_contains(df['query'], r'\bor\b')
            df['has_and'] = safe_contains(df['query'], r'\band\b')
            df['has_not'] = safe_contains(df['query'], r'\bnot\b')
            df['has_union'] = safe_contains(df['query'], self.patterns['union'])
            df['has_select_all'] = safe_contains(df['query'], self.patterns['select_all'])
            df['has_comment'] = safe_contains(df['query'], r'--|#|\/\*')
            df['has_equals'] = safe_contains(df['query'], '=')

            # Comandos perigosos
            df['has_drop'] = safe_contains(df['query'], self.patterns['drop'])
            df['has_exec'] = safe_contains(df['query'], self.patterns['exec'])
            df['has_function_exploit'] = safe_contains(df['query'], self.patterns['function_exploit'])

            # Padrões estruturais
            df['has_single_quote'] = safe_contains(df['query'], self.patterns['single_quote'])
            df['has_parentheses'] = safe_contains(df['query'], self.patterns['parentheses'])
            df['has_semicolon'] = safe_contains(df['query'], self.patterns['semicolon'])
            df['has_concat_symbols'] = safe_contains(df['query'], self.patterns['concat'])
            df['has_hex'] = safe_contains(df['query'], self.patterns['hex'])
            df['has_encoding'] = safe_contains(df['query'], self.patterns['encoding'])

            # Contagens
            df['space_count'] = df['query'].str.count(' ')
            df['quote_count'] = df['query'].str.count("'")
            df['special_char_count'] = df['query'].str.count(r'["#%&|;=()]')

            df['has_union_fragments'] = safe_contains(df['query'], self.patterns['union_fragments'])
            df['has_oracle_exploits'] = safe_contains(df['query'], self.patterns['oracle_exploits'])
            df['has_char_encoding'] = safe_contains(df['query'], self.patterns['char_encoding'])
            df['has_system_tables'] = safe_contains(df['query'], self.patterns['system_tables'])
            df['has_time_delay_fn'] = safe_contains(df['query'], self.patterns['time_delay'])
            df['has_load_file_fn'] = safe_contains(df['query'], self.patterns['load_file'])

            df['has_sleep'] = safe_contains(df['query'], self.patterns['sleep'])
            df['has_waitfor'] = safe_contains(df['query'], self.patterns['waitfor'])
            df['has_benchmark'] = safe_contains(df['query'], self.patterns['benchmark'])
            df['has_information_schema'] = safe_contains(df['query'], self.patterns['information_schema'])
            df['has_stacked_queries'] = safe_contains(df['query'], self.patterns['stacked_queries'])

            df['has_delete'] = safe_contains(df['query'], self.patterns['delete'])
            df['has_truncate'] = safe_contains(df['query'], self.patterns['truncate'])
            df['has_alter'] = safe_contains(df['query'], self.patterns['alter'])
            df['has_update'] = safe_contains(df['query'], self.patterns['update'])
            df['has_insert'] = safe_contains(df['query'], self.patterns['insert'])

        else:
            print("A coluna 'query' não existe no dataframe.")

        return df