import re

class SQLIFeatureExtractor:
    def __init__(self):
        self.patterns = {
            'union': re.compile(r'\bunion\b', re.IGNORECASE), # Detecta a presença de "UNION", que pode ser usado para combinar resultados de mais de uma consulta SELECT em um único conjunto de resultados
            'select_all': re.compile(r'select\s+\*', re.IGNORECASE), # Identifica o uso de "SELECT *", que seleciona todos os campos de uma tabela
            'drop': re.compile(r'\bdrop\b', re.IGNORECASE), # Verifica a presença do comando "DROP", usado para deletar um banco de dados ou tabela
            'exec': re.compile(r'\bexec(?:ute)?\b', re.IGNORECASE), # Procura por "EXEC" ou "EXECUTE", usado para executar um bloco de código SQL
            'function_exploit': re.compile(r'xp_cmdshell|sp_execute|utl_http', re.IGNORECASE), #  Detecta funções potencialmente perigosas como "xp_cmdshell", "sp_execute", que permitem a execução de comandos no sistema operacional
            'single_quote': re.compile(r"'"), # Verifica a presença de aspas simples ('), frequentemente usadas para delimitar strings em SQL
            'parentheses': re.compile(r"[()]"), # Identifica parênteses, usados para agrupar expressões ou listas de parâmetros
            'semicolon': re.compile(r";"), #  Detecta ponto e vírgula (;), utilizado para terminar instruções SQL, permitindo assim consultas empilhadas
            'concat': re.compile(r"\|\||\+"), # Procura operadores de concatenação, como "||" ou "+", que unem strings
            'hex': re.compile(r"0x[0-9a-f]+", re.IGNORECASE), # Identifica literais hexadecimais, que podem ser usados para representar dados binários
            'encoding': re.compile(r"%[0-9a-f]{2}", re.IGNORECASE), # Detecta caracteres codificados em URL (como %20), que podem esconder intenções maliciosas
            'system_tables': re.compile(r'information_schema\.|sys(?:columns|\.)|pg_catalog', re.IGNORECASE), #  Procura referências a tabelas de sistema, como "information_schema", que fornecem metadados do banco de dados
            'union_fragments': re.compile(r'union\s+(?:all\s+)?select|select\s+\*\s+from', re.IGNORECASE), # Detecta fragmentos complexos de consultas UNION
            'oracle_exploits': re.compile(r'\|\|utl_http\.request|dbms_\w+|utl_inaddr', re.IGNORECASE), # Procura por funções específicas a Oracle utilizadas em ataques, como "utl_http"
            'char_encoding': re.compile(r'char\s*$$\s*[\d\s,]+\s*$$', re.IGNORECASE), #  Detecta usos maliciosos da função CHAR, que pode esconder strings
            'time_delay': re.compile(r'\b(?:sleep|waitfor|pg_sleep)\s*$$\s*\d+\s*$$', re.IGNORECASE), # Identifica tentativas de atrasar a execução com funções como "SLEEP", usadas em ataques baseados em tempo.
            'load_file': re.compile(r'\bload_file\s*$$|into\s+(?:out|dump)file', re.IGNORECASE), # Detecta tentativas de carregar ou manipular arquivos
            'sleep': re.compile(r'\bsleep\(\d+\)', re.IGNORECASE), # Procura por chamadas da função "SLEEP", específicas para ataques de injeção baseada em tempo
            'waitfor': re.compile(r'\bwaitfor delay\b', re.IGNORECASE), # Identifica o uso de "WAITFOR DELAY", outro método de introduzir atrasos
            'benchmark': re.compile(r'benchmark\(\d+,', re.IGNORECASE), #  Detecta uso de "BENCHMARK", que mede o tempo de execução de funções SQL
            'information_schema': re.compile(r'information_schema', re.IGNORECASE), # Verifica referências específicas a "information_schema", que pode dar informações sobre o banco
            'stacked_queries': re.compile(r';\s*select\b', re.IGNORECASE), # Detecta consultas empilhadas que permitem executar múltiplas instruções SQL em uma única execução
            'delete': re.compile(r'\bdelete\b', re.IGNORECASE), # Procura por o comando "DELETE", usado para excluir registros de uma tabela
            'truncate': re.compile(r'\btruncate\b', re.IGNORECASE), # Identifica o uso de "TRUNCATE", que remove todos os dados de uma tabela
            'alter': re.compile(r'\balter\b', re.IGNORECASE), #  Detecta "ALTER", usado para modificar a estrutura de tabelas
            'update': re.compile(r'\bupdate\b', re.IGNORECASE), # Verifica a presença de "UPDATE", empregado para modificar registros existentes
            'insert': re.compile(r'\binsert\b', re.IGNORECASE), #  Procura o comando "INSERT", usado para adicionar registros a uma tabela
            'null_byte': re.compile(r'%00|\x00', re.IGNORECASE),
            'second_order': re.compile(r'\b(?:select|insert)\b.*\b(?:select|insert)\b', re.IGNORECASE),
            'mysql_commands': re.compile(r'\b(?:load_file|into\s+outfile|into\s+dumpfile)\b', re.IGNORECASE), # Comandos específicos de bancos
            'mssql_commands': re.compile(r'\b(?:openrowset|opendatasource)\b', re.IGNORECASE) # Comandos específicos de bancos
        } 

    def extract_features(self, df):

        def safe_contains(column, pattern):
            result = df['query'].str.contains(pattern, regex=True).fillna(False)
            result = result.infer_objects(copy=False)
            return result.astype(int)
        
        if 'query' in df.columns:

            df['query'] = df['query'].str.lower() # jogando tudo para lowercase
            df['query_length'] = df['query'].str.len() # pegando o tamanho da 'query'

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
            df['has_unconditional_delete'] = safe_contains(df['query'], r'\bdelete\s+from\b(?!.*\bwhere\b)')
            df['has_second_order'] = safe_contains(df['query'], self.patterns['second_order'])

            df['has_always_true'] = safe_contains(df['query'], r'\b(1\s*=\s*1|true|not\s+false)\b')
            df['has_or_injection'] = safe_contains(df['query'], r'\bor\b\s+\d+\s*=\s*\d+')
            df['has_time_delay'] = safe_contains(df['query'], r'\b(sleep|waitfor|pg_sleep|benchmark)$$\d+$$')
            df['has_hex_injection'] = safe_contains(df['query'], r'0x[0-9a-f]{4,}')
            df['operator_count'] = df['query'].str.count(r'[=<>!]')

            df['has_destructive_command'] = (
                df['has_drop'] | df['has_truncate'] | 
                df['has_unconditional_delete']
            ).astype(int)

        else:
            print("A coluna 'query' não existe no dataframe.")

        return df