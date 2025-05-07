import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def remove_illegal_chars(df):
    # Apenas colunas de texto
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype(str).apply(lambda x: ILLEGAL_CHARACTERS_RE.sub('', x))
    return df

df = pd.read_csv('mocks/dataset.csv')
# nltk.download('punkt_tab')
scaler = MinMaxScaler()

# PRÉ PROCESSAMENTO DO DATASET 

df['query'] = df['query'].str.lower()
df['tokens'] = df['query'].apply(word_tokenize)


# ANÁLISE EXPLORATÓRIA

legit_tokens = Counter(token for tokens in df[df['label'] == 0]['tokens'] for token in tokens)
malicious_tokens = Counter(token for tokens in df[df['label'] == 1]['tokens'] for token in tokens)
print("Termos mais comuns em queries maliciosas:", malicious_tokens.most_common(10))
df['length'] = df['query'].str.len()
df.boxplot(column='length', by='label')

malicious_text = ' '.join(df[df['label'] == 1]['query'])
WordCloud().generate(malicious_text).to_image()

# FEATURE ENGINEERING

vectorizer = TfidfVectorizer(max_features=1000)

X = vectorizer.fit_transform(df['query'])
y = df['label']


# Atributos baseados em padrões de ataques de sql injection

# comparadores lógicos suspeitos 
df['has_or'] = df['query'].str.contains(r'\bOR\b', case=False, regex=True).astype(int)
df['has_and'] = df['query'].str.contains(r'\bAND\b', case=False, regex=True).astype(int)
df['has_not'] = df['query'].str.contains(r'\bNOT\b', case=False, regex=True).astype(int)

# técnicas de bypass

df['has_comment'] = df['query'].str.contains(r'--|#|\/\*', regex=True).astype(int)  # Inclui /* */

df['has_equals'] = df['query'].str.contains('=', regex=False).astype(int)
# comandos perigosos 
df['has_drop'] = df['query'].str.contains(r'\bDROP\b', case=False, regex=True).astype(int)
df['has_exec'] = df['query'].str.contains(r'\bEXEC\b|\bEXECUTE\b', case=False, regex=True).astype(int)
df['has_union'] = df['query'].str.contains(r'\bUNION\b', case=False, regex=True).astype(int)
df['has_select_all'] = df['query'].str.contains(r'SELECT\s*\*', case=False, regex=True).astype(int)

df['has_single_quote'] = df['query'].str.contains("'").astype(int)
df['has_parentheses'] = df['query'].str.contains(r"[()]").astype(int)
df['has_special_minimal'] = df['query'].str.contains(r"^[%&|()';=]+$").astype(int)
df['has_hex'] = df['query'].str.contains(r"0x[0-9a-fA-F]+").astype(int)

df['has_concat_symbols'] = df['query'].str.contains(r"\|\||\+|\|\|").astype(int)
df['has_semicolon'] = df['query'].str.contains(r";").astype(int)
df['has_function_exploit'] = df['query'].str.contains(
    r"utl_http|exec_cmd|xp_cmdshell|sp_execute", case=False
).astype(int)

# injeção de função
df['has_function'] = df['query'].str.contains(r'\bCONCAT\b|\bSUBSTRING\b|\bWAITFOR\b|\bSLEEP\b|\bDATABASE\b', case=False, regex=True).astype(int)

# tautologia classica -> 1 = 1, a = a 
df['has_tautology'] = df['query'].str.contains(r'\d\s*=\s*\d|\'\w\'\s*=\s*\'\w\'', regex=True).astype(int)

# queries maliciosas costumam ter muito espaços
df['space_count'] = df['query'].str.count(' ')

# indicativo de concatenação maliciosa
df['quote_count'] = df['query'].str.count('\'')

# tamanho da query = tamanho mais longo costuma ser maliciosa
df['query_length'] = df['query'].str.len()

df['has_time_delay'] = df['query'].str.contains(
    r'\bSLEEP\s*$$\d+$$|\bWAITFOR\s*DELAY\s*\'\d+:\d+:\d+\'', case=False, regex=True
).astype(int)

# tentativa de bypass com encoding (ex: %27 para aspas)
df['has_encoding'] = df['query'].str.contains(r'%[0-9A-Fa-f]{2}', regex=True).astype(int)

df = remove_illegal_chars(df)

# df.to_excel('results/dataset.xlsx', index=False)

# --- COMBINAR TF-IDF COM FEATURES PERSONALIZADAS ---

custom_features = df[[
    'has_or',                    # Detecta o operador OR - usado para contornar condições (ex.: 'OR 1=1')
    'has_and',                   # Identifica o operador AND - comum em condições maliciosas
    'has_not',                   # Detecta o operador NOT - usado para inverter lógicas em ataques
    'has_comment',               # Identifica comentários SQL (--, #, /* */) usados para truncar queries
    'has_equals',                # Detecta o símbolo '=' - frequentemente usado em tautologias (ex.: '1=1')
    'has_drop',                  # Detecta comandos DROP (ex.: DROP TABLE users)
    'has_exec',                  # Identifica EXEC/EXECUTE - execução arbitrária de comandos
    'has_union',                 # Detecta UNION - usado para unir queries maliciosas   
    'has_select_all',            # Identifica 'SELECT *' - comum em tentativas de extração de dados 
    'has_function',              # Detecta funções como CONCAT, SUBSTRING (usadas em ataques)
    'has_tautology',             # Identifica sempre-verdadeiros (ex.: '1=1', ''='')
    'space_count',               # Conta espaços - queries injetadas costumam ter mais espaços
    'quote_count',               # Conta aspas - desbalanceamento indica concatenação maliciosa
    'query_length',              # Tamanho da query - ataques geralmente são mais longos
    'has_time_delay',            # Detecta delays (SLEEP, WAITFOR) usados em ataques Time-Based
    'has_encoding',              # Detecta codificação (ex.: hex 0x4A4B)
    'has_single_quote',          # Detecta aspas simples isoladas (')
    'has_parentheses',           # Detecta parênteses () sozinhos
    'has_special_minimal',       # Padrões mínimos como %&|()' etc.
    'has_hex',                   # Detecta codificação hexadecimal
    'has_concat_symbols',        # Símbolos de concatenação (||, +)
    'has_semicolon',             # Ponto e vírgula isolado
    'has_function_exploit'       # Funções específicas de BD (utl_http, xp_cmdshell)
]]

tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()) # Converter TF-IDF para DataFrame

custom_features_scaled = scaler.fit_transform(custom_features)

X_combined = pd.concat([tfidf_df, pd.DataFrame(custom_features_scaled, columns=custom_features.columns)], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.3, random_state=42
)

model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))

results = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': y_pred
}, index=X_test.index)  # Usar o mesmo índice que X_test

# 2. Juntar com o DataFrame original usando o índice
false_negatives = df.loc[results[(results['true_label'] == 1) & (results['predicted_label'] == 0)].index]

print("Falsos negativos (ataques não detectados):")
false_negatives[['query']].to_csv('results/falsos_negativos_detalhados.csv', index=False, encoding='utf-8')

# Veja a distribuição de tamanhos das queries maliciosas não detectadas
false_negatives['length'] = false_negatives['query'].str.len()
print("\nEstatísticas de comprimento:")
print(false_negatives['length'].describe())



common_patterns = Counter(false_negatives['query'].str[:30])  # Analisa os primeiros 30 caracteres
print("\nPadrões mais comuns nos falsos negativos:")
print(common_patterns.most_common(10))