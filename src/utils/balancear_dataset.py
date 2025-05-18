import pandas as pd

def corrigir_rotulos_csv(input_csv, output_csv, coluna_rotulo='label'):

    df = pd.read_csv(input_csv)
    

    df[coluna_rotulo] = df[coluna_rotulo].astype(str).str.strip().str.replace('.0', '')
    

    print(f"Rótulos únicos antes do mapeamento: {df[coluna_rotulo].unique()}")
    

    rotulos_validos = {'0': '0', '1': '1'}
    
    df[coluna_rotulo] = df[coluna_rotulo].map(rotulos_validos)
    
    print(f"Rótulos únicos após o mapeamento: {df[coluna_rotulo].unique()}")
    print(f"Total de entradas antes de remover NaN: {len(df)}")
    
    df_corrigido = df.dropna(subset=[coluna_rotulo])
    
    print(f"Entradas restantes após limpeza: {len(df_corrigido)}")
    
    df_corrigido.to_csv(output_csv, index=False)
    print(f"Rótulos corrigidos e arquivo salvo como {output_csv}")

input_csv = 'mocks/dataset-concatenada_v2.csv'
output_csv = 'mocks/dataset-concatenada_v3.csv'
corrigir_rotulos_csv(input_csv, output_csv)