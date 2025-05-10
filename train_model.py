import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
from src.features.sqli_features import SQLIFeatureExtractor
from src.features.text_features import TextFeatureExtractor

def load_data(data_path: str) -> pd.DataFrame:
    """Carrega o dataset de treinamento"""
    df = pd.read_csv(data_path)
    
    # Verifica colunas necessárias
    if 'query' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset deve conter colunas 'query' e 'label'")
    
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessa os dados e extrai features"""
    # Inicializa extratores
    sqli_extractor = SQLIFeatureExtractor()
    text_extractor = TextFeatureExtractor()
    
    # Extrai features
    df = sqli_extractor.extract(df)
    df = text_extractor.extract(df, fit_models=True)
    
    return df

def train_model(df: pd.DataFrame, model_dir: str = "models/production"):
    """Treina e salva o modelo"""
    # Separa features e target
    X = df.drop(['query', 'label'], axis=1, errors='ignore')
    y = df['label']
    
    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Treina o modelo
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Avaliação
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Salva o modelo
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'features': list(X.columns),
        'metadata': {
            'dataset_size': len(df),
            'accuracy': (y_pred == y_test).mean()
        }
    }
    
    joblib.dump(model_data, model_dir / "best_model.joblib")
    print(f"\nModelo salvo em {model_dir/'best_model.joblib'}")

if __name__ == "__main__":
    # Configurações
    DATA_PATH = "mocks/dataset.csv"  # Altere para seu dataset
    MODEL_DIR = "models/production"
    
    # Pipeline completo
    print("Carregando dados...")
    df = load_data(DATA_PATH)
    
    print("\nPreprocessando dados e extraindo features...")
    df = preprocess_data(df)
    
    print("\nTreinando modelo...")
    train_model(df, MODEL_DIR)