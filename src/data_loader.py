import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH, TARGET_COL, DATE_COL, DROP_COLS, RANDOM_STATE, TEST_SIZE

def load_data(file_path=DATA_PATH):
    """Wczytuje zbiór danych, obsługuje timestampy i zmienne czasowe."""
    df = pd.read_csv(file_path)
    
    # Decyzja dotycząca timestampów: konwertujemy do datetime i wyciągamy cechy cykliczne/okresowe (np. miesiąc),
    # a surową kolumnę daty usuwamy, ponieważ algorytmy klasyfikacyjne nie radzą sobie z surowymi stringami dat.
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
        df['month'] = df[DATE_COL].dt.month
        df['day_of_week'] = df[DATE_COL].dt.dayofweek
        
    return df

def split_data(df):
    X = df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in df.columns])
    y = df[TARGET_COL]
    
    # Usunięcie wierszy, gdzie zmienna docelowa ma NaN (jeśli takie są)
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test

def get_column_types(X_train):
    """Identyfikacja kolumn numerycznych oraz kategorycznych (nominalnych/porządkowych)."""
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    return num_cols, cat_cols