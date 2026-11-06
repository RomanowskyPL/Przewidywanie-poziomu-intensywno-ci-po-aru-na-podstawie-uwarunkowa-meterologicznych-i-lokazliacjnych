import pandas as pd
import numpy as np
import os
from config import DATA_PATH, BASE_DIR

def inject_missing_data(input_path, output_path, missing_fraction=0.10, seed=42):
    """
    Ładuje zbiór danych, losowo wprowadza braki w odpowiednim procencie (np. 10%)
    i zapisuje nowy zbiór do pliku.
    """
    print(f"Wczytywanie zbioru z: {input_path}")
    df = pd.read_csv(input_path)
    
    np.random.seed(seed)
    
    # Obliczamy łączną liczbę komórek
    total_cells = df.size
    target_missing = int(missing_fraction * total_cells)
    
    # Losujemy pozycje (indeksy 1D) i konwertujemy na (wiersz, kolumna)
    indices = np.random.choice(total_cells, target_missing, replace=False)
    row_indices, col_indices = np.unravel_index(indices, df.shape)
    
    df_missing = df.copy()
    
    # Wstrzykiwanie wartości NaN
    for r, c in zip(row_indices, col_indices):
        df_missing.iat[r, c] = np.nan
        
    actual_missing_pct = df_missing.isnull().sum().sum() / total_cells * 100
    
    print(f"Ilość wygenerowanych braków: {actual_missing_pct:.2f}%")
    
    df_missing.to_csv(output_path, index=False)
    print(f"Zapisano nowy zbiór do: {output_path}")

if __name__ == "__main__":
    # Definiujemy ścieżkę wyjściową w katalogu 'data'
    output_file = os.path.join(BASE_DIR, 'data', 'wildfire_multi_region_dataset_missing.csv')
    inject_missing_data(DATA_PATH, output_file)