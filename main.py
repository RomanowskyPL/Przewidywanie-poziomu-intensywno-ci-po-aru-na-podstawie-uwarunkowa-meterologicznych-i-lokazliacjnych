from src.data_loader import load_data, split_data, get_column_types
from src.eda import (
    perform_data_audit, 
    check_erroneous_data, 
    analyze_outliers_iqr, 
    plot_distributions_and_transforms, 
    plot_boxplots, 
    plot_correlations, 
    plot_pairplots
)
from src.preprocessing import build_preprocessor
from src.model import get_benchmark_model, train_and_evaluate, test_on_multiple_seeds, test_on_train_sizes

def main():
    print("==========================================")
    print("ROZPOCZĘCIE REALIZACJI ETAPU 2 (EDA & WERYFIKACJA)")
    print("==========================================")
    
    # 1. Ładowanie danych
    df = load_data()
    
    # 2. Podział na zbiór treningowy i testowy przed preprocessingiem
    X_train, X_test, y_train, y_test = split_data(df)
    num_cols, cat_cols = get_column_types(X_train)
    
    print(f"Liczba cech numerycznych: {len(num_cols)}")
    print(f"Liczba cech kategorycznych: {len(cat_cols)}")
    
    # 3. Audyt Pandas i błędów
    perform_data_audit(X_train, y_train)
    check_erroneous_data(X_train)
    
    # 4. Analiza odstających (IQR)
    analyze_outliers_iqr(X_train, num_cols)
    
    # 5. Generowanie i zapisywanie wizualizacji do folderu 'plots/'
    print("\n--- GENEROWANIE I ZAPIS WYKRESÓW DO FOLDERU PLOTS/ ---")
    plot_boxplots(X_train, num_cols)
    plot_distributions_and_transforms(X_train, num_cols)
    plot_correlations(X_train, num_cols)
    plot_pairplots(X_train, y_train)
    
    # 6. Preprocessing (imputacja braków, skalowanie, One-Hot)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    
    # 7. Model benchmarkowy i testy stabilności
    model = get_benchmark_model(preprocessor)
    train_and_evaluate(model, X_train, X_test, y_train, y_test)
    
    test_on_multiple_seeds(preprocessor, X_train, X_test, y_train, y_test)
    test_on_train_sizes(preprocessor, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()