import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Tworzenie folderu na wykresy, jeśli nie istnieje
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def perform_data_audit(X_train, y_train):
    """1. Analiza danych w Pandas: info, describe, audyt jakości."""
    print("=== RAPORT JAKOŚCI DANYCH (EDA) ===")
    profile = pd.DataFrame({
        "typ": X_train.dtypes.astype(str),
        "braki_n": X_train.isna().sum(),
        "braki_%": (100 * X_train.isna().mean()).round(2),
        "unikalne": X_train.nunique(dropna=True)
    })
    print(profile)
    print("\nStatystyki opisowe (describe.T):")
    print(X_train.describe().T)
    print(f"\nLiczba zduplikowanych wierszy w train: {X_train.duplicated().sum()}")

def check_erroneous_data(df):
    """Identyfikacja danych błędnych."""
    print("\n=== IDENTYFIKACJA DANYCH BŁĘDNYCH ===")
    for col in ['temp_max_c', 'wind_max_kmh', 'humidity_pct']:
        if col in df.columns:
            converted = pd.to_numeric(df[col], errors='coerce')
            invalid_mask = df[col].notna() & converted.isna()
            num_invalid = invalid_mask.sum()
            print(f"Kolumna {col}: wykryto {num_invalid} błędnych/nieparsowalnych wartości.")

def analyze_outliers_iqr(X_train, num_cols):
    """Analiza danych odstających metodą IQR."""
    print("\n=== ANALIZA DANYCH ODSTAJĄCYCH (IQR & BOX-PLOT) ===")
    outlier_summary = {}
    for col in num_cols:
        series = X_train[col].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_summary[col] = len(outliers)
        print(f"Kolumna '{col}': wykryto {len(outliers)} wartości odstających.")
    return outlier_summary

def plot_boxplots(X_train, num_cols):
    """
    CZYTELNE WYKRESY PUDEŁKOWE: Każda cecha ma własną niezależną oś w siatce (np. 3 kolumny).
    Dzięki temu małe wartości nie są spłaszczane przez skrajne wartości odstające w innych kolumnach.
    """
    n_cols = len(num_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        sns.boxplot(x=X_train[col].dropna(), ax=axes[i], color='skyblue', fliersize=3)
        axes[i].set_title(f"Rozkład i outliers: {col}", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("")
        axes[i].grid(True, linestyle='--', alpha=0.5)
        
    # Usuwamy puste podwykresy, jeśli siatka jest większa niż liczba cech
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle("Analiza danych odstających (Box-plot) z niezależnymi skalami", fontsize=15, y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(PLOTS_DIR, 'boxplots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Zapisano czysty wykres] -> {output_path}")

def plot_distributions_and_transforms(X_train, num_cols):
    """
    CZYTELNE HISTOGRAMY: Dzielimy cechy na mniejsze paczki po 4, 
    żeby wykresy były duże, wyraźne i łatwe do wklejenia do raportu.
    """
    batch_size = 4
    for batch_idx, i in enumerate(range(0, len(num_cols), batch_size)):
        subset = num_cols[i:i+batch_size]
        fig, axes = plt.subplots(len(subset), 2, figsize=(14, len(subset) * 3.2))
        if len(subset) == 1:
            axes = np.array([axes])
            
        for idx, col in enumerate(subset):
            data = X_train[col].dropna()
            
            # Histogram oryginalny
            sns.histplot(data, ax=axes[idx, 0], kde=True, color='skyblue', bins=30)
            axes[idx, 0].set_title(f'Oryginalny: {col}', fontsize=10, fontweight='bold')
            axes[idx, 0].set_xlabel("")
            axes[idx, 0].grid(True, linestyle='--', alpha=0.3)
            
            # Transformaty logarytmiczne
            if (data > 0).all():
                log_data = np.log1p(data)
                sns.histplot(log_data, ax=axes[idx, 1], kde=True, color='salmon', bins=30)
                axes[idx, 1].set_title(f'Log-transform (np.log1p): {col}', fontsize=10, fontweight='bold')
                axes[idx, 1].set_xlabel("")
                axes[idx, 1].grid(True, linestyle='--', alpha=0.3)
            else:
                axes[idx, 1].text(0.5, 0.5, 'Brak transformacji log\n(wartości <= 0)', 
                                horizontalalignment='center', verticalalignment='center', transform=axes[idx, 1].transAxes)
                axes[idx, 1].set_title(f'Brak transformacji log: {col}', fontsize=10, fontweight='bold')
                
        plt.tight_layout()
        output_path = os.path.join(PLOTS_DIR, f'distributions_batch_{batch_idx+1}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Zapisano paczkę histogramów] -> {output_path}")

def plot_correlations(X_train, num_cols):
    """Macierze korelacji Pearsona i Spearmana."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, method in zip(axes, ['pearson', 'spearman']):
        corr_matrix = X_train[num_cols].corr(method=method)
        sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm', ax=ax, square=True, annot=False)
        ax.set_title(f'Macierz korelacji: {method.capitalize()}', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, 'correlation_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Zapisano wykres] -> {output_path}")

def plot_pairplots(X_train, y_train, sample_size=1000):
    """Wykresy typu Pairplot."""
    df_sample = X_train[['temp_max_c', 'humidity_pct', 'wind_max_kmh', 'precip_mm']].copy()
    df_sample['target'] = y_train.values
    
    if len(df_sample) > sample_size:
        df_sample = df_sample.sample(n=sample_size, random_state=42)
        
    sns.pairplot(df_sample, hue='target', palette='viridis', diag_kind='kde', plot_kws={'alpha': 0.6, 's': 20})
    plt.suptitle("Pairplot wybranego podzbioru cech", y=1.02, fontsize=14, fontweight='bold')
    
    output_path = os.path.join(PLOTS_DIR, 'pairplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Zapisano wykres] -> {output_path}")