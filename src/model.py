from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from src.config import SEEDS_TO_TEST, TRAIN_SIZES, RANDOM_STATE

def get_benchmark_model(preprocessor, seed=RANDOM_STATE):
    """Tworzy model benchmarkowy bez strojenia parametrów."""
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=seed))
    ])

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """Trenuje model i wypisuje raport klasyfikacji."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("--- Raport Klasyfikacji Benchmarkowej ---")
    print(classification_report(y_test, y_pred))
    return model

def test_on_multiple_seeds(preprocessor, X_train, X_test, y_train, y_test):
    """Testuje stabilność modelu na różnych ziarnach losowości."""
    print("\n--- Testy modelu na różnych seedach ---")
    for s in SEEDS_TO_TEST:
        model = get_benchmark_model(preprocessor, seed=s)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Seed {s}: Accuracy = {acc:.4f}")

def test_on_train_sizes(preprocessor, X_train_full, X_test, y_train_full, y_test):
    """Testuje model przy różnych proporcjach danych uczących."""
    print("\n--- Wpływ rozmiaru zbioru treningowego na skuteczność ---")
    for size in TRAIN_SIZES:
        if size == 1.0:
            continue
        
        # Ograniczanie ilości próbek uczących (testujemy różne ratio dla X_train)
        Xt, _, yt, _ = train_test_split(
            X_train_full, y_train_full, train_size=size, random_state=RANDOM_STATE, stratify=y_train_full
        )
        
        model = get_benchmark_model(preprocessor, seed=RANDOM_STATE)
        model.fit(Xt, yt)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Train size {size*100:.0f}% (próbek: {len(Xt)}): Accuracy = {acc:.4f}")