from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(num_cols, cat_cols):
    """
    Buduje potok transformacji uwzględniający:
    - Podstawowe i zaawansowane metody imputacji braków (zgodnie z wytycznymi)
    - Skalowanie danych numerycznych (StandardScaler)
    - Kodowanie zmiennych kategorycznych (OneHotEncoder)
    """
    
    # Przykłady 3 podstawowych metod imputacji (do wyboru / testów w raporcie):
    # - SimpleImputer(strategy='mean')     
    # - SimpleImputer(strategy='median')   
    # - SimpleImputer(strategy='most_frequent') 
    
    # Przykłady 2 zaawansowanych metod imputacji:
    # - KNNImputer(n_neighbors=5)          
    # - IterativeImputer                   
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),  
        ('scaler', StandardScaler())             
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore'))    
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    
    return preprocessor