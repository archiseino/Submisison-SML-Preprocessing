# automate_nama.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Drop unused columns
    df.drop(['PID', 'Order', 'Mas Vnr Type', 'Pool QC', 'Misc Feature', 'Alley', 'Fence'], axis=1, inplace=True)

    # Define ordinal and one-hot columns
    ordinal_cols = ['Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond', 'Heating QC', 'Kitchen Qual', 'Fireplace Qu', 'Garage Qual', 'Garage Cond']
    one_hot_cols = [col for col in df.select_dtypes(include=['object']).columns if col not in ordinal_cols]

    # Numerical columns (exclude SalePrice if you're using it as target)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'SalePrice' in num_cols:
        num_cols.remove('SalePrice')

    # Pipelines
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    one_hot_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    ordinal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('one_hot', one_hot_pipe, one_hot_cols),
        ('ordinal', ordinal_pipe, ordinal_cols)
    ])

    # Fit and transform
    X = preprocessor.fit_transform(df)

    # Convert to DataFrame
    df_processed = pd.DataFrame(X)
    if 'SalePrice' in df.columns:
        df_processed['SalePrice'] = df['SalePrice'].values

    # Save result
    df_processed.to_csv(output_path, index=False)
    joblib.dump(preprocessor, 'preprocessor.joblib')

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    preprocess_data(input_path, output_path)
