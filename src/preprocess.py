from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd

feature_store = {
    'numerical_features': ['MonthlyCharges', 'TotalCharges'],
    'nominal_categorical_features':['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaymentMethod', 'Contract'],
    'binary_categorical_features':['Partner', 'Dependents', 'PaperlessBilling','PhoneService'],
    'target': 'churn'
}

# loading the dataset
def load_dataset(path):
    df = pd.read_csv(path)
    df.drop(columns=['customerID'], inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df

def preprocess_data(df):
    #split dataset
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[feature_store['target']]), df[feature_store['target']], test_size=0.2, random_state=42)
    transformer = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), feature_store['numerical_features']),
        ('nominal_cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), feature_store['nominal_categorical_features']),
        ('binary_cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='if_binary', sparse_output=False))
        ]), feature_store['binary_categorical_features'])
    ])
    X_train_processed = transformer.fit_transform(X_train)
    X_test_processed = transformer.transform(X_test)
    # Encode target variable
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    #saving the preprocessed data to csv files
    X_train_processed_df = pd.DataFrame(X_train_processed)
    X_test_processed_df = pd.DataFrame(X_test_processed)
    #combine target variable with processed features
    X_train_processed_df[feature_store['target']] = y_train_encoded
    X_train_processed_df.to_csv('./data/X_train_processed.csv', index=False)

    X_test_processed_df[feature_store['target']] = y_test_encoded
    X_test_processed_df.to_csv('./data/X_test_processed.csv', index=False)

    return X_train_processed, X_test_processed, y_train_encoded, y_test_encoded

