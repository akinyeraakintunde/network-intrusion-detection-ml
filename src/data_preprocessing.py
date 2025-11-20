import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Convert label to binary: attack = 1, normal = 0
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()
    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])

    return df