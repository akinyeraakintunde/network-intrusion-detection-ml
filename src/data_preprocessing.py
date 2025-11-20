import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('data/dataset.csv')  # Make sure dataset.csv is in the data/ folder

# Encode categorical variables (example columns)
categorical_cols = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Target column: readmitted (convert to 0/1)
df['readmitted'] = df['readmitted'].apply(lambda x: 0 if x=='NO' else 1)

# Features & target
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save processed data
pd.DataFrame(X_train, columns=X.columns).to_csv('data/X_train.csv', index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print("Data preprocessing completed!")