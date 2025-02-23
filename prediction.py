import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from rapidfuzz import process

# Load required datasets
df_customer = pd.read_excel("data/customer.xlsx")
df_discrete = pd.read_excel("data/discrete_persona.xlsx")
df_persona = pd.read_excel("data/relations/customer_discretepersona.xlsx")
df = pd.read_csv("data/previous_adoption.csv")

# Initialize LabelEncoders
le_profession = LabelEncoder()
le_product = LabelEncoder()
le_scheme = LabelEncoder()

# Encode categorical variables
df['OCCUPATION'] = le_profession.fit_transform(df['OCCUPATION'])
df['FINANCIAL_PRODUCT'] = le_product.fit_transform(df['FINANCIAL_PRODUCT'])
df['SCHEME_NAME'] = le_scheme.fit_transform(df['SCHEME_NAME'])

# Store known values for fuzzy matching
known_professions = le_profession.classes_

# Define features and target
X = df.drop(columns=['FINANCIAL_PRODUCT', 'SCHEME_NAME', 'ORIGIN_TYPE'], errors='ignore')
y_product = df['FINANCIAL_PRODUCT']
y_scheme = df['SCHEME_NAME']

# Apply small random adjustment to trend values
for col in X.columns:
    if 'TREND_%' in col or 'TREND_RS' in col:
        X[col] = X[col].apply(lambda x: np.random.randint(max(0, x-3), x+4))

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled_product = smote.fit_resample(X_scaled, y_product)
X_resampled_scheme, y_resampled_scheme = smote.fit_resample(X_scaled, y_scheme)

# Train-test split
X_train, X_test, y_train_product, y_test_product = train_test_split(X_resampled, y_resampled_product, test_size=0.2, random_state=42)
X_train_scheme, X_test_scheme, y_train_scheme, y_test_scheme = train_test_split(X_resampled_scheme, y_resampled_scheme, test_size=0.2, random_state=42)

# Train models
clf_product = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
clf_product.fit(X_train, y_train_product)

clf_scheme = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
clf_scheme.fit(X_train_scheme, y_train_scheme)

# Function to get profession using fuzzy matching
def get_closest_profession(input_profession):
    input_profession = str(input_profession).lower()  # Convert to lowercase
    best_match, score, _ = process.extractOne(input_profession, known_professions)
    return best_match if score > 70 else None



# Function to predict using CIF_NO
def predict_from_cif(cif_no):
    # Get discrete persona ID from df_persona instead of df_customer
    persona_info = df_persona[df_persona['CIF_NO'] == cif_no]
    if persona_info.empty:
        raise ValueError("CIF_NO not found in persona data")
    
    discrete_persona_id = persona_info.iloc[0]['DISCRETE_PERSONA_ID']
    
    # Get discrete persona details
    discrete_info = df_discrete[df_discrete['DISCRETE_PERSONA_ID'] == discrete_persona_id]
    if discrete_info.empty:
        raise ValueError("Discrete persona not found")
    
    # Get continuous persona details
    continuous_info = persona_info  # Now, df_persona already holds relevant information
    if continuous_info.empty:
        raise ValueError("Continuous persona not found")
    
    # Prepare input data
    input_data = discrete_info[['BALANCE_LEVEL', 'SOCIAL_CLASS', 'OCCUPATION']].copy()
    continuous_cols = ['SLOW_UPTREND_%', 'SLOW_UPTREND_RS', 'SLOW_DOWNTREND_%', 'SLOW_DOWNTREND_RS',
                       'FAST_UPTREND_%', 'FAST_UPTREND_RS', 'FAST_DOWNTREND_%', 'FAST_DOWNTREND_RS']
    
    input_data[continuous_cols] = continuous_info[continuous_cols].values

    # Standardize column names to uppercase
    input_data.columns = input_data.columns.str.strip().str.upper().str.replace(" ", "_")

    # Ensure all features match
    for col in X.columns:
        if col not in input_data:
            input_data[col] = 0

    # Adjust trend values
    for col in continuous_cols:
        input_data[col] = np.random.randint(max(0, input_data[col].values[0]-5), input_data[col].values[0]+6)

    # Encode profession
    matched_profession = get_closest_profession(str(input_data['OCCUPATION'].values[0]).lower())
    if matched_profession:
        input_data['OCCUPATION'] = le_profession.transform([matched_profession])[0]
    else:
        raise ValueError("Unknown profession")

    # Normalize input
    input_scaled = scaler.transform(input_data[X.columns])

    # Predict probabilities
    probabilities_product = clf_product.predict_proba(input_scaled)
    probabilities_scheme = clf_scheme.predict_proba(input_scaled)

    # Format output
    result_product = {le_product.classes_[i]: round(float(prob), 4) for i, prob in enumerate(probabilities_product[0])}
    result_scheme = {le_scheme.classes_[i]: round(float(prob), 4) for i, prob in enumerate(probabilities_scheme[0])}

    return result_product, result_scheme

def evaluate_model(model, X_test, y_test, label_encoder, model_name):
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification Report
    class_report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_
    )
    
    # Print results
    print(f"\n===== {model_name} Evaluation Metrics =====\n")
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:\n", class_report)

# Evaluate Product Model
evaluate_model(clf_product, X_test, y_test_product, le_product, "Financial Product Model")

# Evaluate Scheme Model
evaluate_model(clf_scheme, X_test_scheme, y_test_scheme, le_scheme, "Scheme Model")


# Example usage
cif_no = 'CIF102400'
product_probs, scheme_probs = predict_from_cif(cif_no)
print("Financial Product Probabilities:", product_probs)
print("Scheme Probabilities:", scheme_probs)
