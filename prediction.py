import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from rapidfuzz import process

# Load dataset
df = pd.read_csv("data/previous_adoption.csv")

# Standardize column names: lowercase & replace spaces with underscores
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Initialize LabelEncoders
le_profession = LabelEncoder()
le_product = LabelEncoder()
le_scheme = LabelEncoder()

# Encoding categorical variables
if 'profession' in df.columns:
    df['profession'] = le_profession.fit_transform(df['profession'])
else:
    raise KeyError("Column 'profession' not found in dataset.")

if 'financial_product' in df.columns:
    df['financial_product'] = le_product.fit_transform(df['financial_product'])
else:
    raise KeyError("Column 'financial_product' not found in dataset.")

if 'scheme_name' in df.columns:
    df['scheme_name'] = le_scheme.fit_transform(df['scheme_name'])
else:
    raise KeyError("Column 'scheme_name' not found in dataset.")

# Store known professions for fuzzy matching
known_professions = le_profession.classes_
known_schemes = le_scheme.classes_

# Define features and target
X = df.drop(columns=['financial_product', 'scheme_name', 'origin_type'], errors='ignore')
y_product = df['financial_product']
y_scheme = df['scheme_name']

# Save feature names for consistency
feature_names = X.columns.tolist()

# Split data into train and test sets
X_train, X_test, y_train_product, y_test_product = train_test_split(X, y_product, test_size=0.2, random_state=42)
X_train_scheme, X_test_scheme, y_train_scheme, y_test_scheme = train_test_split(X, y_scheme, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
clf_product = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_product.fit(X_train, y_train_product)

clf_scheme = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_scheme.fit(X_train_scheme, y_train_scheme)

# Predictions
y_pred_product = clf_product.predict(X_test)
y_pred_proba_product = clf_product.predict_proba(X_test)

y_pred_scheme = clf_scheme.predict(X_test_scheme)
y_pred_proba_scheme = clf_scheme.predict_proba(X_test_scheme)

# Model evaluation
print("Product Accuracy:", accuracy_score(y_test_product, y_pred_product))
print(classification_report(y_test_product, y_pred_product))

print("Scheme Accuracy:", accuracy_score(y_test_scheme, y_pred_scheme))
print(classification_report(y_test_scheme, y_pred_scheme))

# Function to find closest matching profession
def get_closest_profession(input_profession, known_professions):
    best_match, score, _ = process.extractOne(input_profession, known_professions)
    if score > 70:  # Accept only if confidence is high
        return best_match
    else:
        raise ValueError(f"Unknown profession: {input_profession}. Available: {list(known_professions)}")

# Function to predict financial product and scheme with probability
def predict_financial_product_and_scheme(data):
    """
    Predicts the best financial product and scheme based on user input.
    :param data: Dictionary containing feature values.
    :return: Predicted probabilities for each financial product and scheme.
    """
    input_data = pd.DataFrame([data])

    # Standardize column names to match training data
    input_data.columns = input_data.columns.str.strip().str.lower().str.replace(" ", "_")

    # Ensure all features match
    for col in feature_names:
        if col not in input_data:
            input_data[col] = 0  # Fill missing features with 0

    input_data = input_data[feature_names]  # Ensure correct feature order

    # Ensure profession is encoded before prediction
    if 'profession' in input_data.columns:
        input_profession = data['profession']
        matched_profession = get_closest_profession(input_profession, known_professions)
        input_data['profession'] = le_profession.transform([matched_profession])[0]

    probabilities_product = clf_product.predict_proba(input_data)
    probabilities_scheme = clf_scheme.predict_proba(input_data)

    product_classes = le_product.classes_
    scheme_classes = le_scheme.classes_

    # Convert np.float64 to regular float and format to 4 decimal places
    result_product = {product: round(float(prob), 4) for product, prob in zip(product_classes, probabilities_product[0])}
    result_scheme = {scheme: round(float(prob), 4) for scheme, prob in zip(scheme_classes, probabilities_scheme[0])}
    
    return result_product, result_scheme

# Example usage
sample_input = {
    "balance_level": 4,
    "social_class": 5,
    "slow_uptrend_%": 40,
    "slow_uptrend_rs": 60,
    "slow_downtrend_%": 20,
    "slow_downtrend_rs": 30,
    "slow_sideways_%": 40,
    "slow_sideways_rs": 50,
    "fast_uptrend_%": 30,
    "fast_uptrend_rs": 55,
    "fast_downtrend_%": 20,
    "fast_downtrend_rs": 35,
    "fast_sideways_%": 50,
    "fast_sideways_rs": 50,
    "profession": "Service"
}

product_probs, scheme_probs = predict_financial_product_and_scheme(sample_input)
print("Financial Product Probabilities:", product_probs)
print("Scheme Probabilities:", scheme_probs)
