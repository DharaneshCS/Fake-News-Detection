import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Create model folder
# -----------------------------
os.makedirs('model', exist_ok=True)

# -----------------------------
# 2. Load dataset
# -----------------------------
print("Loading dataset...")
data = pd.read_csv('news.csv')   # make sure this file exists

print("Dataset loaded successfully!")
print(data.head())

# -----------------------------
# 3. Preprocess data
# -----------------------------
data = data.fillna('')

# Combine text columns (adjust if needed)
if 'title' in data.columns and 'text' in data.columns:
    data['content'] = data['title'] + " " + data['text']
else:
    data['content'] = data['text']

X = data['content']

# label column (must exist)
if 'label' in data.columns:
    y = data['label']
else:
    raise Exception("❌ 'label' column not found in dataset")

# -----------------------------
# 4. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Vectorization
# -----------------------------
print("Vectorizing text...")
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 6. Train model
# -----------------------------
print("Training model...")
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# -----------------------------
# 7. Evaluate
# -----------------------------
train_pred = model.predict(X_train_vec)
test_pred = model.predict(X_test_vec)

print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Testing Accuracy:", accuracy_score(y_test, test_pred))

# -----------------------------
# 8. Save model
# -----------------------------
print("Saving model...")

pickle.dump(model, open('model/model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))

print("✅ Model saved successfully!")