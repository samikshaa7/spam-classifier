import pandas as pd
import pickle

# ===== LOAD DATA =====
# Use ONE of the two blocks above depending on your file
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1','v2']]
df.columns = ['label','message']

# ===== PREPROCESS =====
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

# ===== TEXT → NUMBERS =====
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# ===== SPLIT =====
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== MODEL =====
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

# ===== EVALUATE =====
from sklearn.metrics import accuracy_score
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# ===== SAVE =====
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved!")