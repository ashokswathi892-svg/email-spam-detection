import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# -----------------------------
# 1️⃣ Load Kaggle dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\ashok\OneDrive\Desktop\email spam\spam.csv", encoding="latin-1")

# Keep only required columns
# Check first row in your CSV and change ['v1','v2'] if needed
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Map labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# -----------------------------
# 2️⃣ Train Model
# -----------------------------
X = df['message']
y = df['label']

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

print("✅ Model trained successfully with Kaggle dataset")

# -----------------------------
# 3️⃣ Save Model & Vectorizer
# -----------------------------
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("✅ Model & Vectorizer saved for deployment")

# -----------------------------
# 4️⃣ Terminal Deployment Loop
# -----------------------------
while True:
    email = input("\nEnter email text (type 'exit' to stop): ")
    if email.lower() == "exit":
        print("Program stopped ❌")
        break

    # Transform input & predict
    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)

    if prediction[0] == 1:
        print("🚨 Spam Email")
    else:
        print("✅ Not Spam Email")
