import pandas as pd
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i.isalnum()]
    tokens = [i for i in tokens if i not in stopwords.words('english')]
    tokens = [ps.stem(i) for i in tokens]
    return " ".join(tokens)

df = pd.read_csv("spam.csv", encoding="latin-1")

if df.shape[1] > 2:
    df = df.iloc[:, [0, 1]]

df.columns = ["label", "text"]

df["label"] = df["label"].map({"ham": 0, "spam": 1})
df = df.dropna()

df["transformed_text"] = df["text"].apply(transform_text)

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["transformed_text"]).toarray()
y = df["label"].values

model = MultinomialNB()
model.fit(X, y)

pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model & Vectorizer saved successfully")