from flask import Flask, request, render_template
import pickle
from urllib.parse import urlparse
import datetime
import requests

app = Flask(__name__)

# =========================
# 🔹 LOAD MODEL
# =========================
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# 🔹 NewsData API Key
NEWSDATA_API_KEY = "YOUR_NEWSDATA_API_KEY"

# 🇮🇳 ONLY INDIAN SOURCES
TRUSTED_SOURCES = [
    "thehindu.com",
    "timesofindia.indiatimes.com",
    "hindustantimes.com",
    "ndtv.com",
    "indianexpress.com",
    "pib.gov.in",
    "news18.com",
    "indiatoday.in"
]

# =========================
# 🔹 YEAR DETECTION
# =========================
def detect_year(text):
    import re
    years = re.findall(r'\b(20\d{2})\b', text)
    return years[0] if years else str(datetime.datetime.now().year)

# =========================
# 🔹 INDIAN SOURCE CHECK
# =========================
def is_indian_news(url):
    if not url:
        return False

    try:
        domain = urlparse(url).netloc
        return any(site in domain for site in TRUSTED_SOURCES)
    except:
        return False

# =========================
# 🔹 NEWSDATA CHECK 🔥
# =========================
def newsdata_check(text):
    try:
        url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={text}&country=in&language=en"
        res = requests.get(url).json()

        if "results" in res and len(res["results"]) > 0:
            return True  # Found real news
    except:
        pass

    return False

# =========================
# 🔹 ML CHECK
# =========================
def ml_check(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "REAL" if pred == 1 else "FAKE"

# =========================
# 🔹 FINAL DECISION
# =========================
def final_decision(text, url):

    year = detect_year(text)

    # ❌ Non-Indian → FAKE
    if not is_indian_news(url):
        return {
            "Final": f"🚨 FAKE NEWS (Non-Indian Source) ({year})",
            "Year": year
        }

    # 🔥 Step 1: NewsData Check
    if newsdata_check(text):
        return {
            "Final": f"✅ REAL NEWS (Verified Indian News API) ({year})",
            "Year": year,
            "Source": "NewsData.io"
        }

    # 🔥 Step 2: ML fallback
    ml = ml_check(text)

    if ml == "REAL":
        final = f"✅ REAL NEWS (ML Prediction) ({year})"
    else:
        final = f"🚨 FAKE NEWS (ML Prediction) ({year})"

    return {
        "Final": final,
        "Year": year,
        "ML": ml,
        "Source": "ML Model"
    }

# =========================
# 🔹 ROUTES
# =========================
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('news')
    url = request.form.get('url', "")

    result = final_decision(text, url)

    return render_template("index.html", result=result)

# =========================
# 🔹 RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)