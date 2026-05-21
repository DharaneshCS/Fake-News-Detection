import pickle

# Load model and vectorizer
model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

def predict_news(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "REAL NEWS ✅" if prediction[0] == 1 else "FAKE NEWS ❌"

if __name__ == "__main__":
    print("Fake News Detection System")
    news_input = input("Enter news text: ")
    result = predict_news(news_input)
    print("Prediction:", result)