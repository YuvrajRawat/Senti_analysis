from flask import Flask, render_template, request
import joblib

model = joblib.load(r'C:\Users\yuvra\Downloads\sentiment_model.pkl')
vectorizer = joblib.load(r'C:\Users\yuvra\Downloads\tfidf_vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['comment']
    
    input_vector = vectorizer.transform([user_input])
    
    prediction = model.predict(input_vector)
    
    sentiment = prediction[0]

    return render_template('index.html', prediction_text=f'This comment is {sentiment}.')

if __name__ == '__main__':
    app.run(debug=True)
