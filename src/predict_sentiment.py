from keras.models import load_model
from joblib import load

# Load the saved TF-IDF vectorizer and model
tfidf_vectorizer = load('../models/tfidf_vectorizer.joblib')
model = load_model('../models/sentiment_analysis_model.keras')

# Function to predict sentiment
def predict_sentiment(review, model=model, vectorizer=tfidf_vectorizer, expected_input_shape=4999):
    # Transform the input review using the saved TF-IDF vectorizer
    review_tfidf = vectorizer.transform([review])
    
    # Convert sparse matrix to dense
    review_tfidf_dense = review_tfidf.toarray()
    
    # Ensure the input has the same number of features as the model expects
    if review_tfidf_dense.shape[1] != expected_input_shape:
        review_tfidf_dense = review_tfidf_dense[:, :expected_input_shape]  # Trim excess features
    
    # Predict sentiment (output is a probability for the positive class)
    prediction_prob = model.predict(review_tfidf_dense)
    
    # Interpret the result
    if prediction_prob >= 0.75:  # High probability for positive sentiment
        return "Positive", prediction_prob[0][0]
    elif prediction_prob <= 0.25:  # High probability for negative sentiment
        return "Negative", prediction_prob[0][0]
    else:  # Between 0.25 and 0.75 - consider it as Neutral sentiment
        return "Neutral", prediction_prob[0][0]

if __name__ == '__main__':
    review = "The movie was quite dull and boring."
    sentiment, prob = predict_sentiment(review, model, tfidf_vectorizer, expected_input_shape=4999)
    print(f'Sentiment: {sentiment}, Probability: {prob}')