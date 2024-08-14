import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function for text cleaning
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.split()  # Split into words
    text = [word for word in text if word not in stopwords.words('english')]  # Remove stopwords
    text = [lemmatizer.lemmatize(word) for word in text]  # Lemmatize words
    text = ' '.join(text)  # Rejoin into a single string
    return text

# Load the trained model and vectorizer
model_path = r'C:\Users\Vaidehi Suryawanshi\Downloads\SPAM SMS DETECTION\best_spam_model.pkl'
vectorizer_path = r'C:\Users\Vaidehi Suryawanshi\Downloads\SPAM SMS DETECTION\vectorizer.pkl'

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Set the background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://static.vecteezy.com/system/resources/previews/002/188/833/original/chat-wallpaper-social-media-message-background-copy-space-for-a-text-vector.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title('SMS Spam Detection')
st.write('Enter the message below to check if it is spam or ham.')

# Input text box
user_input = st.text_area('Enter SMS message:', '')

if st.button('Predict'):
    # Preprocess the input text
    processed_input = preprocess_text(user_input)
    
    # Vectorize the input text
    input_tfidf = vectorizer.transform([processed_input])
    
    # Predict using the model
    prediction = model.predict(input_tfidf)
    
    # Display the result
    if prediction == 1:
        st.write('The message is Spam.')
    else:
        st.write('The message is Ham.')
