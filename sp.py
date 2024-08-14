import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import joblib

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function for text cleaning
def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.split()  
    text = [word for word in text if word not in stopwords.words('english')] 
    text = [lemmatizer.lemmatize(word) for word in text]  
    text = ' '.join(text)  
    return text

# Load the CSV file
data_path = r"C:/Users/Vaidehi Suryawanshi/Downloads/SPAM SMS DETECTION/spam/spam.csv"
data = pd.read_csv(data_path, encoding='latin-1')

# Drop unnecessary columns and rename for clarity
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Encode labels: 'ham' as 0 and 'spam' as 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Apply preprocessing to the messages
data['message'] = data['message'].apply(preprocess_text)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(data['message'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['label'], test_size=0.2, random_state=42)

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define a function to train and evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else None
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Classification report
    class_report = classification_report(y_test, y_pred)
    
    print(f'Model: {model.__class__.__name__}')
    print(f'Accuracy: {accuracy}')
    print(f'ROC AUC Score: {roc_auc}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{class_report}')
    print('-'*60)
    
    return accuracy, roc_auc

# Initialize models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Dictionary to store results
results = {}

# Evaluate each model
for model_name, model in models.items():
    accuracy, roc_auc = evaluate_model(model, X_train_resampled, y_train_resampled, X_test, y_test)
    results[model_name] = (accuracy, roc_auc)

# Find the best model based on accuracy
best_model_name = max(results, key=lambda k: results[k][0])
best_model = models[best_model_name]

print(f'Best Model: {best_model_name} with accuracy {results[best_model_name][0]}')

# Save the best model and vectorizer
joblib.dump(best_model, 'best_spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

