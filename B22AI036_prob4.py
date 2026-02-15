import time
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def fetch_and_filter_data():
    # Pull the standard BBC news benchmark dataset directly from Hugging Face
    raw_dataset = load_dataset("SetFit/bbc-news")
    
    # Convert the dataset objects into pandas DataFrames for easier manipulation
    train_df = pd.DataFrame(raw_dataset["train"])
    test_df = pd.DataFrame(raw_dataset["test"])
    
    # We only care about the binary classification task between these two topics
    target_labels = ['sport', 'politics']
    
    # Filter out the Business, Tech, and Entertainment articles
    train_filtered = train_df[train_df['label_text'].isin(target_labels)].copy()
    test_filtered = test_df[test_df['label_text'].isin(target_labels)].copy()
    
    # Print the class balance to ensure the data is not heavily skewed
    print(train_filtered['label_text'].value_counts())
    print(test_filtered['label_text'].value_counts())
    
    # Machine learning models require numerical targets. Map the strings to binary digits.
    label_mapping = {'politics': 0, 'sport': 1}
    train_filtered['target'] = train_filtered['label_text'].map(label_mapping)
    test_filtered['target'] = test_filtered['label_text'].map(label_mapping)
    
    # Return the raw text columns and the mapped numerical targets
    return train_filtered['text'], train_filtered['target'], test_filtered['text'], test_filtered['target']

def build_features(train_text, test_text):
    # Base approach: Bag of Words. Strip out structural noise like "the" and "is"
    count_vec = CountVectorizer(stop_words='english')
    train_counts = count_vec.fit_transform(train_text)
    test_counts = count_vec.transform(test_text)
    
    # Main approach: TF-IDF. This penalizes words that appear in almost every article
    tfidf_vec = TfidfVectorizer(stop_words='english')
    train_tfidf = tfidf_vec.fit_transform(train_text)
    test_tfidf = tfidf_vec.transform(test_text)
    
    # Exploratory approach: N-grams. Extract single words and adjacent pairs (bigrams)
    bigram_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    train_bigrams = bigram_vec.fit_transform(train_text)
    test_bigrams = bigram_vec.transform(test_text)
    
    # Output the vocabulary sizes to see the dimensionality explosion when adding bigrams
    print(f"Unigram features: {train_counts.shape[1]}")
    print(f"Bigram features: {train_bigrams.shape[1]}")
    
    # Pass the standard TF-IDF matrices forward to train the models
    return train_tfidf, test_tfidf

def train_classifiers(features_train, labels_train):
    # Initialize three distinct mathematical algorithms to compare their performance
    classifiers = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Linear SVM': SVC(kernel='linear', random_state=42)
    }
    
    trained_models = {}
    
    # Iterate through the algorithms, fit them to the data, and track computation speed
    for name, model in classifiers.items():
        start = time.time()
        model.fit(features_train, labels_train)
        duration = time.time() - start
        
        print(f"{name} trained in {duration:.4f}s")
        trained_models[name] = model
        
    return trained_models

def evaluate_classifiers(models, features_test, labels_test):
    # Map the binary targets back to readable labels for the final output
    class_names = ['Politics (0)', 'Sport (1)']
    
    # Generate predictions and print precision, recall, and f1-scores for each model
    for name, model in models.items():
        predictions = model.predict(features_test)
        print(f"\n{name} Metrics:")
        print(classification_report(labels_test, predictions, target_names=class_names))

def render_confusion_matrix(model, features_test, labels_test, model_name):
    # Compute the matrix to see exact counts of false positives and true negatives
    predictions = model.predict(features_test)
    matrix = confusion_matrix(labels_test, predictions)
    
    # Generate the heatmap visual using matplotlib
    visual = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['Politics', 'Sport'])
    visual.plot(cmap=plt.cm.Blues)
    plt.title(f'{model_name} Matrix')
    plt.show()

def execute_pipeline():
    # 1. Ingest and structure the raw data
    x_train, y_train, x_test, y_test = fetch_and_filter_data()
    print(f"Train size: {len(x_train)} | Test size: {len(x_test)}")
    
    # 2. Convert text to mathematical feature matrices
    train_vecs, test_vecs = build_features(x_train, x_test)
    
    # 3. Train the machine learning models
    active_models = train_classifiers(train_vecs, y_train)
    
    # 4. Evaluate performance on unseen test data
    evaluate_classifiers(active_models, test_vecs, y_test)
    
    # 5. Output a visual confusion matrix for the fastest model
    render_confusion_matrix(active_models['Naive Bayes'], test_vecs, y_test, 'Naive Bayes')

if __name__ == "__main__":
    execute_pipeline()