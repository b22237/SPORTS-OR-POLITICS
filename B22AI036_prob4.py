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
    raw_dataset = load_dataset("SetFit/bbc-news")
    train_df = pd.DataFrame(raw_dataset["train"])
    test_df = pd.DataFrame(raw_dataset["test"])
    target_labels = ['sport', 'politics']
    train_filtered = train_df[train_df['label_text'].isin(target_labels)].copy()
    test_filtered = test_df[test_df['label_text'].isin(target_labels)].copy()
    print(train_filtered['label_text'].value_counts())
    print(test_filtered['label_text'].value_counts())
    label_mapping = {'politics': 0, 'sport': 1}
    train_filtered['target'] = train_filtered['label_text'].map(label_mapping)
    test_filtered['target'] = test_filtered['label_text'].map(label_mapping)
    return train_filtered['text'], train_filtered['target'], test_filtered['text'], test_filtered['target']
def build_features(train_text, test_text):
    count_vec = CountVectorizer(stop_words='english')
    train_counts = count_vec.fit_transform(train_text)
    test_counts = count_vec.transform(test_text)
    tfidf_vec = TfidfVectorizer(stop_words='english')
    train_tfidf = tfidf_vec.fit_transform(train_text)
    test_tfidf = tfidf_vec.transform(test_text)
    bigram_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    train_bigrams = bigram_vec.fit_transform(train_text)
    test_bigrams = bigram_vec.transform(test_text)
    print(f"Unigram features: {train_counts.shape[1]}")
    print(f"Bigram features: {train_bigrams.shape[1]}")
    return train_tfidf, test_tfidf
def train_classifiers(features_train, labels_train):
    classifiers = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Linear SVM': SVC(kernel='linear', random_state=42)
    }
    trained_models = {}
    for name, model in classifiers.items():
        start = time.time()
        model.fit(features_train, labels_train)
        duration = time.time() - start
        print(f"{name} trained in {duration:.4f}s")
        trained_models[name] = model
    return trained_models
def evaluate_classifiers(models, features_test, labels_test):
    class_names = ['Politics (0)', 'Sport (1)']
    for name, model in models.items():
        predictions = model.predict(features_test)
        print(f"\n{name} Metrics:")
        print(classification_report(labels_test, predictions, target_names=class_names))
def render_confusion_matrix(model, features_test, labels_test, model_name):
    predictions = model.predict(features_test)
    matrix = confusion_matrix(labels_test, predictions)
    visual = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['Politics', 'Sport'])
    visual.plot(cmap=plt.cm.Blues)
    plt.title(f'{model_name} Matrix')
    plt.show()
def execute_pipeline():
    x_train, y_train, x_test, y_test = fetch_and_filter_data()
    print(f"Train size: {len(x_train)} | Test size: {len(x_test)}")
    train_vecs, test_vecs = build_features(x_train, x_test)
    active_models = train_classifiers(train_vecs, y_train)
    evaluate_classifiers(active_models, test_vecs, y_test)
    render_confusion_matrix(active_models['Naive Bayes'], test_vecs, y_test, 'Naive Bayes')
if __name__ == "__main__":
    execute_pipeline()