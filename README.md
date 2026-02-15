# 📰 Sports vs. Politics Text Classifier

This repository contains the code, evaluation metrics, and full academic report for a Natural Language Processing (NLP) text classifier. The system is designed to read raw text from news articles and accurately categorize them as either **Sport** or **Politics**.

## 📊 Dataset Description
The data is sourced from the widely used **BBC News Dataset**, dynamically loaded via the Hugging Face `datasets` library. The original 5-category dataset was filtered to isolate the target binary classes.
* **Classes:** Sport, Politics
* **Total Documents:** 928 
* **Train Split:** 517 documents (275 Sport, 242 Politics)
* **Test Split:** 411 documents (236 Sport, 175 Politics)

## ⚙️ Feature Engineering
To convert the raw English text into numerical matrices for machine learning, the text was preprocessed (lowercased, English stop words removed) and vectorized using `scikit-learn`.

Three feature representations were evaluated:
1. **Bag of Words (BoW):** Yielded a vocabulary of 12,866 features.
2. **TF-IDF:** Utilized the 12,866 baseline vocabulary but penalized overly common words. This was selected as the primary feature representation for the final models.
3. **N-Grams (1, 2):** Expanding the TF-IDF vectorizer to include bigrams expanded the feature space to 95,310 unique semantic tokens.

## 🤖 Machine Learning Models & Results
Three distinct machine learning models were trained on the TF-IDF feature matrix and evaluated on the 411-document test set.

| Algorithm | Training Time | Precision | Recall | F1-Score | Overall Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Multinomial Naive Bayes** | `0.0062 s` | 1.00 | 1.00 | 1.00 | **100%** |
| **Logistic Regression** | `0.1933 s` | 1.00 | 1.00 | 1.00 | **100%** |
| **Support Vector Machine (Linear)** | `0.4728 s` | 1.00 | 1.00 | 1.00 | **100%** |

**Conclusion:** Due to the stark differences in vocabulary between political and sports news in the BBC dataset, all three models achieved perfect accuracy. However, **Multinomial Naive Bayes** is the optimal model for this task, achieving 100% accuracy while training approximately 76 times faster than SVM and 31 times faster than Logistic Regression.

## ⚠️ System Limitations
* **Semantic Blindness:** TF-IDF counts word frequencies; it does not understand sentence structure, sarcasm, or complex context.
* **Contextual Overlap:** The model will struggle with edge-case articles where vocabularies heavily intersect (e.g., politicians debating sports funding).
* **Geographic Bias:** Trained entirely on British news, the model heavily weights terms like "Parliament" and "Cricket". It will likely misclassify American news relying on terms like "Congress" and "Baseball".

## 🚀 How to Run the Code
1. Clone this repository to your local machine.
2. Install the required dependencies using the provided `requirements.txt` file:
   ```bash
   pip install -r requirements.txt