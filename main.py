import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# SETUP
# =========================
os.makedirs("outputs/plots", exist_ok=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# =========================
# PREPROCESSING (REUSED)
# =========================
def clean_text(text):
    text = str(text)

    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()

    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return ' '.join(tokens)

# =========================
# LOAD DATA
# =========================
def load_training_data(path):
    texts, labels = [], []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                label = int(parts[0])
                title = parts[1]
                body = parts[3]

                texts.append(title + " " + body)
                labels.append(label)

    return texts, labels


def load_test_data(path):
    texts = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                title = parts[0]
                body = parts[2]

                texts.append(title + " " + body)

    return texts

# =========================
# LOAD + CLEAN
# =========================
X_raw, y = load_training_data("data/trainset.txt")
X_test_raw = load_test_data("data/testsetwithoutlabels.txt")

X_clean = [clean_text(x) for x in X_raw]
X_test_clean = [clean_text(x) for x in X_test_raw]

# =========================
# SPLIT
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X_clean, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# TF-IDF
# =========================
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test_clean)

# =========================
# GRID SEARCH (TUNING)
# =========================
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

grid = GridSearchCV(
    LogisticRegression(max_iter=2000),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_vec, y_train)

best_model = grid.best_estimator_

print("Best Params:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

# =========================
# SAVE CV TABLE
# =========================
cv_results = pd.DataFrame(grid.cv_results_)
cv_table = cv_results[['param_C','param_solver','mean_test_score','rank_test_score']]
cv_table.to_csv("outputs/cv_results.csv", index=False)

# =========================
# EVALUATION
# =========================
y_pred = best_model.predict(X_val_vec)

print("\nAccuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_val, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.savefig("outputs/plots/confusion_matrix.png")
plt.show()

# =========================
# ROC CURVE
# =========================
y_probs = best_model.predict_proba(X_val_vec)[:, 1]

fpr, tpr, _ = roc_curve(y_val, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Curve")
plt.savefig("outputs/plots/roc_curve.png")
plt.show()

# =========================
# PRECISION-RECALL CURVE
# =========================
precision, recall, _ = precision_recall_curve(y_val, y_probs)
ap = average_precision_score(y_val, y_probs)

plt.figure()
plt.plot(recall, precision, label=f"AP={ap:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("outputs/plots/pr_curve.png")
plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================
feature_names = vectorizer.get_feature_names_out()
coef = best_model.coef_[0]

top_pos = np.argsort(coef)[-15:]
top_neg = np.argsort(coef)[:15]

print("\nTop Positive Words:")
for i in reversed(top_pos):
    print(feature_names[i], coef[i])

print("\nTop Negative Words:")
for i in top_neg:
    print(feature_names[i], coef[i])

# =========================
# PREDICT TEST SET
# =========================
test_predictions = best_model.predict(X_test_vec)

with open("outputs/predictions.txt", "w") as f:
    for p in test_predictions:
        f.write(str(p) + "\n")

print("\nPredictions saved to outputs/predictions.txt")