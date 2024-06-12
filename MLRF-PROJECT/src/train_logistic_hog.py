from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, train_features, train_labels):
    model.fit(train_features, train_labels)
    return model

def train_model_incremental(model, train_features, train_labels, batch_size=1000):
    num_samples = train_features.shape[0]
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        model.partial_fit(train_features[start:end], train_labels[start:end], classes=np.unique(train_labels))
    return model

def evaluate_model(model, test_features, test_labels, model_name):
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Evaluation report for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(test_labels, predictions))
    print(confusion_matrix(test_labels, predictions))

    if hasattr(model, "predict_proba"):
        try:
            fpr, tpr, _ = roc_curve(test_labels, model.predict_proba(test_features)[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()
        except ValueError as e:
            print(f"Could not compute ROC curve for {model_name}: {e}")

def main():
    # Load the labels
    train_labels = np.load('../data/processed/train_labels.npy')
    test_labels = np.load('../data/processed/test_labels.npy')

    # Load the features
    train_hog_features = np.load('../data/processed/train_hog_features.npy')
    test_hog_features = np.load('../data/processed/test_hog_features.npy')

    # Scale the features
    scaler = StandardScaler()
    train_hog_features_scaled = scaler.fit_transform(train_hog_features)
    test_hog_features_scaled = scaler.transform(test_hog_features)

    # Hyperparameter tuning for Logistic Regression using GridSearchCV
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['lbfgs', 'saga', 'liblinear'],
        'max_iter': [100, 500, 1000]
    }

    print("Training and evaluating Logistic Regression with GridSearchCV on original HOG features without PCA...")
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(train_hog_features_scaled, train_labels)

    print(f"Best parameters found: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    evaluate_model(best_model, test_hog_features_scaled, test_labels, "Logistic Regression with GridSearchCV")

if __name__ == "__main__":
    main()
