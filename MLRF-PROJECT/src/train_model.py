from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
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
    print(f"Evaluation report for {model_name}:")
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
    train_hog_pca = np.load('../data/processed/train_hog_pca.npy')
    test_hog_pca = np.load('../data/processed/test_hog_pca.npy')
    train_sift_pca = np.load('../data/processed/train_sift_pca.npy')
    test_sift_pca = np.load('../data/processed/test_sift_pca.npy')

    # Concatenate HOG and SIFT features
    train_features = np.hstack((train_hog_pca, train_sift_pca))
    test_features = np.hstack((test_hog_pca, test_sift_pca))

    # Scale the features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    models = {
        'incremental_logistic_regression': SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3),
        'logistic_regression': LogisticRegression(max_iter=1000),
        'svm': SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3),
        'random_forest': RandomForestClassifier(n_estimators=100)
    }

    for model_name, model in models.items():
        print(f'Training {model_name}...')
        if model_name in ['incremental_logistic_regression', 'svm']:
            trained_model = train_model_incremental(model, train_features, train_labels)
        else:
            trained_model = train_model(model, train_features, train_labels)
        
        print(f'Evaluating {model_name}...')
        evaluate_model(trained_model, test_features, test_labels, model_name)

if __name__ == "__main__":
    main()
