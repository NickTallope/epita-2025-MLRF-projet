from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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
    train_hog_pca = np.load('../data/processed/train_hog_pca.npy')
    test_hog_pca = np.load('../data/processed/test_hog_pca.npy')
    train_hog_features = np.load('../data/processed/train_hog_features.npy')
    test_hog_features = np.load('../data/processed/test_hog_features.npy')
    train_flatten_features = np.load('../data/processed/train_flattened.npy')
    test_flatten_features = np.load('../data/processed/test_flattened.npy')
    train_sift_pca = np.load('../data/processed/train_sift_pca.npy')
    test_sift_pca = np.load('../data/processed/test_sift_pca.npy')

    models = {
        # 'logistic_regression': LogisticRegression(max_iter=1000),
        # 'random_forest': RandomForestClassifier(n_estimators=100),
        # 'svm_rbf': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
        'naive_bayes': GaussianNB(),
    }

    # Scale the features
    scaler = StandardScaler()

    # Train and evaluate models on HOG features with PCA
    # print("Training and evaluating models on HOG features with PCA...")
    # train_hog_pca_scaled = scaler.fit_transform(train_hog_pca)
    # test_hog_pca_scaled = scaler.transform(test_hog_pca)


    # for model_name, model in models.items():
    #     print(f'Training {model_name} on HOG features with PCA...')
    #     trained_model = train_model(model, train_hog_pca_scaled, train_labels)
    #     print(f'Evaluating {model_name} on HOG features with PCA...')
    #     evaluate_model(trained_model, test_hog_pca_scaled, test_labels, model_name)

    # Train and evaluate models on original HOG features without PCA
    print("Training and evaluating models on original HOG features without PCA...")
    train_hog_features_scaled = scaler.fit_transform(train_hog_features)
    test_hog_features_scaled = scaler.transform(test_hog_features)

    for model_name, model in models.items():
        print(f'Training {model_name} on original HOG features without PCA...')
        trained_model = train_model(model, train_hog_features_scaled, train_labels)
        print(f'Evaluating {model_name} on original HOG features without PCA...')
        evaluate_model(trained_model, test_hog_features_scaled, test_labels, model_name)


    # print("Training and evaluating models on flattened data")
    # train_flattened_scaled = scaler.fit_transform(train_flatten_features)
    # test_flattened_scaled = scaler.transform(test_flatten_features)

    # for model_name, model in models.items():
    #     print(f'Training {model_name} on original flattened')
    #     trained_model = train_model(model, train_flattened_scaled, train_labels)
    #     print(f'Evaluating {model_name} on original HOG features without PCA...')
    #     evaluate_model(trained_model, test_flattened_scaled, test_labels, model_name)

    # # Train and evaluate models on SIFT features
    # print("Training and evaluating models on SIFT features...")
    # train_sift_features_scaled = scaler.fit_transform(train_sift_pca)
    # test_sift_features_scaled = scaler.transform(test_sift_pca)

    # for model_name, model in models.items():
    #     print(f'Training {model_name} on SIFT features...')
    #     trained_model = train_model(model, train_sift_features_scaled, train_labels)
    #     print(f'Evaluating {model_name} on SIFT features...')
    #     evaluate_model(trained_model, test_sift_features_scaled, test_labels, model_name)

if __name__ == "__main__":
    main()
