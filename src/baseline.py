import mlflow
import mlflow.sklearn
import mlflow.models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay

from dataset import get_uci_har_dataset


LABEL_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,       # unlimited — log explicitly so it appears in MLflow UI
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1,
}
PCA_VARIANCE = 0.99
PCA_RANDOM_STATE = 42


def plot_confusion_matrix(y_pred, y_true, path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Random Forest — Confusion Matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_rf_baseline():    
    X_train, y_train, X_test, y_test = get_uci_har_dataset()
    
    pca = PCA(n_components=PCA_VARIANCE, random_state=PCA_RANDOM_STATE)
    pca.fit(X_train)
    
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    n_components_selected = pca.n_components_
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("USI_HAR_RF_Baseline")

    with mlflow.start_run(run_name="rf_pca_baseline"):
        mlflow.set_tags({
            "model_type": "baseline",
            "approach": "supervised",
            "feature_type": "hand_crafted_pca",
        })

        mlflow.log_params({
            "train_samples": X_train.shape[0],
            "test_samples":  X_test.shape[0],
            "n_raw_features": X_train.shape[1],
        })
 
        mlflow.log_params({
            "pca_variance_threshold": PCA_VARIANCE,
            "pca_random_state": PCA_RANDOM_STATE,
            "pca_n_components_selected": n_components_selected,
        })
 
        mlflow.log_params(RF_PARAMS)

        
        rf = RandomForestClassifier(**RF_PARAMS)        
        rf.fit(X_train_pca, y_train)
        
        train_preds = rf.predict(X_train_pca)
        test_preds  = rf.predict(X_test_pca)
 
        train_acc = accuracy_score(y_train, train_preds)
        test_acc  = accuracy_score(y_test,  test_preds)
        test_f1_weighted = f1_score(y_test, test_preds, average="weighted")
        test_f1_macro    = f1_score(y_test, test_preds, average="macro")

        
        per_class_f1 = f1_score(y_test, test_preds, average=None)
        per_class_metrics = {
            f"f1_{name}": float(score)
            for name, score in zip(LABEL_NAMES, per_class_f1)
        }
 
        mlflow.log_metrics({
            "train_accuracy":    float(train_acc),
            "test_accuracy":     float(test_acc),
            "test_f1_weighted":  float(test_f1_weighted),
            "test_f1_macro":     float(test_f1_macro),
            **per_class_metrics,
        })


        report = classification_report(y_test, test_preds, target_names=LABEL_NAMES)
        with open("/tmp/classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("/tmp/classification_report.txt")
 
        cm_path = "/tmp/confusion_matrix.png"
        plot_confusion_matrix(y_test, test_preds, cm_path)
        mlflow.log_artifact(cm_path)
 
        signature = mlflow.models.infer_signature(X_train_pca, train_preds)
        mlflow.sklearn.log_model(
            rf,
            "random_forest_baseline",
            signature=signature,
            input_example=X_train_pca[:5],
        )
 


if __name__ == "__main__":
    run_rf_baseline()