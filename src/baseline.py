import mlflow
import mlflow.sklearn
import mlflow.models
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay

from dataset import get_uci_har_dataset
from config_loader import ConfigLoader


LABEL_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


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
    # Load config
    config = ConfigLoader().load_experiment("uci_har", "baseline_rf")
    seed = config["training"]["seed"]
    set_seed(seed)

    print("=" * 60)
    print("RANDOM FOREST BASELINE")
    print("=" * 60)
    print(f"PCA Variance Threshold: {config['model']['pca']['variance_threshold']}")
    print(f"RF Estimators: {config['model']['random_forest']['n_estimators']}")
    print("=" * 60 + "\n")

    X_train, y_train, X_test, y_test = get_uci_har_dataset()

    # Ensure proper dtypes for sklearn compatibility
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    pca_config = config["model"]["pca"]
    pca = PCA(n_components=pca_config["variance_threshold"], random_state=pca_config["random_state"])
    pca.fit(X_train)
    
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    n_components_selected = pca.n_components_

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        mlflow.set_tags(config["mlflow"]["tags"])

        mlflow.log_params({
            "train_samples": X_train.shape[0],
            "test_samples":  X_test.shape[0],
            "n_raw_features": X_train.shape[1],
        })
 
        mlflow.log_params({
            "pca_variance_threshold": pca_config["variance_threshold"],
            "pca_random_state": pca_config["random_state"],
            "pca_n_components_selected": n_components_selected,
        })

        # Build RF params from config
        rf_params = {
            "n_estimators": config["model"]["random_forest"]["n_estimators"],
            "max_depth": config["model"]["random_forest"]["max_depth"],
            "min_samples_leaf": config["model"]["random_forest"]["min_samples_leaf"],
            "random_state": seed,
            "n_jobs": config["model"]["random_forest"]["n_jobs"],
        }
        mlflow.log_params(rf_params)

        print("Training Random Forest...")
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train_pca, y_train)
        print("✓ Training complete\n")

        train_preds = rf.predict(X_train_pca)
        test_preds  = rf.predict(X_test_pca)

        train_acc = accuracy_score(y_train, train_preds)
        test_acc  = accuracy_score(y_test,  test_preds)
        test_f1_weighted = f1_score(y_test, test_preds, average="weighted")
        test_f1_macro    = f1_score(y_test, test_preds, average="macro")

        # ✓ Print results
        print("=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Train Accuracy:        {train_acc:.4f}")
        print(f"Test Accuracy:         {test_acc:.4f}")
        print(f"Test F1 (weighted):    {test_f1_weighted:.4f}")
        print(f"Test F1 (macro):       {test_f1_macro:.4f}")
        print("=" * 60 + "\n")


        per_class_f1 = np.asarray(f1_score(y_test, test_preds, average=None), dtype=float)
        per_class_metrics = {
            f"f1_{name}": float(score)
            for name, score in zip(LABEL_NAMES, per_class_f1)
        }

        # ✓ Print per-class results
        print("Per-Class F1 Scores:")
        print("-" * 60)
        for name, score in zip(LABEL_NAMES, per_class_f1):
            print(f"  {name:20s}: {float(score):.4f}")
        print("-" * 60 + "\n")
 
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
            artifact_path="random_forest_model",
            signature=signature,
            input_example=X_train_pca[:5],
            serialization_format="skops"
        )


if __name__ == "__main__":
    run_rf_baseline()