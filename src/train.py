# src/train.py

import mlflow
import mlflow.sklearn
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import os

def train_with_mlflow(X_train, y_train, X_test, y_test, preprocessor=None):
    """
    Train model with MLflow tracking
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        preprocessor: Optional preprocessor (vectorizer)
    
    Returns:
        model: Trained model
        metrics: Dictionary of evaluation metrics
    """
    # Configure MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Create or get experiment
    experiment_name = "comcast_churn_prediction"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    model = LogisticRegression(max_iter=200)
    
    with mlflow.start_run():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log preprocessor if provided
        if preprocessor is not None:
            import os
            tfidf_path = os.path.join(os.path.dirname(__file__), "..", "Data", "processed", "tfidf.pkl")
            if os.path.exists(tfidf_path):
                mlflow.log_artifact(tfidf_path, "preprocessor")
        
        print("🏆 Model Training Complete!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\n" + classification_report(y_test, y_pred, zero_division=0))
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        return model, metrics


def main():
    # Configure MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Create or get experiment
    experiment_name = "comcast_churn_prediction"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    df = pd.read_parquet("Data/processed/reviews_raw.parquet")

    # Convert rating -> binary sentiment
    df["label"] = (df["rating"] >= 4).astype(int)

    # Load features
    with open("Data/processed/features.pkl", "rb") as f:
        X = pickle.load(f)

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)

    with mlflow.start_run():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "review_model")

        print("🏆 Model Accuracy:", acc)
        print(classification_report(y_test, preds))

        # Save model to disk
        with open("Data/processed/model.pkl", "wb") as f:
            pickle.dump(model, f)

    print("✅ Model training complete & logged to MLflow")


if __name__ == "__main__":
    main()
