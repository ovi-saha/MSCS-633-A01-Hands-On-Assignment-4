import os
import kagglehub
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from pyod.models.auto_encoder import AutoEncoder
import matplotlib.pyplot as plt

# === Download Dataset from KaggleHub ===
# Automatically downloads and unzips the dataset
path = kagglehub.dataset_download("whenamancodes/fraud-detection")
print("Path to dataset files:", path)

# === Visualization Function: Anomaly Score Distribution ===
def plot_anomaly_scores(y_true, scores):
    """
    Plot the distribution of anomaly scores for normal and fraudulent transactions.

    Parameters:
    - y_true: array-like of true labels (0 = normal, 1 = fraud)
    - scores: array-like of anomaly scores from the AutoEncoder
    """
    plt.figure(figsize=(10,6))

    # Separate anomaly scores by actual class
    normal_scores = scores[y_true == 0]
    fraud_scores = scores[y_true == 1]

    # Histogram of scores: blue for normal, red for fraud
    plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
    plt.hist(fraud_scores, bins=50, alpha=0.6, label='Fraud', color='red', density=True)

    # Formatting the plot
    plt.title('Distribution of Anomaly Scores')
    plt.xlabel('Anomaly Score (Reconstruction Error)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Main Logic ===
def main():
    # Load the dataset from the downloaded KaggleHub folder
    data_path = os.path.join(path, "creditcard.csv")  # Path to CSV inside the dataset folder
    print("Loading dataset...")
    df = pd.read_csv(data_path)  # Load data into DataFrame

    # Separate input features (V1 to V28, Amount, Time) and labels (Class)
    X = df.drop(columns=['Class']).values
    y = df['Class'].values  # 0 = normal, 1 = fraud

    # === Feature Scaling ===
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Normalize features for better model performance

    # === Training Data Selection ===
    # Use only normal transactions (Class = 0) to train the AutoEncoder
    X_train = X_scaled[y == 0]
    print(f"Training data shape (only normal transactions): {X_train.shape}")

    # === Initialize AutoEncoder Model ===
    clf = AutoEncoder(
        epoch_num=30,        # Number of training passes over the data
        batch_size=32,       # Mini-batch size for training
        contamination=0.01,  # Estimated fraud ratio (used for threshold)
        verbose=1,           # Show training progress
        random_state=42      # Ensures consistent results across runs
    )

    # === Model Training ===
    print("Training AutoEncoder...")
    clf.fit(X_train)  # Learns how normal data looks (low reconstruction error)

    # === Prediction on All Data (Normal + Fraud) ===
    print("Predicting anomalies...")
    y_pred = clf.predict(X_scaled)               # 0 = normal, 1 = predicted anomaly
    y_scores = clf.decision_function(X_scaled)  # Continuous anomaly score (higher = more anomalous)

    # === Evaluation ===
    print("\nClassification Report:")
    print(classification_report(y, y_pred))     # Precision, recall, f1-score summary

    roc_auc = roc_auc_score(y, y_scores)        # Area under ROC curve (better for imbalanced data)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # === Visualization ===
    # Plot histogram of anomaly scores for both normal and fraud
    plot_anomaly_scores(y, y_scores)

    # === Save Results ===
    results_df = df.copy()
    results_df['Anomaly_Prediction'] = y_pred  # Add binary predictions to DataFrame
    results_df['Anomaly_Score'] = y_scores     # Add anomaly scores
    results_df.to_csv('fraud_detection_results.csv', index=False)  # Save to CSV
    print("Results saved to 'fraud_detection_results.csv'")

# === Run script ===
if __name__ == "__main__":
    main()
