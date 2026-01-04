"""
Transformer Models for Emotion Detection
This script handles text preprocessing and Transformer-based emotion classification.
Models: BERT, ELECTRA, RoBERTa
"""

import os

# Force transformers to use PyTorch only (must be set before importing transformers)
os.environ["TRANSFORMERS_NO_TF"] = "1"

import itertools
import json
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    BertModel,
    BertTokenizer,
    ElectraModel,
    ElectraTokenizer,
    RobertaModel,
    RobertaTokenizer,
)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
TRAIN_FILE = "./data/train.csv"
VALIDATION_FILE = "./data/validation.csv"
TEST_FILE = "./data/test.csv"
RESULTS_FOLDER = "./results"
SAVE_MODELS_FOLDER = "./hp_models"

# Models parameters
BERT_MODEL_NAME = "bert-base-uncased"
ELECTRA_MODEL_NAME = "google/electra-small-discriminator"
ROBERTA_MODEL_NAME = "roberta-base"

MAX_LENGTH = 128
PARAM_GRID = {
    "dropout_rate": [0.1, 0.3],
    "lr": [2e-5, 3e-5],
    "batch_size": [16, 32],
    "max_length": [128, 256],
}


def load_data(data_file, dataset_name="Data"):
    """
    Load data from CSV file.

    Args:
        data_file: Path to the CSV file
        dataset_name: Name of the dataset

    Returns:
        tuple: (texts, labels)
    """
    print("=" * 70)
    print(f"Loading {dataset_name}")
    print("=" * 70)
    print(f"Reading file: {data_file}")
    print(f"File exists: {os.path.exists(data_file)}")

    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} samples from {data_file}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")

    text_column = "text" if "text" in df.columns else df.columns[0]
    label_column = "label" if "label" in df.columns else None
    print(f"Using text column: '{text_column}'")
    print(f"Using label column: '{label_column}'")

    texts = df[text_column].values
    print(f"Texts dtype: {texts.dtype}, shape: {texts.shape}")
    if label_column is None:
        print(f"Number of samples: {len(texts)}")
        print(f"No labels found, returning texts only")
        return texts, None

    labels = df[label_column].values
    print(f"Labels dtype: {labels.dtype}, shape: {labels.shape}")
    print(f"Number of samples: {len(texts)}")
    print(f"Number of unique emotions: {len(np.unique(labels))}")
    print(f"Emotion distribution: {np.bincount(labels)}")
    print(f"Label range: [{labels.min()}, {labels.max()}]")

    return texts, labels


def preprocess_text(text):
    """
    Clean and preprocess text data.

    Args:
        text: Input text string

    Returns:
        Cleaned and preprocessed text
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&\w+;", "", text)

    html_artifacts = ["href", "nofollow", "permalink", "pagetitle", "rel", "target"]
    for artifact in html_artifacts:
        text = text.replace(artifact, "")

    text = re.sub(r"@\w+|#\w+", "", text)

    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
    }
    for contraction, replacement in contractions.items():
        text = text.replace(contraction, replacement)

    text = re.sub(r"'s\b", "", text)
    text = text.replace("'", "")

    malformed_contractions = {
        "wont": "will not",
        "cant": "cannot",
        "dont": "do not",
        "doesnt": "does not",
        "didnt": "did not",
        "isnt": "is not",
        "arent": "are not",
        "wasnt": "was not",
        "werent": "were not",
        "hasnt": "has not",
        "havent": "have not",
        "hadnt": "had not",
        "shouldnt": "should not",
        "wouldnt": "would not",
        "couldnt": "could not",
        "mightnt": "might not",
        "mustnt": "must not",
        "shant": "shall not",
        "shouldve": "should have",
        "wouldve": "would have",
        "couldve": "could have",
        "mustve": "must have",
        "mightve": "might have",
        "youre": "you are",
        "theyre": "they are",
        "were": "we are",
        "youve": "you have",
        "theyve": "they have",
        "weve": "we have",
        "ive": "i have",
        "youll": "you will",
        "theyll": "they will",
        "well": "we will",
        "ill": "i will",
        "youd": "you would",
        "theyd": "they would",
        "hed": "he would",
        "shed": "she would",
        "wed": "we would",
        "itd": "it would",
        "im": "i am",
        "hes": "he is",
        "shes": "she is",
        "its": "it is",
        "thats": "that is",
        "whats": "what is",
        "wheres": "where is",
        "whos": "who is",
        "hows": "how is",
        "theres": "there is",
    }

    for contraction, replacement in malformed_contractions.items():
        text = re.sub(r"\b" + contraction + r"\b", replacement, text)

    text = re.sub(r"[^a-z\s.,!?]", "", text)
    text = re.sub(r"([.,!?])\1+", r"\1", text)
    text = re.sub(r"\s+[.,!?]+\s+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def plot_history(history, title="Training History"):
    """
    Plot training/validation metrics from training history.

    Args:
        history: Dictionary with metrics keys (loss, accuracy, precision, recall, f1, auc_pr, etc.)
        title: Title for the overall figure
    """
    # Handle both PyTorch dict and TensorFlow History object
    if hasattr(history, "history"):
        # TensorFlow History object
        acc = history.history.get("accuracy", history.history.get("acc"))
        val_acc = history.history.get("val_accuracy", history.history.get("val_acc"))
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        precision = history.history.get("precision", [])
        val_precision = history.history.get("val_precision", [])
        recall = history.history.get("recall", [])
        val_recall = history.history.get("val_recall", [])
        f1 = history.history.get("f1", [])
        val_f1 = history.history.get("val_f1", [])
        auc_pr = history.history.get("auc_pr", [])
        val_auc_pr = history.history.get("val_auc_pr", [])
    else:
        # PyTorch dictionary
        acc = history.get("accuracy", history.get("acc"))
        val_acc = history.get("val_accuracy", history.get("val_acc"))
        loss = history["loss"]
        val_loss = history["val_loss"]
        precision = history.get("precision", [])
        val_precision = history.get("val_precision", [])
        recall = history.get("recall", [])
        val_recall = history.get("val_recall", [])
        f1 = history.get("f1", [])
        val_f1 = history.get("val_f1", [])
        auc_pr = history.get("auc_pr", [])
        val_auc_pr = history.get("val_auc_pr", [])

    epochs = range(1, len(loss) + 1)

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16)

    # Loss
    axes[0, 0].plot(epochs, loss, "b-", label="Training loss")
    axes[0, 0].plot(epochs, val_loss, "r--", label="Validation loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, acc, "b-", label="Training accuracy")
    axes[0, 1].plot(epochs, val_acc, "r--", label="Validation accuracy")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision
    if precision and val_precision:
        axes[0, 2].plot(epochs, precision, "b-", label="Training precision")
        axes[0, 2].plot(epochs, val_precision, "r--", label="Validation precision")
        axes[0, 2].set_title("Precision (Macro)")
        axes[0, 2].set_xlabel("Epochs")
        axes[0, 2].set_ylabel("Precision")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # Recall
    if recall and val_recall:
        axes[1, 0].plot(epochs, recall, "b-", label="Training recall")
        axes[1, 0].plot(epochs, val_recall, "r--", label="Validation recall")
        axes[1, 0].set_title("Recall (Macro)")
        axes[1, 0].set_xlabel("Epochs")
        axes[1, 0].set_ylabel("Recall")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # F1 Score
    if f1 and val_f1:
        axes[1, 1].plot(epochs, f1, "b-", label="Training F1")
        axes[1, 1].plot(epochs, val_f1, "r--", label="Validation F1")
        axes[1, 1].set_title("F1 Score (Macro)")
        axes[1, 1].set_xlabel("Epochs")
        axes[1, 1].set_ylabel("F1 Score")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # AUC-PR
    if auc_pr and val_auc_pr:
        axes[1, 2].plot(epochs, auc_pr, "b-", label="Training AUC-PR")
        axes[1, 2].plot(epochs, val_auc_pr, "r--", label="Validation AUC-PR")
        axes[1, 2].set_title("AUC-PR (Macro)")
        axes[1, 2].set_xlabel("Epochs")
        axes[1, 2].set_ylabel("AUC-PR")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    fname = f"{title.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_FOLDER, fname))
    print(f"Saved training plot: {fname}")
    plt.close()


def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    label_prefix="",
):
    """
    Print and plot the confusion matrix.

    Args:
        cm: Confusion matrix
        classes: List of class names
        normalize: If True, normalize the confusion matrix
        title: Plot title
        cmap: Matplotlib colormap
        label_prefix: Prefix for saved filename
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fname = f"{title.replace(' ', '_').lower()}_{label_prefix}.png"
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_FOLDER, fname))
    print(f"Saved confusion matrix plot: {fname}")
    plt.show()


def prepare_model_data(texts, labels=None, model_type="bert", model_name=None, max_length=MAX_LENGTH):
    """
    Prepare data for transformer models using appropriate tokenizer.
    Reuses preprocess_text() for text cleaning, then applies model-specific tokenization.

    Args:
        texts: Array of text strings
        labels: Optional array of labels
        model_type: Type of model ("bert", "electra", or "roberta")
        model_name: Name of pretrained model (if None, uses default for model_type)
        max_length: Maximum sequence length

    Returns:
        dict: Dictionary with 'input_ids', 'attention_mask', and optionally 'labels'
    """
    print(f"\nStarting data preparation for {model_type.upper()}")
    print(f"Input texts count: {len(texts)}")
    print(f"Max length: {max_length}")
    print(f"Labels provided: {labels is not None}")
    
    # Set default model names
    if model_name is None:
        model_names = {"bert": BERT_MODEL_NAME, "electra": ELECTRA_MODEL_NAME, "roberta": ROBERTA_MODEL_NAME}
        model_name = model_names[model_type]

    print(f"\nPreparing {model_type.upper()} data using {model_name}...")
    print("Loading tokenizer... (downloading if first time)")

    # Select appropriate tokenizer
    if model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_name)
        print("BERT tokenizer loaded.")
    elif model_type == "electra":
        tokenizer = ElectraTokenizer.from_pretrained(model_name)
        print("ELECTRA tokenizer loaded.")
    elif model_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        print("RoBERTa tokenizer loaded.")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Preprocess texts
    print(f"Preprocessing {len(texts)} texts...")
    processed_texts = [preprocess_text(str(text)) for text in texts]
    print(f"Preprocessing complete. Sample length range: [{min(len(t) for t in processed_texts)}, {max(len(t) for t in processed_texts)}]")

    # Tokenize
    print(f"Tokenizing texts with max_length={max_length}...")
    encoded = tokenizer(processed_texts, add_special_tokens=True, max_length=max_length, padding="max_length", truncation=True, return_attention_mask=True, return_tensors="np")
    print(f"Tokenization complete")

    result = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "tokenizer": tokenizer}

    if labels is not None:
        result["labels"] = labels
        print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")

    print(f"{model_type.upper()} data prepared. Input shape: {encoded['input_ids'].shape}")
    print(f"Input IDs dtype: {encoded['input_ids'].dtype}, range: [{encoded['input_ids'].min()}, {encoded['input_ids'].max()}]")
    print(f"Attention mask shape: {encoded['attention_mask'].shape}")
    return result


class TransformerClassifier(nn.Module):
    """PyTorch transformer-based classification model."""

    def __init__(self, model_type, model_name, num_classes, dropout_rate=0.1):
        super(TransformerClassifier, self).__init__()
        print(f"Initializing TransformerClassifier: type={model_type}, model_name={model_name}, num_classes={num_classes}, dropout_rate={dropout_rate}")

        # Load pretrained transformer
        print(f"Loading pretrained {model_type.upper()} model from {model_name}...")
        if model_type == "bert":
            self.transformer = BertModel.from_pretrained(model_name, use_safetensors=True)
        elif model_type == "electra":
            self.transformer = ElectraModel.from_pretrained(model_name, use_safetensors=True)
        elif model_type == "roberta":
            self.transformer = RobertaModel.from_pretrained(model_name, use_safetensors=True)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        print(f"{model_type.upper()} base model loaded successfully")

        # Classification head
        hidden_size = self.transformer.config.hidden_size
        print(f"Transformer hidden size: {hidden_size}")
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate / 2)
        self.fc2 = nn.Linear(128, num_classes)
        print(f"Classification head: {hidden_size} -> 128 -> {num_classes}")

    def forward(self, input_ids, attention_mask):
        # Get transformer output
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Classification head
        x = self.dropout1(cls_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits


def build_model(model_type="bert", model_name=None, num_classes=6, max_length=MAX_LENGTH, dropout_rate=0.1, lr=2e-5):
    """
    Build and compile a transformer-based classification model.

    Args:
        model_type: Type of model ("bert", "electra", or "roberta")
        model_name: Name of pretrained model (if None, uses default for model_type)
        num_classes: Number of output classes
        max_length: Maximum sequence length (not used in PyTorch version but kept for compatibility)
        dropout_rate: Dropout rate for classification head
        lr: Learning rate for AdamW optimizer

    Returns:
        Tuple of (model, optimizer, device)
    """
    print(f"\nBuilding {model_type.upper()} model...")
    print(f"Parameters: num_classes={num_classes}, dropout_rate={dropout_rate}, lr={lr}")
    
    # Set default model names
    if model_name is None:
        model_names = {"bert": BERT_MODEL_NAME, "electra": ELECTRA_MODEL_NAME, "roberta": ROBERTA_MODEL_NAME}
        model_name = model_names[model_type]

    print(f"Building {model_type.upper()} model from {model_name}...")

    # Create model
    model = TransformerClassifier(model_type, model_name, num_classes, dropout_rate)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Moving model to device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    model.to(device)

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    print(f"Optimizer created: AdamW with lr={lr}")

    # Calculate model size
    model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # 4 bytes per parameter (float32)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{model_type.upper()} model built successfully!")
    print(f"Model device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")

    return model, optimizer, device


def train_model(model, optimizer, device, X_train, y_train, X_val, y_val, epochs=3, batch_size=16, patience=3):
    """
    Train a PyTorch model with early stopping.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        device: torch device
        X_train: Dictionary with 'input_ids' and 'attention_mask' for training
        y_train: Training labels
        X_val: Dictionary with 'input_ids' and 'attention_mask' for validation
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        patience: Early stopping patience

    Returns:
        dict: Training history
    """
    print(f"\nStarting training...")
    print(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Patience: {patience}")
    
    # Convert data to tensors
    print(f"Converting data to tensors...")
    train_input_ids = torch.tensor(X_train["input_ids"], dtype=torch.long)
    train_attention_mask = torch.tensor(X_train["attention_mask"], dtype=torch.long)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    print(f"Train tensors - input_ids: {train_input_ids.shape}, attention_mask: {train_attention_mask.shape}, labels: {train_labels.shape}")

    val_input_ids = torch.tensor(X_val["input_ids"], dtype=torch.long)
    val_attention_mask = torch.tensor(X_val["attention_mask"], dtype=torch.long)
    val_labels = torch.tensor(y_val, dtype=torch.long)
    print(f"Val tensors - input_ids: {val_input_ids.shape}, attention_mask: {val_attention_mask.shape}, labels: {val_labels.shape}")

    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Train dataloader: {len(train_loader)} batches")

    val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Val dataloader: {len(val_loader)} batches")

    # Calculate class weights to handle class imbalance
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    num_classes = len(class_counts)

    # Inverse frequency weighting: weight = total_samples / (num_classes * class_count)
    class_weights = torch.FloatTensor([total_samples / (num_classes * count) for count in class_counts])
    class_weights = class_weights.to(device)

    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training history
    history = {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc_pr": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auc_pr": [],
    }

    # Early stopping variables
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    # Training loop
    print(f"\nBeginning training loop...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_all_preds = []
        train_all_labels = []
        train_all_probs = []

        for batch_idx, (batch_input_ids, batch_attention_mask, batch_labels) in enumerate(train_loader):
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_input_ids, batch_attention_mask)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
            
            # Collect predictions and probabilities for metrics
            train_all_preds.extend(predicted.cpu().numpy())
            train_all_labels.extend(batch_labels.cpu().numpy())
            train_all_probs.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())
            
            if batch_idx == 0:
                print(f"  First batch - loss: {loss.item():.4f}, batch_size: {batch_labels.size(0)}")

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Calculate additional metrics for training
        train_all_probs = np.vstack(train_all_probs)
        train_precision = precision_score(train_all_labels, train_all_preds, average="macro", zero_division=0)
        train_recall = recall_score(train_all_labels, train_all_preds, average="macro", zero_division=0)
        train_f1 = f1_score(train_all_labels, train_all_preds, average="macro", zero_division=0)
        try:
            train_auc_pr = average_precision_score(train_all_labels, train_all_probs, average="macro")
        except Exception as e:
            print(f"  Warning: Could not calculate training AUC-PR: {e}")
            train_auc_pr = 0.0
        
        print(f"  Train - loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, precision: {train_precision:.4f}, recall: {train_recall:.4f}, f1: {train_f1:.4f}, auc_pr: {train_auc_pr:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_labels = []
        val_all_probs = []

        with torch.no_grad():
            for batch_input_ids, batch_attention_mask, batch_labels in val_loader:
                batch_input_ids = batch_input_ids.to(device)
                batch_attention_mask = batch_attention_mask.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_input_ids, batch_attention_mask)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
                
                # Collect predictions and probabilities for metrics
                val_all_preds.extend(predicted.cpu().numpy())
                val_all_labels.extend(batch_labels.cpu().numpy())
                val_all_probs.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Calculate additional metrics for validation
        val_all_probs = np.vstack(val_all_probs)
        val_precision = precision_score(val_all_labels, val_all_preds, average="macro", zero_division=0)
        val_recall = recall_score(val_all_labels, val_all_preds, average="macro", zero_division=0)
        val_f1 = f1_score(val_all_labels, val_all_preds, average="macro", zero_division=0)
        try:
            val_auc_pr = average_precision_score(val_all_labels, val_all_probs, average="macro")
        except Exception as e:
            print(f"  Warning: Could not calculate validation AUC-PR: {e}")
            val_auc_pr = 0.0
        
        print(f"  Val   - loss: {val_loss:.4f}, accuracy: {val_acc:.4f}, precision: {val_precision:.4f}, recall: {val_recall:.4f}, f1: {val_f1:.4f}, auc_pr: {val_auc_pr:.4f}")

        # Update history
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["precision"].append(train_precision)
        history["recall"].append(train_recall)
        history["f1"].append(train_f1)
        history["auc_pr"].append(train_auc_pr)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)
        history["val_auc_pr"].append(val_auc_pr)

        print(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - prec: {train_precision:.4f} - rec: {train_recall:.4f} - f1: {train_f1:.4f} - auc_pr: {train_auc_pr:.4f}")
        print(f"         Val    - loss: {val_loss:.4f} - acc: {val_acc:.4f} - prec: {val_precision:.4f} - rec: {val_recall:.4f} - f1: {val_f1:.4f} - auc_pr: {val_auc_pr:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model weights (best_val_loss: {best_val_loss:.4f})")
    else:
        print("No best model state to restore")

    print(f"Training completed. History contains {len(history['loss'])} epochs")
    return history


def hyperparameter_search(model_type, train_texts, y_train, val_texts, y_val, num_classes, model_name=None, param_grid=None, max_models=None, results_filename=None):
    """
    Run hyperparameter search for any transformer model.

    Args:
        model_type: Type of model ("bert", "electra", or "roberta")
        train_texts: Training text data (raw texts, not tokenized)
        y_train: Training labels
        val_texts: Validation text data (raw texts, not tokenized)
        y_val: Validation labels
        num_classes: Number of output classes
        model_name: Name of pretrained model (if None, uses default for model_type)
        param_grid: Dictionary of hyperparameters to search
        max_models: Maximum number of models to train
        results_filename: Filename to save results (if None, auto-generated)

    Returns:
        dict: Summary with best model info and all results
    """
    print(f"\nStarting hyperparameter search for {model_type.upper()}")
    print(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    print(f"Number of classes: {num_classes}")
    
    if param_grid is None:
        param_grid = PARAM_GRID

    if results_filename is None:
        results_filename = f"hp_results_{model_type}.json"

    # Generate all combinations using itertools.product
    param_names = list(param_grid.keys())
    param_values = [param_grid[key] for key in param_names]
    
    combos = []
    for combo_values in itertools.product(*param_values):
        combo_dict = dict(zip(param_names, combo_values))
        combos.append(combo_dict)

    print(f"Parameter grid: {param_grid}")
    print(f"{model_type.upper()} hyperparameter search will run {len(combos)} combos (max_models={max_models})")
    print(f"All parameter combinations: {combos}")
    
    all_results = []
    best_val_acc = -1.0
    best_info = None
    model_count = 0

    # Track total training time
    total_start_time = time.time()
    print(f"Hyperparameter search started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))}")

    for params in combos:
        if (max_models is not None) and (model_count >= max_models):
            break
        model_count += 1
        print("\n" + "=" * 60)
        print(f"{model_type.upper()} Model {model_count}/{len(combos)} - params: {params}")

        # Track training time for this model
        model_start_time = time.time()

        # Extract parameters with defaults
        dropout_rate = float(params.get("dropout_rate", 0.1))
        lr = float(params.get("lr", 2e-5))
        batch_size = int(params.get("batch_size", 16))
        max_length = int(params.get("max_length", MAX_LENGTH))

        # Prepare data for this max_length
        print(f"Preparing data with max_length={max_length}...")
        X_train = prepare_model_data(train_texts, y_train, model_type, model_name, max_length)
        X_val = prepare_model_data(val_texts, y_val, model_type, model_name, max_length)

        model, optimizer, device = build_model(model_type=model_type, model_name=model_name, num_classes=num_classes, max_length=max_length, dropout_rate=dropout_rate, lr=lr)

        # Calculate model size
        model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)
        total_params = sum(p.numel() for p in model.parameters())

        history = train_model(model, optimizer, device, X_train, y_train, X_val, y_val, epochs=3, batch_size=batch_size, patience=3)

        model_train_time = time.time() - model_start_time

        val_acc = max(history["val_accuracy"])
        val_precision = max(history.get("val_precision", [0]))
        val_recall = max(history.get("val_recall", [0]))
        val_f1 = max(history.get("val_f1", [0]))
        val_auc_pr = max(history.get("val_auc_pr", [0]))
        
        print(f"Finished training. Best val_accuracy: {val_acc:.4f}")
        print(f"Best val_precision: {val_precision:.4f}, val_recall: {val_recall:.4f}, val_f1: {val_f1:.4f}, val_auc_pr: {val_auc_pr:.4f}")
        print(f"Training time: {model_train_time:.2f} seconds ({model_train_time / 60:.2f} minutes)")

        os.makedirs(SAVE_MODELS_FOLDER, exist_ok=True)
        # Create model path with all parameters
        param_parts = []
        for k, v in sorted(params.items()):
            if isinstance(v, float):
                # Format float values, replacing . with p and e- with e
                v_str = str(v).replace(".", "p").replace("-", "m").replace("e", "e")
            else:
                v_str = str(v)
            param_parts.append(f"{k}{v_str}")
        param_str = "_".join(param_parts)
        model_path = f"{SAVE_MODELS_FOLDER}/{model_type}_{param_str}.pt"
        print(f"Saving model to {model_path}...")
        torch.save(model.state_dict(), model_path)
        print(f"Saved {model_type.upper()} model to {model_path}")

        plot_history(history, title=f"{model_type.upper()}_{param_str}")

        result = {
            "params": params,
            "val_accuracy": float(val_acc),
            "val_precision": float(val_precision),
            "val_recall": float(val_recall),
            "val_f1": float(val_f1),
            "val_auc_pr": float(val_auc_pr),
            "model_path": model_path,
            "history_keys": list(history.keys()),
            "training_time_seconds": float(model_train_time),
            "model_size_mb": float(model_size_mb),
            "total_parameters": int(total_params),
        }
        all_results.append(result)
        print(f"Current best val_accuracy: {best_val_acc:.4f}, this model: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_info = result
            print(f"New best model found! Val accuracy: {best_val_acc:.4f}")

    total_train_time = time.time() - total_start_time
    print(f"\nHyperparameter search completed in {total_train_time:.2f} seconds ({total_train_time / 60:.2f} minutes)")
    print(f"Trained {model_count} models, best val_accuracy: {best_val_acc:.4f}")

    summary = {"best_model_info": best_info, "all_results": all_results, "total_training_time_seconds": float(total_train_time), "num_models_trained": model_count}
    summary_path = os.path.join(RESULTS_FOLDER, results_filename)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    print(f"Saving summary to {summary_path}...")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {model_type.upper()} hyperparameter search summary to {summary_path}")

    return summary


def evaluate_best_model(model_type, hp_summary, val_texts, y_val, class_names):
    """
    Evaluate the best model from hyperparameter search.

    Args:
        model_type: Type of model ("bert", "electra", or "roberta")
        hp_summary: Hyperparameter search summary dictionary
        val_texts: Validation text data (raw texts, not tokenized)
        y_val: Validation labels
        class_names: List of class names

    Returns:
        dict: Evaluation metrics including model and predictions
    """
    if not hp_summary or not hp_summary.get("best_model_info"):
        print(f"No best model info found for {model_type.upper()}")
        return None

    print("\n" + "=" * 70)
    print(f"Evaluating Best {model_type.upper()} Model")
    print("=" * 70)

    # Load best model
    best_model_path = hp_summary["best_model_info"]["model_path"]
    best_params = hp_summary["best_model_info"]["params"]
    print(f"Loading best {model_type.upper()} model from {best_model_path} ...")
    print(f"Best model params: {best_params}")
    print(f"Best model path exists: {os.path.exists(best_model_path)}")

    # Recreate model architecture
    model_name_map = {"bert": BERT_MODEL_NAME, "electra": ELECTRA_MODEL_NAME, "roberta": ROBERTA_MODEL_NAME}
    model_name = model_name_map[model_type]
    num_classes = len(class_names)
    print(f"Recreating model architecture: {model_type}, {num_classes} classes")

    # Extract parameters with defaults
    dropout_rate = float(best_params.get("dropout_rate", 0.1))
    lr = float(best_params.get("lr", 2e-5))
    max_length = int(best_params.get("max_length", MAX_LENGTH))

    model, _, device = build_model(model_type=model_type, model_name=model_name, num_classes=num_classes, dropout_rate=dropout_rate, lr=lr)

    # Load saved weights
    print(f"Loading weights from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()
    print("Model weights loaded and set to eval mode")

    # Prepare validation data using the best model's max_length
    print(f"Preparing validation data with max_length={max_length}...")
    X_val = prepare_model_data(val_texts, y_val, model_type, model_name, max_length)
    val_input_ids = torch.tensor(X_val["input_ids"], dtype=torch.long).to(device)
    val_attention_mask = torch.tensor(X_val["attention_mask"], dtype=torch.long).to(device)
    print(f"Validation data shape: {val_input_ids.shape}")

    # Make predictions
    print(f"Making predictions on {len(val_input_ids)} samples...")
    with torch.no_grad():
        batch_size = 16
        all_probs = []
        num_batches = (len(val_input_ids) + batch_size - 1) // batch_size
        print(f"Processing {num_batches} batches with batch_size={batch_size}")
        for i in range(0, len(val_input_ids), batch_size):
            batch_input_ids = val_input_ids[i : i + batch_size]
            batch_attention_mask = val_attention_mask[i : i + batch_size]

            outputs = model(batch_input_ids, batch_attention_mask)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

        preds_proba = np.vstack(all_probs)
        preds = preds_proba.argmax(axis=1)
        print(f"Predictions shape: {preds.shape}, probabilities shape: {preds_proba.shape}")
        print(f"Prediction range: [{preds.min()}, {preds.max()}], unique predictions: {len(np.unique(preds))}")

    # Calculate metrics
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    precision = precision_score(y_val, preds, average="macro", zero_division=0)
    recall = recall_score(y_val, preds, average="macro", zero_division=0)

    # Print metrics
    print(f"\n{model_type.upper()} Validation Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

    # Generate classification report
    cm = confusion_matrix(y_val, preds)
    report = classification_report(y_val, preds, target_names=class_names, zero_division=0)
    print(f"\n{model_type.upper()} Classification Report:\n")
    print(report)

    # Save report
    report_path = os.path.join(RESULTS_FOLDER, f"{model_type}_best_model_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Best {model_type.upper()} model path: {best_model_path}\n\n")
        f.write(json.dumps(hp_summary["best_model_info"], indent=2))
        f.write("\n\nValidation Classification Report:\n")
        f.write(report)
    print(f"Saved best {model_type.upper()} model report to {report_path}")

    # Plot confusion matrices
    plot_confusion_matrix(cm, classes=class_names, normalize=False, title=f"{model_type.upper()} Confusion matrix (counts)", label_prefix=model_type)
    plot_confusion_matrix(cm, classes=class_names, normalize=True, title=f"{model_type.upper()} Confusion matrix (normalized)", label_prefix=model_type)

    return {"model": model, "predictions": preds, "probabilities": preds_proba, "metrics": {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}}


def run_inference(weights_path, csv_path, model_type=None, output_path=None, num_classes=6, max_length=128, class_names=None):
    """
    Run inference on new data using a trained model.

    Args:
        weights_path: Path to the saved model weights (.pt file)
        csv_path: Path to CSV file containing texts to classify
        model_type: Model type ("bert", "electra", or "roberta"). If None, infers from weights_path
        output_path: Path to save predictions CSV. If None, saves next to input CSV
        num_classes: Number of output classes (default: 6)
        max_length: Maximum sequence length (default: 128)
        class_names: List of class names for labeling (optional)

    Returns:
        DataFrame with predictions and confidence scores
    """
    print("\n" + "=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70)

    # Infer model type from weights filename if not provided
    if model_type is None:
        weights_filename = os.path.basename(weights_path).lower()
        if "bert" in weights_filename:
            model_type = "bert"
        elif "electra" in weights_filename:
            model_type = "electra"
        elif "roberta" in weights_filename:
            model_type = "roberta"
        else:
            raise ValueError(f"Cannot infer model type from weights path: {weights_path}. Please specify model_type.")

    print(f"Model type: {model_type.upper()}")
    print(f"Weights: {weights_path}")
    print(f"Weights file exists: {os.path.exists(weights_path)}")
    print(f"Input CSV: {csv_path}")
    print(f"Input CSV exists: {os.path.exists(csv_path)}")
    print(f"Parameters: num_classes={num_classes}, max_length={max_length}")

    # Load CSV data
    print(f"\nLoading CSV data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"CSV loaded. Shape: {df.shape}, Columns: {df.columns.tolist()}")
    text_column = "text" if "text" in df.columns else df.columns[0]
    texts = df[text_column].values
    print(f"Loaded {len(texts)} samples from {csv_path}")
    print(f"Using text column: '{text_column}'")

    # Check if labels exist (for evaluation)
    has_labels = "label" in df.columns
    if has_labels:
        true_labels = df["label"].values
        print(f"Found labels column - will compute accuracy. Label shape: {true_labels.shape}, unique labels: {len(np.unique(true_labels))}")
    else:
        print("No labels column found - inference only mode")

    # Get model name
    model_name_map = {"bert": BERT_MODEL_NAME, "electra": ELECTRA_MODEL_NAME, "roberta": ROBERTA_MODEL_NAME}
    model_name = model_name_map[model_type]

    # Prepare data
    print(f"\nPreparing {model_type.upper()} data...")
    X_data = prepare_model_data(texts, None, model_type, model_name, max_length)

    # Extract hyperparameters from filename if possible
    # Default values
    dropout_rate = 0.1
    lr = 2e-5

    # Try to parse from filename (e.g., "bert_dr0.1_lr2e-05.pt")
    weights_filename = os.path.basename(weights_path)
    if "_dr" in weights_filename and "_lr" in weights_filename:
        try:
            dr_part = weights_filename.split("_dr")[1].split("_")[0]
            lr_part = weights_filename.split("_lr")[1].split(".pt")[0]
            dropout_rate = float(dr_part)
            lr = float(lr_part.replace("e-", "e-"))
            print(f"Extracted hyperparameters from filename: dropout_rate={dropout_rate}, lr={lr}")
        except:
            print("Could not parse hyperparameters from filename, using defaults")

    # Build model architecture
    print(f"\nBuilding {model_type.upper()} model...")
    model, _, device = build_model(model_type=model_type, model_name=model_name, num_classes=num_classes, max_length=max_length, dropout_rate=dropout_rate, lr=lr)

    # Load weights
    print(f"Loading weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    # Prepare data tensors
    input_ids = torch.tensor(X_data["input_ids"], dtype=torch.long).to(device)
    attention_mask = torch.tensor(X_data["attention_mask"], dtype=torch.long).to(device)

    # Run inference
    print(f"\nRunning inference on {len(texts)} samples...")
    print(f"Input tensors shape - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
    with torch.no_grad():
        batch_size = 16
        all_probs = []
        num_batches = (len(input_ids) + batch_size - 1) // batch_size
        print(f"Processing {num_batches} batches with batch_size={batch_size}")
        for i in range(0, len(input_ids), batch_size):
            batch_input_ids = input_ids[i : i + batch_size]
            batch_attention_mask = attention_mask[i : i + batch_size]

            outputs = model(batch_input_ids, batch_attention_mask)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

        preds_proba = np.vstack(all_probs)
        preds = preds_proba.argmax(axis=1)
        confidence = preds_proba.max(axis=1)
        print(f"Inference complete. Predictions shape: {preds.shape}, probabilities shape: {preds_proba.shape}")
        print(f"Confidence range: [{confidence.min():.4f}, {confidence.max():.4f}], mean: {confidence.mean():.4f}")

    # Create output dataframe
    output_df = df.copy()
    output_df["predicted_label"] = preds
    output_df["predicted_confidence"] = confidence

    # Add class names if provided
    if class_names:
        output_df["predicted_emotion"] = [class_names[p] for p in preds]

    # Calculate accuracy if labels exist
    if has_labels:
        acc = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds, average="macro")
        print("\nEvaluation Metrics:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Macro F1: {f1:.4f}")

    # Save predictions
    if output_path is None:
        input_dir = os.path.dirname(csv_path)
        input_base = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(input_dir, f"{input_base}_predictions.csv")

    output_df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to: {output_path}")
    print("=" * 70)

    return output_df


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Starting Emotion Detection Pipeline")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  TRAIN_FILE: {TRAIN_FILE}")
    print(f"  VALIDATION_FILE: {VALIDATION_FILE}")
    print(f"  TEST_FILE: {TEST_FILE}")
    print(f"  RESULTS_FOLDER: {RESULTS_FOLDER}")
    print(f"  SAVE_MODELS_FOLDER: {SAVE_MODELS_FOLDER}")
    print(f"  MAX_LENGTH: {MAX_LENGTH}")
    print(f"  PARAM_GRID: {PARAM_GRID}")
    
    # Load data
    train_texts, train_labels = load_data(TRAIN_FILE, "Training Data")
    val_texts, val_labels = load_data(VALIDATION_FILE, "Validation Data")

    num_classes = len(np.unique(train_labels))
    class_names = ["sadness", "joy", "love", "anger", "fear", "suprise"]
    print(f"\nNumber of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    if len(class_names) != num_classes:
        print(f"Warning: Number of class names ({len(class_names)}) doesn't match num_classes ({num_classes})")

    # Run BERT hyperparameter search
    print("\n" + "=" * 70)
    print("BERT HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(f"\nStarting BERT hyperparameter search...")
    hp_summary_bert = hyperparameter_search(
        model_type="bert",
        train_texts=train_texts,
        y_train=train_labels,
        val_texts=val_texts,
        y_val=val_labels,
        num_classes=num_classes,
        model_name=BERT_MODEL_NAME,
        param_grid=PARAM_GRID,
        max_models=None,  # Will train all combinations
        results_filename="hp_results_bert.json",
    )

    # Evaluate best BERT model
    print(f"\nEvaluating best BERT model...")
    bert_eval = evaluate_best_model("bert", hp_summary_bert, val_texts, val_labels, class_names)
    print(f"BERT evaluation complete")

    # ============================================================================
    # ELECTRA MODEL TRAINING
    # ============================================================================
    print("\n" + "=" * 70)
    print("ELECTRA HYPERPARAMETER SEARCH")
    print("=" * 70)

    # Run ELECTRA hyperparameter search
    print(f"\nStarting ELECTRA hyperparameter search...")
    hp_summary_electra = hyperparameter_search(
        model_type="electra",
        train_texts=train_texts,
        y_train=train_labels,
        val_texts=val_texts,
        y_val=val_labels,
        num_classes=num_classes,
        model_name=ELECTRA_MODEL_NAME,
        param_grid=PARAM_GRID,
        max_models=None,  # Will train all combinations
        results_filename="hp_results_electra.json",
    )

    # Evaluate best ELECTRA model
    print(f"\nEvaluating best ELECTRA model...")
    electra_eval = evaluate_best_model("electra", hp_summary_electra, val_texts, val_labels, class_names)
    print(f"ELECTRA evaluation complete")

    # ============================================================================
    # ROBERTA MODEL TRAINING
    # ============================================================================
    print("\n" + "=" * 70)
    print("ROBERTA HYPERPARAMETER SEARCH")
    print("=" * 70)

    # Run RoBERTa hyperparameter search
    print(f"\nStarting RoBERTa hyperparameter search...")
    hp_summary_roberta = hyperparameter_search(
        model_type="roberta",
        train_texts=train_texts,
        y_train=train_labels,
        val_texts=val_texts,
        y_val=val_labels,
        num_classes=num_classes,
        model_name=ROBERTA_MODEL_NAME,
        param_grid=PARAM_GRID,
        max_models=None,  # Will train all combinations
        results_filename="hp_results_roberta.json",
    )

    # Evaluate best RoBERTa model
    print(f"\nEvaluating best RoBERTa model...")
    roberta_eval = evaluate_best_model("roberta", hp_summary_roberta, val_texts, val_labels, class_names)
    print(f"RoBERTa evaluation complete")

    # ============================================================================
    # MODEL COMPARISON
    # ============================================================================
    print("\n" + "=" * 70)
    print("Model Comparison Summary")
    print("=" * 70)
    print("Collecting results from all models...")

    comparison_results = []
    print(f"BERT summary available: {hp_summary_bert is not None and hp_summary_bert.get('best_model_info') is not None}")
    print(f"ELECTRA summary available: {hp_summary_electra is not None and hp_summary_electra.get('best_model_info') is not None}")
    print(f"RoBERTa summary available: {hp_summary_roberta is not None and hp_summary_roberta.get('best_model_info') is not None}")

    if hp_summary_bert and hp_summary_bert.get("best_model_info"):
        bert_info = hp_summary_bert["best_model_info"]
        comparison_results.append(
            {
                "model": "BERT",
                "best_params": bert_info["params"],
                "val_accuracy": bert_info.get("val_accuracy", 0),
                "val_precision": bert_info.get("val_precision", 0),
                "val_recall": bert_info.get("val_recall", 0),
                "val_f1": bert_info.get("val_f1", 0),
                "val_auc_pr": bert_info.get("val_auc_pr", 0),
                "model_path": bert_info["model_path"],
                "training_time_seconds": bert_info.get("training_time_seconds", 0),
                "model_size_mb": bert_info.get("model_size_mb", 0),
                "total_parameters": bert_info.get("total_parameters", 0),
                "total_training_time": hp_summary_bert.get("total_training_time_seconds", 0),
            }
        )

    if hp_summary_electra and hp_summary_electra.get("best_model_info"):
        electra_info = hp_summary_electra["best_model_info"]
        comparison_results.append(
            {
                "model": "ELECTRA",
                "best_params": electra_info["params"],
                "val_accuracy": electra_info.get("val_accuracy", 0),
                "val_precision": electra_info.get("val_precision", 0),
                "val_recall": electra_info.get("val_recall", 0),
                "val_f1": electra_info.get("val_f1", 0),
                "val_auc_pr": electra_info.get("val_auc_pr", 0),
                "model_path": electra_info["model_path"],
                "training_time_seconds": electra_info.get("training_time_seconds", 0),
                "model_size_mb": electra_info.get("model_size_mb", 0),
                "total_parameters": electra_info.get("total_parameters", 0),
                "total_training_time": hp_summary_electra.get("total_training_time_seconds", 0),
            }
        )

    if hp_summary_roberta and hp_summary_roberta.get("best_model_info"):
        roberta_info = hp_summary_roberta["best_model_info"]
        comparison_results.append(
            {
                "model": "RoBERTa",
                "best_params": roberta_info["params"],
                "val_accuracy": roberta_info.get("val_accuracy", 0),
                "val_precision": roberta_info.get("val_precision", 0),
                "val_recall": roberta_info.get("val_recall", 0),
                "val_f1": roberta_info.get("val_f1", 0),
                "val_auc_pr": roberta_info.get("val_auc_pr", 0),
                "model_path": roberta_info["model_path"],
                "training_time_seconds": roberta_info.get("training_time_seconds", 0),
                "model_size_mb": roberta_info.get("model_size_mb", 0),
                "total_parameters": roberta_info.get("total_parameters", 0),
                "total_training_time": hp_summary_roberta.get("total_training_time_seconds", 0),
            }
        )

    # Sort by validation accuracy (descending)
    comparison_results.sort(key=lambda x: x["val_accuracy"], reverse=True)

    # Print comprehensive comparison table
    print("\n" + "=" * 120)
    print(" " * 40 + "COMPREHENSIVE MODEL COMPARISON")
    print("=" * 120)
    print(f"{'Model':<12} {'Val Acc':<10} {'Parameters':<15} {'Size (MB)':<12} {'Train Time':<15} {'Total Time':<15}")
    print("-" * 120)
    for result in comparison_results:
        model_name = result["model"]
        val_acc = f"{result['val_accuracy']:.4f}"
        params = f"{result['total_parameters']:,}"
        size_mb = f"{result['model_size_mb']:.2f}"
        train_time = f"{result['training_time_seconds']:.1f}s"
        total_time = f"{result['total_training_time'] / 60:.1f}min"
        print(f"{model_name:<12} {val_acc:<10} {params:<15} {size_mb:<12} {train_time:<15} {total_time:<15}")
    print("=" * 120)

    print("\nDetailed Model Rankings (by validation accuracy):")
    print("-" * 70)
    for i, result in enumerate(comparison_results, 1):
        print(f"{i}. {result['model']}: {result['val_accuracy']:.4f}")
        print(f"   Metrics - Accuracy: {result.get('val_accuracy', 0):.4f}, Precision: {result.get('val_precision', 0):.4f}, Recall: {result.get('val_recall', 0):.4f}, F1: {result.get('val_f1', 0):.4f}, AUC-PR: {result.get('val_auc_pr', 0):.4f}")
        print(f"   Best params: {result['best_params']}")
        print(f"   Model size: {result['model_size_mb']:.2f} MB ({result['total_parameters']:,} parameters)")
        print(f"   Training time: {result['training_time_seconds']:.2f} seconds ({result['training_time_seconds'] / 60:.2f} minutes)")
        print(f"   Total time (all models): {result['total_training_time'] / 60:.2f} minutes")
        print(f"   Model path: {result['model_path']}")
        print()

    # Save comparison to JSON
    comparison_path = os.path.join(RESULTS_FOLDER, "model_comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Saved model comparison to {comparison_path}")

    # Save comparison table as CSV
    comparison_df = pd.DataFrame(
        [
            {
                "Model": r["model"],
                "Validation Accuracy": r.get("val_accuracy", 0),
                "Validation Precision": r.get("val_precision", 0),
                "Validation Recall": r.get("val_recall", 0),
                "Validation F1": r.get("val_f1", 0),
                "Validation AUC-PR": r.get("val_auc_pr", 0),
                "Total Parameters": r["total_parameters"],
                "Model Size (MB)": r["model_size_mb"],
                "Training Time (seconds)": r["training_time_seconds"],
                "Training Time (minutes)": r["training_time_seconds"] / 60,
                "Total Search Time (minutes)": r["total_training_time"] / 60,
                "Dropout Rate": r["best_params"].get("dropout_rate", 0),
                "Learning Rate": r["best_params"].get("lr", 0),
                "Batch Size": r["best_params"].get("batch_size", 0),
                "Max Length": r["best_params"].get("max_length", 0),
            }
            for r in comparison_results
        ]
    )
    comparison_csv_path = os.path.join(RESULTS_FOLDER, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"Saved model comparison table to {comparison_csv_path}")

    # ============================================================================
    # TEST PREDICTIONS
    # ============================================================================
    print(f"\nChecking if test predictions should be generated...")
    print(f"Comparison results available: {len(comparison_results) > 0}")
    print(f"TEST_FILE exists: {os.path.exists(TEST_FILE) if TEST_FILE else False}")
    
    if comparison_results and os.path.exists(TEST_FILE):
        print("\n" + "=" * 70)
        print("Generating Test Predictions with Best Model")
        print("=" * 70)

        # Get best model info
        best_result = comparison_results[0]
        best_model_name = best_result["model"]
        best_model_type = best_model_name.lower()
        print(f"\nUsing {best_model_name} model (val_accuracy: {best_result['val_accuracy']:.4f})")
        print(f"Best model path: {best_result['model_path']}")

        # Load test data
        test_df = pd.read_csv(TEST_FILE)
        test_texts, test_labels = load_data(TEST_FILE, "Test Data")

        # Prepare test data for the best model
        model_name_map = {"bert": BERT_MODEL_NAME, "electra": ELECTRA_MODEL_NAME, "roberta": ROBERTA_MODEL_NAME}
        X_test = prepare_model_data(test_texts, test_labels, best_model_type, model_name_map[best_model_type], MAX_LENGTH)

        # Load the best model
        best_params = best_result["best_params"]
        model, _, device = build_model(model_type=best_model_type, model_name=model_name_map[best_model_type], num_classes=num_classes, dropout_rate=best_params["dropout_rate"], lr=best_params["lr"])
        model.load_state_dict(torch.load(best_result["model_path"], map_location=device, weights_only=True))
        model.eval()

        # Prepare test data tensors
        test_input_ids = torch.tensor(X_test["input_ids"], dtype=torch.long).to(device)
        test_attention_mask = torch.tensor(X_test["attention_mask"], dtype=torch.long).to(device)

        # Make predictions
        with torch.no_grad():
            batch_size = 16
            all_probs = []
            for i in range(0, len(test_input_ids), batch_size):
                batch_input_ids = test_input_ids[i : i + batch_size]
                batch_attention_mask = test_attention_mask[i : i + batch_size]

                outputs = model(batch_input_ids, batch_attention_mask)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())

            preds_proba = np.vstack(all_probs)
            preds = preds_proba.argmax(axis=1)
            probs = preds_proba.max(axis=1)

        if test_labels is not None:
            acc = accuracy_score(test_labels, preds)
            print(f"Test accuracy: {acc:.4f}")

        out_df = test_df.copy().reset_index(drop=True)
        out_df["predicted_label"] = preds
        out_df["predicted_confidence"] = probs

        output_path = os.path.join(os.path.dirname(TEST_FILE), "test_predictions.csv")
        out_df.to_csv(output_path, index=False)
        print(f"Saved test predictions to {output_path}")
        print(f"Predictions generated using {best_model_name} model")
    else:
        print("Skipping test predictions (no comparison results or test file not found)")

    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print(f"Results saved to: {RESULTS_FOLDER}")
    print(f"Models saved to: {SAVE_MODELS_FOLDER}")
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"Pipeline finished at: {end_time}")
