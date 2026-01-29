"""
Transformer Models for Emotion Detection
This script handles text preprocessing and Transformer-based emotion classification.
Models: BERT, ELECTRA, RoBERTa
"""

print("Starting imports...")
print("Importing basic modules...")

import os
import copy
import sys

# Force transformers to use PyTorch only (must be set before importing transformers)
os.environ["TRANSFORMERS_NO_TF"] = "1"

import itertools
import json
import re
import time

print("Importing scientific libraries (numpy, pandas, matplotlib)...")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("Importing PyTorch and sklearn...")
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from sklearn.preprocessing import label_binarize
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

print("Importing transformers library (this may take a moment)...")
from transformers import (
    BertModel,
    BertTokenizer,
    ElectraModel,
    ElectraTokenizer,
    RobertaModel,
    RobertaTokenizer,
)
print("All imports completed successfully!")

# ============================================================================
# OUTPUT CAPTURE CLASS
# ============================================================================
class TeeOutput:
    """Captures stdout to both console and a file (like tee command)."""
    
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def isatty(self):
        """Return False since TeeOutput is not a terminal."""
        return False
    
    def close(self):
        self.log.close()

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
TRAIN_FILE = "./data/train.csv"
VALIDATION_FILE = "./data/validation.csv"
RESULTS_FOLDER = "./results"
SAVE_MODELS_FOLDER = "./hp_models"
CHECKPOINT_FILE = "./results/training_checkpoint.json"  # Tracks training progress for resume

# Models parameters
BERT_MODEL_NAME = "bert-base-uncased"
ELECTRA_MODEL_NAME = "google/electra-small-discriminator"
ROBERTA_MODEL_NAME = "roberta-base"
MODEL_NAMES = {"bert": BERT_MODEL_NAME, "electra": ELECTRA_MODEL_NAME, "roberta": ROBERTA_MODEL_NAME}

MAX_LENGTH = 128
PARAM_GRID = {
     "dropout_rate": [0.1, 0.3],
     "lr": [2e-5, 3e-5],
     "batch_size": [16, 32],
     "weight_decay": [0.0, 0.01],
}

# PARAM_GRID = {
#     "dropout_rate": [0.1],
#     "lr": [2e-5],
#     "batch_size": [16],
#     "weight_decay": [0.0],
# }
NUM_CLASSES = 6
CLASS_NAMES = ["sadness", "joy", "love", "anger", "fear", "suprise"]
NUM_EPOCHS = 10
PATIENCE = 1  # Early stopping patience (epochs without improvement before stopping)
FREEZE_TRANSFORMER = False  # Set to True to freeze transformer weights during training

### Inference test data section
RUN_INFERENCE_ONLY = False
TEST_FILE = "./data/test.csv"
BEST_MODEL_WEIGHTS = ""

def set_seed(seed=42):
    """
    Set constant seed for all random variables.

    Args:
        seed: the seed to configure
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint():
    """
    Load training checkpoint to resume from previous run.
    
    Returns:
        dict: Checkpoint data with completed models and stage info, or empty dict if no checkpoint
    """
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
            print(f"Loaded checkpoint from {CHECKPOINT_FILE}")
            print(f"Previously completed: {checkpoint.get('completed_models', [])} models")
            return checkpoint
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return {}
    return {}


def save_checkpoint(checkpoint):
    """
    Save training checkpoint to resume later if interrupted.
    
    Args:
        checkpoint: dict with training progress information
    """
    try:
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"Checkpoint saved to {CHECKPOINT_FILE}")
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}")


def is_model_trained(checkpoint, model_type, params):
    """
    Check if a specific model configuration has already been trained.
    
    Args:
        checkpoint: Checkpoint dictionary
        model_type: Type of model (bert, electra, roberta)
        params: Model parameters dict
        
    Returns:
        bool: True if model is already trained
    """
    completed = checkpoint.get('completed_models', [])
    model_key = f"{model_type}_{json.dumps(params, sort_keys=True)}"
    return model_key in completed


def mark_model_complete(checkpoint, model_type, params, result=None):
    """
    Mark a model configuration as completed in checkpoint.
    
    Args:
        checkpoint: Checkpoint dictionary
        model_type: Type of model
        params: Model parameters dict
        result: Optional result dictionary to store with the checkpoint
    """
    if 'completed_models' not in checkpoint:
        checkpoint['completed_models'] = []
    model_key = f"{model_type}_{json.dumps(params, sort_keys=True)}"
    if model_key not in checkpoint['completed_models']:
        checkpoint['completed_models'].append(model_key)
    
    # Store results for later comparison
    if result is not None:
        if 'model_results' not in checkpoint:
            checkpoint['model_results'] = {}
        checkpoint['model_results'][model_key] = result


def get_model_result(checkpoint, model_type, params):
    """
    Get stored result for a completed model from checkpoint.
    
    Args:
        checkpoint: Checkpoint dictionary
        model_type: Type of model
        params: Model parameters dict
        
    Returns:
        dict: Stored result dictionary or None if not found
    """
    model_key = f"{model_type}_{json.dumps(params, sort_keys=True)}"
    return checkpoint.get('model_results', {}).get(model_key)


def load_data(data_file, dataset_name="Data"):
    """
    Load data from CSV file and do pre-process of the texts.

    Args:
        data_file: Path to the CSV file OR a pandas DataFrame
        dataset_name: Name of the dataset

    Returns:
        tuple: (Preprocess texts, labels)
    """
    print("=" * 70)
    print(f"Loading {dataset_name}")
    print("=" * 70)
    
    # Check if data_file is a DataFrame or a file path
    if isinstance(data_file, pd.DataFrame):
        df = data_file
        print(f"Using provided DataFrame")
    else:
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

    # Preprocess texts
    print(f"Preprocessing {len(texts)} texts...")
    processed_texts = [preprocess_text(str(text)) for text in texts]
    print(f"Preprocessing complete. Sample length range: [{min(len(t) for t in processed_texts)}, {max(len(t) for t in processed_texts)}]")

    if label_column is None:
        print(f"Number of samples: {len(texts)}")
        print(f"No labels found, returning processed texts only")
        return processed_texts, None

    labels = df[label_column].values
    print(f"Number of samples: {len(texts)}")
    print(f"Number of unique emotions: {len(np.unique(labels))}")
    print(f"Emotion distribution: {np.bincount(labels)}")
    print(f"Label range: [{labels.min()}, {labels.max()}]")

    return processed_texts, labels


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
    
    # Determine plot style based on number of epochs
    # Use markers only for single epoch, lines for multiple epochs
    if len(loss) == 1:
        train_style = "bo"  # Blue circle marker only
        val_style = "ro"    # Red circle marker only
    else:
        train_style = "b-"  # Blue line
        val_style = "r--"   # Red dashed line

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16)

    # Loss
    axes[0, 0].plot(epochs, loss, train_style, label="Training loss", markersize=8)
    axes[0, 0].plot(epochs, val_loss, val_style, label="Validation loss", markersize=8)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, acc, train_style, label="Training accuracy", markersize=8)
    axes[0, 1].plot(epochs, val_acc, val_style, label="Validation accuracy", markersize=8)
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision
    if precision and val_precision:
        axes[0, 2].plot(epochs, precision, train_style, label="Training precision", markersize=8)
        axes[0, 2].plot(epochs, val_precision, val_style, label="Validation precision", markersize=8)
        axes[0, 2].set_title("Precision (Macro)")
        axes[0, 2].set_xlabel("Epochs")
        axes[0, 2].set_ylabel("Precision")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # Recall
    if recall and val_recall:
        axes[1, 0].plot(epochs, recall, train_style, label="Training recall", markersize=8)
        axes[1, 0].plot(epochs, val_recall, val_style, label="Validation recall", markersize=8)
        axes[1, 0].set_title("Recall (Macro)")
        axes[1, 0].set_xlabel("Epochs")
        axes[1, 0].set_ylabel("Recall")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # F1 Score
    if f1 and val_f1:
        axes[1, 1].plot(epochs, f1, train_style, label="Training F1", markersize=8)
        axes[1, 1].plot(epochs, val_f1, val_style, label="Validation F1", markersize=8)
        axes[1, 1].set_title("F1 Score (Macro)")
        axes[1, 1].set_xlabel("Epochs")
        axes[1, 1].set_ylabel("F1 Score")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # AUC-PR
    if auc_pr and val_auc_pr:
        axes[1, 2].plot(epochs, auc_pr, train_style, label="Training AUC-PR", markersize=8)
        axes[1, 2].plot(epochs, val_auc_pr, val_style, label="Validation AUC-PR", markersize=8)
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


def prepare_model_data(processed_texts, labels=None, model_type="bert"):
    """
    Prepare data for transformer models using appropriate tokenizer.
    Applies model-specific tokenization.

    Args:
        processed_texts: Array of clean text strings
        labels: Optional array of labels
        model_type: Type of model ("bert", "electra", or "roberta")

    Returns:
        dict: Dictionary with 'input_ids', 'attention_mask', and optionally 'labels'
    """
    print(f"\nStarting data preparation for {model_type.upper()}")
    print(f"Input texts count: {len(processed_texts)}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Labels provided: {labels is not None}")

    model_name = MODEL_NAMES[model_type]
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

    # Tokenize
    print(f"Tokenizing texts with max_length={MAX_LENGTH}...")
    encoded = tokenizer(processed_texts, add_special_tokens=True, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_attention_mask=True, return_tensors="np")
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

    def __init__(self, model_type, dropout_rate=0.1):
        model_name = MODEL_NAMES[model_type]
        super(TransformerClassifier, self).__init__()
        print(f"Initializing TransformerClassifier: type={model_type}, model_name={model_name}, num_classes={NUM_CLASSES}, dropout_rate={dropout_rate}")

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
        self.fc2 = nn.Linear(128, NUM_CLASSES)
        print(f"Classification head: {hidden_size} -> 128 -> {NUM_CLASSES}")

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


def build_model(model_type="bert", dropout_rate=0.1, lr=2e-5, weight_decay=0.0, freeze_transformer_weights=False):
    """
    Build and compile a transformer-based classification model.

    Args:
        model_type: Type of model ("bert", "electra", or "roberta")
        dropout_rate: Dropout rate for classification head
        lr: Learning rate for AdamW optimizer
        weight_decay: Weight decay for AdamW optimizer
        freeze_transformer_weights: If True, freeze transformer parameters (only train classification head)

    Returns:
        Tuple of (model, optimizer, device)
    """
    print(f"\nBuilding {model_type.upper()} model...")
    print(f"Parameters: num_classes={NUM_CLASSES}, dropout_rate={dropout_rate}, lr={lr}, weight_decay={weight_decay}")

    # Create model
    model = TransformerClassifier(model_type, dropout_rate)

    # Freeze transformer params if requested
    if freeze_transformer_weights:
        freeze_transformer(model)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Moving model to device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    model.to(device)

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"Optimizer created: AdamW with lr={lr}, weight_decay={weight_decay}")

    # Calculate model size
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{model_type.upper()} model built successfully!")
    print(f"Model device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model, optimizer, device


def train_model(model, optimizer, device, X_train, y_train, X_val, y_val, epochs=NUM_EPOCHS, batch_size=16, patience=PATIENCE):
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

    # Inverse frequency weighting: weight = total_samples / (NUM_CLASSES * class_count)
    class_weights = torch.FloatTensor([total_samples / (NUM_CLASSES * count) for count in class_counts])
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
        "train_run_time": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auc_pr": [],
        "val_run_time": [],
        "val_confusion_matrix": None,
        "val_classification_report": None
    }

    # Early stopping variables
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    # Training loop
    print(f"\nBeginning training loop...")
    for epoch in range(epochs):
        print(f"\nStarting Epoch {epoch + 1}/{epochs}")
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_all_preds = []
        train_all_labels = []
        train_all_probs = []

        if device.type == "cuda":
            torch.cuda.synchronize()
        train_start_time = time.perf_counter()

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
            
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_run_time = time.perf_counter() - train_start_time

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Calculate additional metrics for training
        train_all_probs = np.vstack(train_all_probs)
        train_precision = precision_score(train_all_labels, train_all_preds, average="macro", zero_division=0)
        train_recall = recall_score(train_all_labels, train_all_preds, average="macro", zero_division=0)
        train_f1 = f1_score(train_all_labels, train_all_preds, average="macro", zero_division=0)
        train_labels_bin = label_binarize(train_all_labels, classes=range(NUM_CLASSES))
        train_auc_pr = average_precision_score(train_labels_bin, train_all_probs, average="macro")
        
        print(f"Epoch {epoch + 1}/{epochs} results:\n\tTrain - loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, precision: {train_precision:.4f}, recall: {train_recall:.4f}, f1: {train_f1:.4f}, auc_pr: {train_auc_pr:.4f}, time: {train_run_time:.2f}s")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_labels = []
        val_all_probs = []

        if device.type == "cuda":
            torch.cuda.synchronize()
        val_start_time = time.perf_counter()

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

        if device.type == "cuda":
            torch.cuda.synchronize()
        val_run_time = time.perf_counter()-val_start_time

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Calculate additional metrics for validation
        val_all_probs = np.vstack(val_all_probs)
        val_precision = precision_score(val_all_labels, val_all_preds, average="macro", zero_division=0)
        val_recall = recall_score(val_all_labels, val_all_preds, average="macro", zero_division=0)
        val_f1 = f1_score(val_all_labels, val_all_preds, average="macro", zero_division=0)
        val_labels_bin = label_binarize(val_all_labels, classes=range(NUM_CLASSES))
        val_auc_pr = average_precision_score(val_labels_bin, val_all_probs, average="macro")
        
        print(f"  Val   - loss: {val_loss:.4f}, accuracy: {val_acc:.4f}, precision: {val_precision:.4f}, recall: {val_recall:.4f}, f1: {val_f1:.4f}, auc_pr: {val_auc_pr:.4f}, time: {val_run_time:.2f}s")

        # Update history
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["precision"].append(train_precision)
        history["recall"].append(train_recall)
        history["f1"].append(train_f1)
        history["auc_pr"].append(train_auc_pr)
        history["train_run_time"].append(train_run_time)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)
        history["val_auc_pr"].append(val_auc_pr)
        history["val_run_time"].append(val_run_time)

        print(f"  Total epoch time: {train_run_time + val_run_time:.2f}s (train: {train_run_time:.2f}s, val: {val_run_time:.2f}s)")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            history["val_confusion_matrix"] = confusion_matrix(val_all_labels, val_all_preds)
            history["val_classification_report"] = classification_report(val_all_labels, val_all_preds, target_names=CLASS_NAMES, zero_division=0)
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


def hyperparameter_search(model_type, train_texts, y_train, val_texts, y_val, param_grid=None, max_models=None, results_filename=None):
    """
    Run hyperparameter search for any transformer model.

    Args:
        model_type: Type of model ("bert", "electra", or "roberta")
        train_texts: Training text data (raw texts, not tokenized)
        y_train: Training labels
        val_texts: Validation text data (raw texts, not tokenized)
        y_val: Validation labels
        param_grid: Dictionary of hyperparameters to search
        max_models: Maximum number of models to train
        results_filename: Filename to save results (if None, auto-generated)

    Returns:
        dict: Summary with best model info and all results
    """
    print(f"\nStarting hyperparameter search for {model_type.upper()}")
    print(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    print(f"Number of classes: {NUM_CLASSES}")
    
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

    # Load checkpoint to resume from previous run
    checkpoint = load_checkpoint()
    
    # Prepare data with fixed max_length
    print(f"Preparing data with max_length={MAX_LENGTH}...")
    X_train = prepare_model_data(train_texts, y_train, model_type)
    X_val = prepare_model_data(val_texts, y_val, model_type)

    all_results = []
    best_val_f1 = -1.0
    best_cm = None
    best_report = None
    best_info = None
    model_count = 0
    skipped_count = 0

    # Track total training time
    total_start_time = time.time()
    print(f"Hyperparameter search started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))}")

    for params in combos:
        # Check if this model was already trained
        if is_model_trained(checkpoint, model_type, params):
            skipped_count += 1
            print(f"\n[SKIPPED] {model_type.upper()} model with params {params} already trained (found in checkpoint)")
            
            # Try to load existing results
            param_parts = []
            for k, v in sorted(params.items()):
                v_str = str(v)
                param_parts.append(f"{k}{v_str}")
            param_str = "_".join(param_parts)
            model_path = f"{SAVE_MODELS_FOLDER}/{model_type}_{param_str}.pt"
            
            # If model file exists, we can skip; otherwise retrain
            if os.path.exists(model_path):
                print(f"Model file found: {model_path}")
                
                # Load stored results from checkpoint for comparison
                stored_result = get_model_result(checkpoint, model_type, params)
                if stored_result is not None:
                    all_results.append(stored_result)
                    val_f1 = stored_result["val_f1"]
                    print(f"Loaded stored result: val_f1={val_f1:.4f}")
                    
                    # Check if this is the best model so far
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_info = copy.deepcopy(stored_result)
                        print(f"This skipped model is currently the best! Val f1: {best_val_f1:.4f}")
                        
                        # Note: We don't have confusion matrix and classification report in checkpoint
                        # They will be regenerated at the end if this is the final best model
                        best_cm = None
                        best_report = None
                else:
                    print(f"Warning: No stored results found in checkpoint for this model")
                
                continue
            else:
                print(f"Warning: Model file not found, will retrain")
        
        if (max_models is not None) and (model_count >= max_models):
            break
        model_count += 1
        # Calculate remaining models to train (total - skipped)
        remaining_to_train = len(combos) - skipped_count
        print("\n" + "=" * 60)
        print(f"{model_type.upper()} Model {model_count}/{remaining_to_train} - params: {params}")

        # Extract parameters with defaults
        dropout_rate = float(params.get("dropout_rate", 0.1))
        lr = float(params.get("lr", 2e-5))
        batch_size = int(params.get("batch_size", 16))
        weight_decay = float(params.get("weight_decay", 0.0))

        model, optimizer, device = build_model(model_type=model_type, dropout_rate=dropout_rate, lr=lr, weight_decay=weight_decay, freeze_transformer_weights=FREEZE_TRANSFORMER)

        # Calculate model size
        total_params = sum(p.numel() for p in model.parameters())

        history = train_model(model, optimizer, device, X_train, y_train, X_val, y_val, epochs=NUM_EPOCHS, batch_size=batch_size, patience=PATIENCE)

        # Restore model metric results on validation set
        idx = history["val_loss"].index(min(history["val_loss"]))
        val_acc = history["val_accuracy"][idx]
        val_precision = history["val_precision"][idx]
        val_recall = history["val_recall"][idx]
        val_f1 = history["val_f1"][idx]
        val_auc_pr = history["val_auc_pr"][idx]
        train_run_time = sum(history["train_run_time"])/len(history["train_run_time"])
        val_run_time = sum(history["val_run_time"])/len(history["val_run_time"])
        epoch_run_time = train_run_time + val_run_time
        
        print(f"Finished training. val_accuracy: {val_acc:.4f}, val_precision: {val_precision:.4f}, val_recall: {val_recall:.4f}, val_f1: {val_f1:.4f}, val_auc_pr: {val_auc_pr:.4f}")
        print(f"Average epoch time: {epoch_run_time:.2f} seconds ({epoch_run_time / 60:.2f} minutes) - train: {train_run_time:.2f}s, val: {val_run_time:.2f}s")
        print(f"Inference time (on average): {val_run_time:.2f} seconds ({val_run_time / 60:.2f} minutes)")

        os.makedirs(SAVE_MODELS_FOLDER, exist_ok=True)
        # Create model path with all parameters
        param_parts = []
        for k, v in sorted(params.items()):
            v_str = str(v)
            param_parts.append(f"{k}{v_str}")
        param_str = "_".join(param_parts)
        model_path = f"{SAVE_MODELS_FOLDER}/{model_type}_{param_str}.pt"
        print(f"Saving model to {model_path}...")
        torch.save(model.state_dict(), model_path)
        print(f"Saved {model_type.upper()} model to {model_path}")

        model_size_mb = os.path.getsize(model_path) / (1024 ** 2)

        plot_history(history, title=f"{model_type.upper()}_{param_str}")

        result = {
            "params": params,
            "val_accuracy": float(val_acc),
            "val_precision": float(val_precision),
            "val_recall": float(val_recall),
            "val_f1": float(val_f1),
            "val_auc_pr": float(val_auc_pr),
            "model_path": model_path,
            "val_run_time_seconds": float(val_run_time),
            "model_size_mb": float(model_size_mb),
            "total_parameters": int(total_params),
        }
        all_results.append(result)
        print(f"Current best val_f1: {best_val_f1:.4f}, this model: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_info = copy.deepcopy(result)
            best_cm = history["val_confusion_matrix"].copy()
            best_report = history["val_classification_report"]
            print(f"New best model found! Val f1: {best_val_f1:.4f}")
        
        # Mark model as complete and save checkpoint with results
        mark_model_complete(checkpoint, model_type, params, result)
        save_checkpoint(checkpoint)

    total_train_time = time.time() - total_start_time
    print(f"\nHyperparameter search completed in {total_train_time:.2f} seconds ({total_train_time / 60:.2f} minutes)")
    print(f"Total combinations: {len(combos)} | Trained: {model_count} | Skipped: {skipped_count}")
    print(f"Best val_f1: {best_val_f1:.4f}")
    
    # If best model was skipped, we need to regenerate confusion matrix and report
    if best_cm is None or best_report is None:
        print(f"\nBest model was loaded from checkpoint. Regenerating evaluation metrics...")
        best_params = best_info["params"]
        dropout_rate = float(best_params.get("dropout_rate", 0.1))
        lr = float(best_params.get("lr", 2e-5))
        weight_decay = float(best_params.get("weight_decay", 0.0))
        
        # Load the best model
        model, optimizer, device = build_model(model_type=model_type, dropout_rate=dropout_rate, lr=lr, weight_decay=weight_decay, freeze_transformer_weights=FREEZE_TRANSFORMER)
        model.load_state_dict(torch.load(best_info["model_path"], map_location=device, weights_only=True))
        model.eval()
        
        # Evaluate to get confusion matrix and classification report
        eval_result = evaluate_model(model_type=model_type, X=X_val, y=y_val, model=model, 
                                      model_params=best_params, model_prefix='best_reeval')
        
        from sklearn.metrics import confusion_matrix, classification_report
        best_cm = confusion_matrix(y_val, eval_result["predictions"])
        best_report = classification_report(y_val, eval_result["predictions"], 
                                           target_names=CLASS_NAMES, zero_division=0)
        print(f"Regenerated metrics for best model")

    # Generate classification report
    print(f"\n{model_type.upper()} Classification Report:\n")
    print(best_report)

    # Save report
    report_path = os.path.join(RESULTS_FOLDER, f"{model_type}_best_model_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f'Best {model_type.upper()} model path: {best_info["model_path"]}\n\n')
        f.write(json.dumps(best_info, indent=2))
        f.write("\n\nValidation Classification Report:\n")
        f.write(best_report)
    print(f"Saved best {model_type.upper()} model report to {report_path}")

    # Plot confusion matrices
    plot_confusion_matrix(best_cm, classes=CLASS_NAMES, normalize=False,
                          title=f"{model_type.upper()} Confusion matrix (counts)", label_prefix=model_type)
    plot_confusion_matrix(best_cm, classes=CLASS_NAMES, normalize=True,
                          title=f"{model_type.upper()} Confusion matrix (normalized)", label_prefix=model_type)

    summary = {"best_model_info": best_info, "all_results": all_results,
               "total_training_time_seconds": float(total_train_time), "num_models_trained": model_count}
    summary_path = os.path.join(RESULTS_FOLDER, results_filename)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    print(f"Saving summary to {summary_path}...")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {model_type.upper()} hyperparameter search summary to {summary_path}")

    return summary


def evaluate_model(model_type, texts = None, X = None, y = None, model = None, model_weights_path = None, model_params = None, model_prefix = '', force_cpu = False):
    """
    Evaluate model.

    Args:
        model_type: Type of model ("bert", "electra", or "roberta")
        texts: clean text data (raw texts, not tokenized)
        X: texts data that is already prepare for inference (tokenized). either texts or X should be given!
        y: labels
        model: transformer model
        model_weights_path: path for model weights file
        model_params: model hyperparameters
        model_prefix: description for the model (Optional for printing)
        force_cpu: Force model to run on CPU (required for quantized models)

    Returns:
        dict: Evaluation model and predictions (including metrics if there are true labels)
    """

    print("\n" + "=" * 70)
    print(f"Evaluating {model_prefix} {model_type.upper()} Model")
    print("=" * 70)

    # Recreate model architecture
    print(f"Recreating model architecture: {model_type}, {NUM_CLASSES} classes")

    # Extract parameters with defaults
    if model_params is None:
        model_params = {}
    dropout_rate = float(model_params.get("dropout_rate", 0.1))
    lr = float(model_params.get("lr", 2e-5))
    weight_decay = float(model_params.get("weight_decay", 0.0))
    batch_size = int(model_params.get("batch_size", 16))
    # Force CPU for quantized models (PyTorch qint8 limitation)
    if force_cpu:
        print(f"Quantized model: using CPU (PyTorch qint8 limitation)")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model, _, _ = build_model(model_type=model_type, dropout_rate=dropout_rate, lr=lr, weight_decay=weight_decay, freeze_transformer_weights=FREEZE_TRANSFORMER)
    else:
        # If model is provided, move it to the appropriate device
        model.to(device)
        model.eval()
        print(f"Model moved to {device}")

    # Load saved weights
    if model_weights_path is not None:
        print(f"Loading weights from {model_weights_path}...")
        model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
        model.eval()
        print("Model weights loaded and set to eval mode")

    # Prepare data with fixed max_length
    print(f"Preparing data with max_length={MAX_LENGTH}...")
    if X is None:
        if texts is not None:
            X = prepare_model_data(texts, y, model_type)
        else:
            raise ValueError("Missing input data (raw/tokenized)!")
    input_ids = torch.tensor(X["input_ids"], dtype=torch.long).to(device)
    attention_mask = torch.tensor(X["attention_mask"], dtype=torch.long).to(device)
    print(f"Data shape: {input_ids.shape}")

    # Make predictions
    print(f"Making predictions on {len(input_ids)} samples...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        all_probs = []
        num_batches = (len(input_ids) + batch_size - 1) // batch_size
        print(f"Processing {num_batches} batches with batch_size={batch_size}")
        for batch_idx, i in enumerate(range(0, len(input_ids), batch_size)):
            if batch_idx % 25 == 0:  # Print progress every 25 batches
                print(f"  Processing batch {batch_idx + 1}/{num_batches}...")
            batch_input_ids = input_ids[i : i + batch_size]
            batch_attention_mask = attention_mask[i : i + batch_size]

            outputs = model(batch_input_ids, batch_attention_mask)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

        preds_proba = np.vstack(all_probs)
        preds = preds_proba.argmax(axis=1)
        print(f"Predictions complete!")
        print(f"Predictions shape: {preds.shape}, probabilities shape: {preds_proba.shape}")
        print(f"Prediction range: [{preds.min()}, {preds.max()}], unique predictions: {len(np.unique(preds))}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    run_time = time.perf_counter() - start_time

    metrics = {}
    if y is not None:
        print(f"Calculating metrics...")
        # Calculate metrics
        metrics["accuracy"] = accuracy_score(y, preds)
        metrics["f1"] = f1_score(y, preds, average="macro")
        metrics["precision"] = precision_score(y, preds, average="macro", zero_division=0)
        metrics["recall"] = recall_score(y, preds, average="macro", zero_division=0)
        labels_bin = label_binarize(y, classes=range(NUM_CLASSES))
        metrics["auc_pr"] = average_precision_score(labels_bin, preds_proba, average="macro")

        # Print metrics
        print(f"\n{model_type.upper()} Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Macro F1:  {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Auc_PR:    {metrics['auc_pr']:.4f}")

    return {"model": model, "predictions": preds, "probabilities": preds_proba, "metrics": metrics, "run_time":run_time}

def freeze_transformer(model):
    """
    Freeze transformer params.

    Args:
        model: transformer model
    """
    for param in model.transformer.parameters():
        param.requires_grad = False

def quantize_model(model):
    """
    Apply dynamic quantization to linear layers.
    
    Note: Quantized models (qint8) only work on CPU in PyTorch.

    Args:
        model: transformer model

    Returns:
        quantized model (on CPU)
    """
    model.eval()
    # Move to CPU as quantized models don't work on GPU
    model.to(torch.device("cpu"))
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model


def prune_transformer_linear_layers(model, amount=0.3):
    """
    Globally prune `amount` of weights across ALL nn.Linear layers
    in the transformer encoder ONLY (excluding the classification head).

    Args:
        model: transformer model
        amount: amount of pruning in percentage

    Returns:
        pruned model
    """
    parameters_to_prune = []

    for name, module in model.transformer.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )

    return model

def finalize_pruned_transformer(model):
    """
    make the pruning permanent
    Args:
        model: transformer model

    Returns:
        fixed pruned model
    """
    for module in model.transformer.modules():
        if isinstance(module, nn.Linear):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass
    return model

def model_compressions(model_type, train_texts, y_train, val_texts, y_val, model_params, model_weights):
    """
        Run 2 compressions for a given model.

        Args:
            model_type: Type of model ("bert", "electra", or "roberta")
            train_texts: Training text data (raw texts, not tokenized)
            y_train: Training labels
            val_texts: Validation text data (raw texts, not tokenized)
            y_val: Validation labels
            model_params: Dictionary of model hyperparameters
            model_weights: Path to the saved model weights (.pt file)

        Returns:
            dict: results prune compression.
            dict: results quantize compression.
        """
    print(f"\nStarting 2 compressions for best {model_type.upper()} model")
    
    # Load checkpoint
    checkpoint = load_checkpoint()

    # Prepare data with fixed max_length
    print(f"Preparing data with max_length={MAX_LENGTH}...")
    X_train = prepare_model_data(train_texts, y_train, model_type)
    X_val = prepare_model_data(val_texts, y_val, model_type)
    summary = {}

    # Track total training time
    total_start_time = time.time()
    print(f"Compressions of best model started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))}")

    # Extract parameters
    dropout_rate = model_params["dropout_rate"]
    lr = model_params["lr"]
    batch_size = model_params["batch_size"]
    weight_decay = model_params["weight_decay"]

    # Restore best model
    print(f"Restore best model...")
    model, optimizer, device = build_model(model_type=model_type, dropout_rate=dropout_rate, lr=lr,weight_decay=weight_decay, freeze_transformer_weights=FREEZE_TRANSFORMER)
    model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
    model.eval()

    ### Compression 1 pruning

    print(f"Compression 1 - pruning")
    
    # Create param_str for file naming
    param_parts = []
    for k, v in sorted(model_params.items()):
        v_str = str(v)
        param_parts.append(f"{k}{v_str}")
    param_str = "_".join(param_parts)
    prune_model_path = f"{SAVE_MODELS_FOLDER}/prune_{model_type}_{param_str}.pt"
    
    # Check if pruned model already exists
    if os.path.exists(prune_model_path):
        print(f"[SKIPPED] Pruned model already exists: {prune_model_path}")
        print(f"Loading existing pruned model...")
        prune_model = copy.deepcopy(model)
        prune_model = prune_transformer_linear_layers(prune_model, 0.3)
        prune_model = finalize_pruned_transformer(prune_model)
        prune_model.load_state_dict(torch.load(prune_model_path, map_location=device, weights_only=True))
        prune_model.eval()
        # Run evaluation only (no training)
        print(f"Evaluating existing pruned model on validation set...")
        history = {"val_loss": [0], "val_accuracy": [0], "val_precision": [0], "val_recall": [0], 
                   "val_f1": [0], "val_auc_pr": [0], "train_run_time": [0], "val_run_time": [0]}
        # Quick evaluation
        prune_eval = evaluate_model(model_type=model_type, X=X_val, y=y_val, model=prune_model, 
                                     model_params=model_params, model_prefix='pruned')
        history["val_confusion_matrix"] = confusion_matrix(y_val, prune_eval["predictions"])
        history["val_classification_report"] = classification_report(y_val, prune_eval["predictions"], 
                                                                      target_names=CLASS_NAMES, zero_division=0)
        val_acc = prune_eval["metrics"]["accuracy"]
        val_precision = prune_eval["metrics"]["precision"]
        val_recall = prune_eval["metrics"]["recall"]
        val_f1 = prune_eval["metrics"]["f1"]
        val_auc_pr = prune_eval["metrics"]["auc_pr"]
        val_run_time = prune_eval["run_time"]
        train_run_time = 0
        epoch_run_time = val_run_time
    else:
        print(f"Training new pruned model...")
        prune_model = copy.deepcopy(model)
        prune_model = prune_transformer_linear_layers(prune_model, 0.3)

        # It is best practice to train the model again after pruning
        history = train_model(prune_model, optimizer, device, X_train, y_train, X_val, y_val, epochs=NUM_EPOCHS, batch_size=batch_size, patience=PATIENCE)
        
        # Restore prune model metric results on validation set
        idx = history["val_loss"].index(min(history["val_loss"]))
        val_acc = history["val_accuracy"][idx]
        val_precision = history["val_precision"][idx]
        val_recall = history["val_recall"][idx]
        val_f1 = history["val_f1"][idx]
        val_auc_pr = history["val_auc_pr"][idx]
        train_run_time = sum(history["train_run_time"]) / len(history["train_run_time"])
        val_run_time = sum(history["val_run_time"]) / len(history["val_run_time"])
        epoch_run_time = train_run_time + val_run_time

    print(f"Finished. val_accuracy: {val_acc:.4f}, val_precision: {val_precision:.4f}, val_recall: {val_recall:.4f}, val_f1: {val_f1:.4f}, val_auc_pr: {val_auc_pr:.4f}")
    if train_run_time > 0:
        print(f"Average epoch time: {epoch_run_time:.2f} seconds ({epoch_run_time / 60:.2f} minutes) - train: {train_run_time:.2f}s, val: {val_run_time:.2f}s")
    print(f"Inference time: {val_run_time:.2f} seconds ({val_run_time / 60:.2f} minutes)")

    os.makedirs(SAVE_MODELS_FOLDER, exist_ok=True)
    model_path = prune_model_path
    
    # Save model if it doesn't exist yet
    if not os.path.exists(model_path):
        print(f"Saving prune model to {model_path}...")
        prune_model = finalize_pruned_transformer(prune_model)
        torch.save(prune_model.state_dict(), model_path)
        print(f"Saved prune {model_type.upper()} model to {model_path}")

    # get size of pruned model
    total_params = sum(p.numel() for p in prune_model.parameters())
    model_size_mb = os.path.getsize(model_path) / (1024 ** 2)

    plot_history(history, title=f"prune_{model_type.upper()}_{param_str}")

    prune_result = {
        "params": model_params,
        "val_accuracy": float(val_acc),
        "val_precision": float(val_precision),
        "val_recall": float(val_recall),
        "val_f1": float(val_f1),
        "val_auc_pr": float(val_auc_pr),
        "model_path": model_path,
        "val_run_time_seconds": float(val_run_time),
        "model_size_mb": float(model_size_mb),
        "total_parameters": int(total_params),
    }

    total_train_time = time.time() - total_start_time

    # Generate classification report
    print(f"\nPrune {model_type.upper()} Classification Report:\n")
    print(history["val_classification_report"])

    # Save report
    report_path = os.path.join(RESULTS_FOLDER, f"prune_{model_type}_best_model_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f'Prune best {model_type.upper()} model path: {model_path}\n\n')
        f.write(json.dumps(prune_result, indent=2))
        f.write("\n\nValidation Classification Report:\n")
        f.write(history["val_classification_report"])
    print(f"Saved prune best {model_type.upper()} model report to {report_path}")

    # Plot confusion matrices
    plot_confusion_matrix(history["val_confusion_matrix"], classes=CLASS_NAMES, normalize=False,
                          title=f"{model_type.upper()} Confusion matrix (counts)", label_prefix="prune"+model_type)
    plot_confusion_matrix(history["val_confusion_matrix"], classes=CLASS_NAMES, normalize=True,
                          title=f"{model_type.upper()} Confusion matrix (normalized)", label_prefix="prune"+model_type)

    summary["prune"] = {"best_model_info": prune_result, "all_results": [prune_result],
               "total_training_time_seconds": float(total_train_time), "num_models_trained": 1}

    ### Compression 2 quantization

    print(f"Compression 2 - quantization")
    print(f"Note: Quantized models run on CPU (PyTorch qint8 limitation)")
    
    # Create quantize model path
    quantize_model_path = f"{SAVE_MODELS_FOLDER}/quantize_{model_type}_{param_str}.pt"
    
    # Check if quantized model already exists
    if os.path.exists(quantize_model_path):
        print(f"[SKIPPED] Quantized model already exists: {quantize_model_path}")
        print(f"Loading existing quantized model...")
        quantized_model = copy.deepcopy(model)
        quantized_model = quantize_model(quantized_model)
        quantized_model.load_state_dict(torch.load(quantize_model_path, map_location=torch.device("cpu"), weights_only=True))
        quantized_model.eval()
    else:
        print(f"Creating new quantized model...")
        quantized_model = copy.deepcopy(model)
        quantized_model = quantize_model(quantized_model)
        print(f"Saving quantize model to {quantize_model_path}...")
        torch.save(quantized_model.state_dict(), quantize_model_path)
        print(f"Saved quantize {model_type.upper()} model to {quantize_model_path}")

    # There is no need to retrain just make evaluation
    quantize_info = evaluate_model(model_type = model_type, X=X_val, y=y_val, model=quantized_model, model_params=model_params,model_prefix='quantize_best', force_cpu=True)
    model_path = quantize_model_path
    quantize_result = {
        "params": model_params,
        "val_accuracy": float(quantize_info["metrics"]["accuracy"]),
        "val_precision": float(quantize_info["metrics"]["precision"]),
        "val_recall": float(quantize_info["metrics"]["recall"]),
        "val_f1": float(quantize_info["metrics"]["f1"]),
        "val_auc_pr": float(quantize_info["metrics"]["auc_pr"]),
        "model_path": model_path,
        "val_run_time_seconds": float(quantize_info["run_time"]),
        "model_size_mb": float(os.path.getsize(model_path) / (1024 ** 2)),
        "total_parameters": int(sum(p.numel() for p in quantized_model.parameters())),
    }
    quantize_cm = confusion_matrix(y_val, quantize_info["predictions"])
    quantize_classification_report = classification_report(y_val, quantize_info["predictions"],target_names=CLASS_NAMES, zero_division=0)

    # Generate classification report
    print(f"\nQuantize {model_type.upper()} Classification Report:\n")
    print(quantize_classification_report)

    # Save report
    report_path = os.path.join(RESULTS_FOLDER, f"quantize_{model_type}_best_model_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f'Quantize best {model_type.upper()} model path: {model_path}\n\n')
        f.write(json.dumps(quantize_result, indent=2))
        f.write("\n\nValidation Classification Report:\n")
        f.write(quantize_classification_report)
    print(f"Saved quantize best {model_type.upper()} model report to {report_path}")

    # Plot confusion matrices
    plot_confusion_matrix(quantize_cm, classes=CLASS_NAMES, normalize=False,
                          title=f"{model_type.upper()} Confusion matrix (counts)", label_prefix="quantize" + model_type)
    plot_confusion_matrix(quantize_cm, classes=CLASS_NAMES, normalize=True,
                          title=f"{model_type.upper()} Confusion matrix (normalized)",label_prefix="quantize" + model_type)

    summary["quantize"] = {"best_model_info": quantize_result, "all_results": [quantize_result],
                           "total_training_time_seconds": float(0), "num_models_trained": 1}

    summary_path = os.path.join(RESULTS_FOLDER, f"compressions_results_{model_type}.json")
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    print(f"Saving compressions summary to {summary_path}...")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {model_type.upper()} compressions summary to {summary_path}")

    return summary["prune"], summary["quantize"]

def run_inference(weights, csv):
    """
    Run inference on test data using a trained model and make predictions to csv file.

    Args:
        weights: Path to the saved model weights (.pt file)
        csv: Path to CSV file containing texts to classify

    """
    print("\n" + "=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70)

    # Infer model type from weights filename
    model_type = None
    weights_filename = os.path.basename(weights).lower()
    if "bert" in weights_filename:
        model_type = "bert"
    elif "electra" in weights_filename:
        model_type = "electra"
    elif "roberta" in weights_filename:
        model_type = "roberta"

    if model_type is None:
        raise ValueError(f"Cannot extract model type from weights path file name: {weights}. Please don't change the name of the given weights file!")

    print(f"Model type: {model_type.upper()}")
    print(f"Weights file exists: {os.path.exists(weights)}")
    print(f"Input CSV exists: {os.path.exists(csv)}")
    print(f"Parameters: num_classes={NUM_CLASSES}, max_length={MAX_LENGTH}")

    # Load CSV data
    print(f"\nLoading CSV data...")
    df = pd.read_csv(csv)
    test_texts, test_labels = load_data(df, "Test Data")
    model_params = {}

    # Extract hyperparameters from filename
    try:
        weights_filename = os.path.basename(weights)
        model_params["batch_size"] = int(weights_filename.split("batch_size")[1].split("_")[0])
        model_params["dropout_rate"] = float(weights_filename.split("dropout_rate")[1].split("_")[0])
        model_params["lr"] = float(weights_filename.split("lr")[1].split("_")[0])
        model_params["weight_decay"] = float(weights_filename.split("weight_decay")[1].split(".")[0])
    except:
        raise ValueError(f"Cannot extract model hyper params from weights path file name: {weights}. Please don't change the name of the given weights file!")

    print(f'Extracted hyperparameters from filename: batch_size={model_params["batch_size"]}, dropout_rate={model_params["dropout_rate"]}, lr={model_params["lr"]}, weight_decay={model_params["weight_decay"]}')

    results = evaluate_model(model_type = model_type, texts=test_texts, y=test_labels, model_weights_path=weights, model_params=model_params,model_prefix='best')

    # Create output dataframe
    output_df = df.copy()
    output_df["predicted_label"] = results["predictions"]
    output_path = os.path.join(os.path.dirname(csv), 'predictions.csv')
    output_df.to_csv(output_path, index=False)
    print(f"Saved test predictions to {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    # Set up output capture to both console and file
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    # Use different output files for training vs inference mode
    if RUN_INFERENCE_ONLY:
        output_file_path = os.path.join(RESULTS_FOLDER, "inference_output.txt")
    else:
        output_file_path = os.path.join(RESULTS_FOLDER, "train_and_val_output.txt")
    tee_output = TeeOutput(output_file_path)
    original_stdout = sys.stdout
    sys.stdout = tee_output
    
    try:
        # Set seed for random variables
        set_seed(42)

        print("\n" + "=" * 70)
        print("Starting Emotion Detection Pipeline")
        print("=" * 70)
        print(f"Output is being saved to: {output_file_path}")
        print(f"Configuration:")
        print(f"  TRAIN_FILE: {TRAIN_FILE}")
        print(f"  VALIDATION_FILE: {VALIDATION_FILE}")
        print(f"  TEST_FILE: {TEST_FILE}")
        print(f"  RESULTS_FOLDER: {RESULTS_FOLDER}")
        print(f"  SAVE_MODELS_FOLDER: {SAVE_MODELS_FOLDER}")
        print(f"  MAX_LENGTH: {MAX_LENGTH}")
        print(f"  PARAM_GRID: {PARAM_GRID}")

        if RUN_INFERENCE_ONLY:
            run_inference(BEST_MODEL_WEIGHTS,TEST_FILE)

        else:
            # Load checkpoint to check progress
            main_checkpoint = load_checkpoint()
            
            # Load data
            train_texts, train_labels = load_data(TRAIN_FILE, "Training Data")
            val_texts, val_labels = load_data(VALIDATION_FILE, "Validation Data")

            print(f"\nNumber of classes: {NUM_CLASSES}")
            print(f"Class names: {CLASS_NAMES}")

            # ============================================================================
            # BERT MODEL TRAINING
            # ============================================================================
            # ============================================================================
            # BERT MODEL TRAINING
            # ============================================================================
            print("\n" + "=" * 70)
            print("BERT HYPERPARAMETER SEARCH")
            print("=" * 70)

            # Run BERT hyperparameter search
            print(f"\nStarting BERT hyperparameter search...")
            hp_summary_bert = hyperparameter_search(
                model_type="bert",
                train_texts=train_texts,
                y_train=train_labels,
                val_texts=val_texts,
                y_val=val_labels,
                param_grid=PARAM_GRID,
                max_models=None,  # Will train all combinations
                results_filename="hp_results_bert.json",
            )

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
                param_grid=PARAM_GRID,
                max_models=None,  # Will train all combinations
                results_filename="hp_results_electra.json",
            )

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
                param_grid=PARAM_GRID,
                max_models=None,  # Will train all combinations
                results_filename="hp_results_roberta.json",
            )

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
                        "val_run_time_seconds": bert_info.get("val_run_time_seconds", 0),
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
                        "val_run_time_seconds": electra_info.get("val_run_time_seconds", 0),
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
                        "val_run_time_seconds": roberta_info.get("val_run_time_seconds", 0),
                        "model_size_mb": roberta_info.get("model_size_mb", 0),
                        "total_parameters": roberta_info.get("total_parameters", 0),
                        "total_training_time": hp_summary_roberta.get("total_training_time_seconds", 0),
                    }
                )

            # Sort by validation f1 (descending)
            comparison_results.sort(key=lambda x: x["val_f1"], reverse=True)

            # Get best model info for making compressions
            best_result = comparison_results[0]
            best_model_name = best_result["model"]
            best_model_type = best_model_name.lower()
            print(f"\nUsing {best_model_name} model (val_f1: {best_result['val_f1']:.4f}) for compressions")
            print(f"Best model path: {best_result['model_path']}")

            prune_summary, quantize_summary = model_compressions(best_model_type, train_texts, train_labels, val_texts, val_labels, best_result['best_params'], best_result['model_path'])

            # Add compressions results to comparison results

            prune_info = prune_summary["best_model_info"]
            comparison_results.append(
                {
                    "model": f"Prune {best_model_name}",
                    "best_params": prune_info["params"],
                    "val_accuracy": prune_info.get("val_accuracy", 0),
                    "val_precision": prune_info.get("val_precision", 0),
                    "val_recall": prune_info.get("val_recall", 0),
                    "val_f1": prune_info.get("val_f1", 0),
                    "val_auc_pr": prune_info.get("val_auc_pr", 0),
                    "model_path": prune_info["model_path"],
                    "val_run_time_seconds": prune_info.get("val_run_time_seconds", 0),
                    "model_size_mb": prune_info.get("model_size_mb", 0),
                    "total_parameters": prune_info.get("total_parameters", 0),
                    "total_training_time": prune_summary.get("total_training_time_seconds", 0),
                }
            )

            quantize_info = quantize_summary["best_model_info"]
            comparison_results.append(
                {
                    "model": f"Quantize {best_model_name}",
                    "best_params": quantize_info["params"],
                    "val_accuracy": quantize_info.get("val_accuracy", 0),
                    "val_precision": quantize_info.get("val_precision", 0),
                    "val_recall": quantize_info.get("val_recall", 0),
                    "val_f1": quantize_info.get("val_f1", 0),
                    "val_auc_pr": quantize_info.get("val_auc_pr", 0),
                    "model_path": quantize_info["model_path"],
                    "val_run_time_seconds": quantize_info.get("val_run_time_seconds", 0),
                    "model_size_mb": quantize_info.get("model_size_mb", 0),
                    "total_parameters": quantize_info.get("total_parameters", 0),
                    "total_training_time": quantize_summary.get("total_training_time_seconds", 0),
                }
            )

            # Sort again by validation f1 (descending)
            comparison_results.sort(key=lambda x: x["val_f1"], reverse=True)

            # Print comprehensive comparison table
            print("\n" + "=" * 120)
            print(" " * 40 + "COMPREHENSIVE MODEL COMPARISON")
            print("=" * 120)
            print(f"{'Model':<12} {'Val F1':<10} {'Parameters':<15} {'Size (MB)':<12} {'Val Time':<15} {'Total Time':<15}")
            print("-" * 120)
            for result in comparison_results:
                model_name = result["model"]
                val_f1 = f"{result['val_f1']:.4f}"
                params = f"{result['total_parameters']:,}"
                size_mb = f"{result['model_size_mb']:.2f}"
                val_time = f"{result['val_run_time_seconds']:.1f}s"
                total_time = f"{result['total_training_time'] / 60:.1f}min"
                print(f"{model_name:<12} {val_f1:<10} {params:<15} {size_mb:<12} {val_time:<15} {total_time:<15}")
            print("=" * 120)

            print("\nDetailed Model Rankings (by validation f1):")
            print("-" * 70)
            for i, result in enumerate(comparison_results, 1):
                print(f"{i}. {result['model']}: {result['val_f1']:.4f}")
                print(f"   Metrics - Accuracy: {result.get('val_accuracy', 0):.4f}, Precision: {result.get('val_precision', 0):.4f}, Recall: {result.get('val_recall', 0):.4f}, F1: {result.get('val_f1', 0):.4f}, AUC-PR: {result.get('val_auc_pr', 0):.4f}")
                print(f"   Best params: {result['best_params']}")
                print(f"   Model size: {result['model_size_mb']:.2f} MB ({result['total_parameters']:,} parameters)")
                print(f"   Inference time: {result['val_run_time_seconds']:.2f} seconds ({result['val_run_time_seconds'] / 60:.2f} minutes)")
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
                        "Inference Time (seconds)": r["val_run_time_seconds"],
                        "Inference Time (minutes)": r["val_run_time_seconds"] / 60,
                        "Total Search Time (minutes)": r["total_training_time"] / 60,
                        "Dropout Rate": r["best_params"].get("dropout_rate", 0),
                        "Learning Rate": r["best_params"].get("lr", 0),
                        "Batch Size": r["best_params"].get("batch_size", 0),
                        "Weight Decay": r["best_params"].get("weight_decay", 0),
                    }
                    for r in comparison_results
                ]
            )
            comparison_csv_path = os.path.join(RESULTS_FOLDER, "model_comparison.csv")
            comparison_df.to_csv(comparison_csv_path, index=False)
            print(f"Saved model comparison table to {comparison_csv_path}")

            print("\n" + "=" * 70)
            print("Pipeline Complete!")
            print("=" * 70)
            print(f"Results saved to: {RESULTS_FOLDER}")
            print(f"Models saved to: {SAVE_MODELS_FOLDER}")
            end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            print(f"Pipeline finished at: {end_time}")
    
    finally:
        # Restore original stdout and close log file
        sys.stdout = original_stdout
        tee_output.close()
        print(f"Pipeline output saved to: {output_file_path}")
