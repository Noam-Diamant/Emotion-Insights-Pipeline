"""
Data Preparation and GloVe Embedding Setup for Emotion Detection
This script handles text preprocessing, tokenization, and GloVe embedding matrix creation.
"""

import numpy as np
import pandas as pd
import re
import os
import pickle
import zipfile
import requests
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import itertools
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
MAX_SEQUENCE_LENGTH = 66  # Maximum length in dataset - no truncation
EMBEDDING_DIM = 100
GLOVE_FILE = './data/glove.6B.100d.txt'
TRAIN_FILE = './data/train.csv'
VALIDATION_FILE = './data/validation.csv'
TEST_FILE = './data/test.csv'
RESULTS_fOLDER = './results'
SAVE_MODELS_FOLDER = './hp_models'
BEST_MODEL_KIND = 'gru' #  'lstm' or 'gru'

###  params for gru and lstm networks. the code runs all the possible combinations of params for each network.
### (there is a threshold for max combination in hyperparameter_search function [max_models])

PARAM_GRID_LSTM = {
'hidden_units': [64],
'dropout_rate': [0.3],
'trainable_embeddings': [True],
'lr': [5e-4]
}

PARAM_GRID_GRU = {
'hidden_units': [64],
'dropout_rate': [0.3],
'trainable_embeddings': [True],
'lr': [8e-3]
}

def load_data(data_file, dataset_name="Data"):
    """
    Load data from CSV file.

    Args:
        data_file: Path to the CSV file
        dataset_name: Name of the dataset (for logging)

    Returns:
        tuple: (texts, labels)
    """
    print("=" * 70)
    print(f"Loading {dataset_name}")
    print("=" * 70)

    # Load the data
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} samples from {data_file}")
    print(f"Columns: {df.columns.tolist()}")

    # Extract text and labels
    text_column = 'text' if 'text' in df.columns else df.columns[0]
    label_column = 'label' if 'label' in df.columns else None

    texts = df[text_column].values
    if label_column is None: # can be in test.csv
        print(f"Number of samples: {len(texts)}")
        return texts, None

    labels = df[label_column].values
    print(f"Number of samples: {len(texts)}")
    print(f"Number of unique emotions: {len(np.unique(labels))}")
    print(f"Emotion distribution: {np.bincount(labels)}")

    return texts, labels

def preprocess_text(text):
    """
    Clean and preprocess text data.

    Args:
        text: Input text string

    Returns:
        Cleaned and preprocessed text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove HTML tags and attributes
    text = re.sub(r'<[^>]+>', '', text)

    # Remove HTML entities/artifacts
    text = re.sub(r'&\w+;', '', text)

    # Remove common HTML/web artifacts
    html_artifacts = ['href', 'nofollow', 'permalink', 'pagetitle', 'rel', 'target']
    for artifact in html_artifacts:
        text = text.replace(artifact, '')

    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # Fix common contractions (before removing apostrophes)
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am"
    }
    for contraction, replacement in contractions.items():
        text = text.replace(contraction, replacement)

    # Remove possessives (e.g., "john's" -> "john")
    text = re.sub(r"'s\b", "", text)

    # Remove remaining apostrophes
    text = text.replace("'", "")

    # Fix malformed contractions (without apostrophes - common in social media)
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
        "were": "we are",  # Only replace standalone "were" carefully
        "youve": "you have",
        "theyve": "they have",
        "weve": "we have",
        "ive": "i have",
        "youll": "you will",
        "theyll": "they will",
        "well": "we will",  # Careful - could be the adverb "well"
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
        "theres": "there is"
    }

    # Use word boundaries to avoid replacing parts of words
    for contraction, replacement in malformed_contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', replacement, text)

    # Remove special characters and digits, keep only letters and basic punctuation
    text = re.sub(r'[^a-z\s.,!?]', '', text)

    # Remove repeated punctuation
    text = re.sub(r'([.,!?])\1+', r'\1', text)

    # Remove standalone punctuation
    text = re.sub(r'\s+[.,!?]+\s+', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_texts(texts, dataset_name="Data"):
    """
    Apply preprocessing to all texts.

    Args:
        texts: Array of text strings
        dataset_name: Name of the dataset (for logging)

    Returns:
        List of preprocessed texts
    """
    print(f"\nPreprocessing {dataset_name}...")
    processed_texts = [preprocess_text(str(text)) for text in texts]

    print(f"Example original text: {texts[0]}")
    print(f"Example processed text: {processed_texts[0]}")

    return processed_texts

def fit_tokenizer(processed_texts):
    """
    Fit tokenizer on training data to build vocabulary.

    IMPORTANT: Only call this on TRAINING data, not validation/test data!

    Args:
        processed_texts: List of preprocessed text strings from TRAINING set

    Returns:
        tuple: (tokenizer, vocab_size)
    """
    print("\n" + "=" * 70)
    print("STEP 3: Fitting Tokenizer (Training Data Only)")
    print("=" * 70)

    # Initialize tokenizer
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(processed_texts)

    # Get vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of unique words: {len(tokenizer.word_index)}")

    return tokenizer, vocab_size

def transform_and_pad(processed_texts, tokenizer, max_sequence_length, dataset_name="Data"):
    """
    Transform texts to sequences using fitted tokenizer and pad them.

    Args:
        processed_texts: List of preprocessed text strings
        tokenizer: Fitted Keras Tokenizer
        max_sequence_length: Maximum sequence length for padding
        dataset_name: Name of the dataset (for logging)

    Returns:
        numpy.ndarray: Padded sequences
    """
    print("\n" + "=" * 70)
    print(f"STEP 4: Transforming and Padding {dataset_name}")
    print("=" * 70)

    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(processed_texts)
    print(f"Example sequence (first 10 tokens): {sequences[0][:10]}")

    # Pad sequences
    padded_sequences = pad_sequences(
        sequences,
        maxlen=max_sequence_length,
        padding='post',
        truncating='post'
    )

    print(f"Padded sequences shape: {padded_sequences.shape}")
    print(f"Example padded sequence: {padded_sequences[0]}")

    return padded_sequences

def ensure_glove_exists(glove_path="./data/glove.6B.100d.txt"):
    """
    Ensure GloVe embeddings (100d) exist locally.
    If not found, automatically download and extract glove.6B.zip from Stanford.

    Args:
        glove_path (str): Expected path to glove.6B.100d.txt

    Returns:
        str: Path to glove.6B.100d.txt (after verification or download)
    """

    if os.path.exists(glove_path):
        print("GloVe file already exists:", glove_path)
        return glove_path

    # The zip file we need to download
    zip_path = "./data/glove.6B.zip"
    os.makedirs("./data", exist_ok=True)

    print("GloVe file not found. Downloading from Stanford NLP (~860 MB)...")

    url = "http://nlp.stanford.edu/data/glove.6B.zip"

    # Stream download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(zip_path, "wb") as f, tqdm(
            desc="Downloading glove.6B.zip",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

    print("Download complete. Extracting...")

    # Extract only the needed file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if "glove.6B.100d.txt" not in zip_ref.namelist():
            raise RuntimeError("glove.6B.100d.txt not found inside downloaded archive!")

        zip_ref.extract("glove.6B.100d.txt", "./data")

    print("Extraction complete.")

    # Optional: delete ZIP to save space
    os.remove(zip_path)

    print("Ready:", glove_path)
    return glove_path

def load_glove_embeddings(glove_file):
    """
    Load pre-trained GloVe word embeddings.

    Args:
        glove_file: Path to the GloVe embeddings file

    Returns:
        dict: Dictionary mapping words to embedding vectors
    """

    print(f"Loading GloVe vectors from {glove_file}...")

    embeddings_index = {}

    try:
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print(f"Loaded {len(embeddings_index)} word vectors from GloVe")

    except FileNotFoundError:
        print(f"ERROR: GloVe file '{glove_file}' not found.")
        print("\nTo download GloVe embeddings, run:")
        print("  python download_glove.py")
        print("\nOr download manually from: https://nlp.stanford.edu/projects/glove/")
        print("Extract 'glove.6B.100d.txt' to the ./data/ directory")
        raise

    return embeddings_index

def create_embedding_matrix(tokenizer, embeddings_index, vocab_size, embedding_dim):
    """
    Create embedding matrix from GloVe vectors for the dataset vocabulary.

    Args:
        tokenizer: Fitted Keras Tokenizer
        embeddings_index: Dictionary of GloVe word vectors
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of embeddings

    Returns:
        numpy.ndarray: Embedding matrix of shape (vocab_size, embedding_dim)
    """
    print("\n" + "=" * 70)
    print("STEP 6: Creating Embedding Matrix")
    print("=" * 70)

    # Initialize embedding matrix with zeros
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # Counter for words found in GloVe
    words_found = 0
    words_not_found = 0

    # Fill embedding matrix
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Word found in GloVe
            embedding_matrix[i] = embedding_vector
            words_found += 1
        else:
            # Word not found in GloVe - leave as zeros
            words_not_found += 1

    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    print(f"Words found in GloVe: {words_found}")
    print(f"Words not found in GloVe (initialized to zero): {words_not_found}")
    print(f"Coverage: {words_found / len(tokenizer.word_index) * 100:.2f}%")

    return embedding_matrix

def save_processed_data(train_sequences, train_labels, val_sequences, val_labels,
                       embedding_matrix, tokenizer, output_dir='./processed_data'):
    """
    Save processed data to disk for later use.

    Args:
        train_sequences: Training padded sequences
        train_labels: Training labels
        val_sequences: Validation padded sequences
        val_labels: Validation labels
        embedding_matrix: Embedding matrix
        tokenizer: Fitted tokenizer
        output_dir: Directory to save the files (default: './processed_data')
    """
    print("\nSaving processed data...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save training data
    np.save(f'{output_dir}/X_train.npy', train_sequences)
    np.save(f'{output_dir}/y_train.npy', train_labels)

    # Save validation data
    np.save(f'{output_dir}/X_val.npy', val_sequences)
    np.save(f'{output_dir}/y_val.npy', val_labels)

    # Save embedding matrix
    np.save(f'{output_dir}/embedding_matrix.npy', embedding_matrix)

    # Save tokenizer
    with open(f'{output_dir}/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    print("Saved:")
    print(f"  - {output_dir}/X_train.npy")
    print(f"  - {output_dir}/y_train.npy")
    print(f"  - {output_dir}/X_val.npy")
    print(f"  - {output_dir}/y_val.npy")
    print(f"  - {output_dir}/embedding_matrix.npy")
    print(f"  - {output_dir}/tokenizer.pkl")

def main_data_preparation():
    """
    Main function to orchestrate the data preparation pipeline.

    Process:
    1. Load train and validation data
    2. Preprocess both datasets
    3. Fit tokenizer on TRAINING data only
    4. Transform both datasets with the same tokenizer
    5. Create embedding matrix from training vocabulary
    6. Save all processed data

    Returns:
        dict: Dictionary containing all processed data and parameters
    """
    # ========================================================================
    # STEP 1: Load Training and Validation Data
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Loading Datasets")
    print("=" * 70)

    train_texts, train_labels = load_data(TRAIN_FILE, "Training Data")
    val_texts, val_labels = load_data(VALIDATION_FILE, "Validation Data")

    # ========================================================================
    # STEP 2: Preprocess Texts
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Preprocessing Texts")
    print("=" * 70)

    train_processed = preprocess_texts(train_texts, "Training Data")
    val_processed = preprocess_texts(val_texts, "Validation Data")

    # ========================================================================
    # STEP 3: Fit Tokenizer on Training Data ONLY
    # ========================================================================
    tokenizer, vocab_size = fit_tokenizer(train_processed)

    # ========================================================================
    # STEP 4: Transform Both Datasets with Same Tokenizer
    # ========================================================================
    train_sequences = transform_and_pad(
        train_processed, tokenizer, MAX_SEQUENCE_LENGTH, "Training Data"
    )
    val_sequences = transform_and_pad(
        val_processed, tokenizer, MAX_SEQUENCE_LENGTH, "Validation Data"
    )

    # ========================================================================
    # STEP 5: Load GloVe Embeddings
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Loading GloVe Embeddings")
    print("=" * 70)

    glove_file_path = ensure_glove_exists(GLOVE_FILE)
    embeddings_index = load_glove_embeddings(glove_file_path)

    # ========================================================================
    # STEP 6: Create Embedding Matrix
    # ========================================================================
    embedding_matrix = create_embedding_matrix(
        tokenizer, embeddings_index, vocab_size, EMBEDDING_DIM
    )

    # ========================================================================
    # Save Processed Data for debug
    # ========================================================================
    # save_processed_data(
    #     train_sequences, train_labels, val_sequences, val_labels,
    #     embedding_matrix, tokenizer
    # )

    # ========================================================================
    # Return All Data
    # ========================================================================
    return {
        'X_train': train_sequences,
        'y_train': train_labels,
        'X_val': val_sequences,
        'y_val': val_labels,
        'embedding_matrix': embedding_matrix,
        'tokenizer': tokenizer,
        'vocab_size': vocab_size,
        'embedding_dim': EMBEDDING_DIM,
        'max_sequence_length': MAX_SEQUENCE_LENGTH,
        'num_classes': len(np.unique(train_labels))
    }

def plot_history(history, title="Training History"):
    """
    Plot training/validation loss and accuracy from a Keras History object.

    Args:
        history (tensorflow.keras.callbacks.History): History object returned by model.fit().
        title (str): Title for the overall figure.
    Returns:
        None (shows and saves a PNG file named '{title}.png' in the current working directory)
    """
    acc = history.history.get('accuracy', history.history.get('acc'))
    val_acc = history.history.get('val_accuracy', history.history.get('val_acc'))
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'r--', label='Validation loss')
    plt.title(f'{title} — Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'b-', label='Training acc')
    plt.plot(epochs, val_acc, 'r--', label='Validation acc')
    plt.title(f'{title} — Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    fname = f"{title.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    os.makedirs(RESULTS_fOLDER, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_fOLDER,fname))
    print(f"Saved training plot: {fname}")
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,label_prefix=''):
    """
    Print and plot the confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix (n_classes x n_classes).
        classes (list): List of class names in index order.
        normalize (bool): If True, normalize the confusion matrix rows to percentages.
        title (str): Plot title.
        cmap: Matplotlib colormap.
        label_prefix: prefix used to name saved files for clarity.

    Returns:
        None (shows and saves a PNG file named '{title}.png').
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fname = f"{title.replace(' ', '_').lower()}_{label_prefix}.png"
    os.makedirs(RESULTS_fOLDER, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_fOLDER,fname))
    print(f"Saved confusion matrix plot: {fname}")
    plt.show()

def build_rnn_model(kind, vocab_size, embedding_dim, embedding_matrix, max_len, num_classes,
                    hidden_units=128, dropout_rate=0.5, trainable_embeddings=False, lr=1e-3):
    """
    Build and compile either an LSTM or GRU model with configurable hyperparameters.

    Args:
        kind (str): "lstm" or "gru".
        vocab_size (int): Tokenizer vocabulary size.
        embedding_dim (int): Embedding dimensionality.
        embedding_matrix (np.ndarray): Pretrained embedding matrix.
        max_len (int): Input length for sequences.
        num_classes (int): Number of classes.
        hidden_units (int): Hidden units in the recurrent layer.
        dropout_rate (float): Dropout after recurrent layer.
        trainable_embeddings (bool): Whether to allow embedding weights to be trainable.
        lr (float): Learning rate for Adam optimizer.

    Returns:
        tensorflow.keras.Model: Compiled model.
    """
    model = Sequential()
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=trainable_embeddings
    ))
    if kind.lower() == 'lstm':
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=False)))
    else:
        model.add(Bidirectional(GRU(hidden_units, return_sequences=False)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(int(hidden_units/2), activation='relu'))
    model.add(Dropout(max(0.2, dropout_rate/2)))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def hyperparameter_search(kind, X_train, y_train, X_val, y_val, vocab_size, embedding_dim,
                          embedding_matrix, max_len, num_classes, param_grid, max_models=None ,results_filename='hp_results.json'):
    """
    Run a grid search (simple loop) over provided hyperparameters to find best model by val_accuracy.

    Args:
        kind (str): 'lstm' or 'gru' specifying model type.
        X_train, y_train, X_val, y_val: Train/Val arrays.
        vocab_size, embedding_dim, embedding_matrix, max_len, num_classes: data parameters.
        param_grid (dict): Dictionary mapping hyperparam names to lists of values. Keys supported:
                           'hidden_units', 'dropout_rate', 'trainable_embeddings', 'lr'.
        max_models (int or None): Stop after training this many models (helps limit runtime).
        results_filename (str): Name of JSON file to save search results summary.

    Returns:
        dict: Summary containing 'best_model_info' (params + val_accuracy + model_path) and 'all_results' list.
    """
    combos = []
    for hu in param_grid.get('hidden_units', [128]):
        for dr in param_grid.get('dropout_rate', [0.5]):
            for te in param_grid.get('trainable_embeddings', [False]):
                for lr in param_grid.get('lr', [1e-3]):
                    combos.append({'hidden_units': hu, 'dropout_rate': dr, 'trainable_embeddings': te, 'lr': lr})

    print(f"\nHyperparameter search will run {len(combos)} combos (max_models={max_models})")
    all_results = []
    best_val_acc = -1.0
    best_info = None
    model_count = 0

    for i, params in enumerate(combos):
        if (max_models is not None) and (model_count >= max_models):
            break
        model_count += 1
        print("\n" + "="*60)
        print(f"Model {model_count}/{len(combos)} - params: {params}")
        model = build_rnn_model(kind=kind,
                                vocab_size=vocab_size,
                                embedding_dim=embedding_dim,
                                embedding_matrix=embedding_matrix,
                                max_len=max_len,
                                num_classes=num_classes,
                                hidden_units=int(params['hidden_units']),
                                dropout_rate=float(params['dropout_rate']),
                                trainable_embeddings=bool(params['trainable_embeddings']),
                                lr=float(params['lr']))

        # Train with early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=20, batch_size=64, callbacks=[early_stop], verbose=1)

        val_acc = max(history.history.get('val_accuracy', [0.0]))
        print(f"Finished training. Best val_accuracy: {val_acc:.4f}")

        # Save model to disk
        os.makedirs(SAVE_MODELS_FOLDER, exist_ok=True)
        model_path = f"{SAVE_MODELS_FOLDER}/{kind}_hu{params['hidden_units']}_dr{params['dropout_rate']}_te{int(params['trainable_embeddings'])}_lr{params['lr']}.h5"
        model.save(model_path)
        print(f"Saved model to {model_path}")

        # Save history plot for this model
        plot_history(history, title=f"{kind.upper()}_hu{params['hidden_units']}_dr{params['dropout_rate']}_te{int(params['trainable_embeddings'])}_lr{params['lr']}")

        result = {
            'params': params,
            'val_accuracy': float(val_acc),
            'model_path': model_path,
            'history_keys': list(history.history.keys())
        }
        all_results.append(result)

        # Update best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_info = result

    summary = {'best_model_info': best_info, 'all_results': all_results}
    summary_path = os.path.join(RESULTS_fOLDER,results_filename)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved hyperparameter search summary to {summary_path}")

    return summary

def evaluate_on_validation(model, X_val, y_val, label_prefix, class_names=None):
    """
    Evaluate a trained model on validation set and print common classification metrics.

    Args:
        model (tensorflow.keras.Model): Trained Keras model.
        X_val (np.ndarray): Validation sequences.
        y_val (np.ndarray): True validation labels.
        label_prefix: prefix used to name saved files for clarity.
        class_names (list or None): Optional list of class names (len == num_classes).
                                    If None, indices [0..num_classes-1] are used as labels.


    Returns:
        dict: Dictionary with metrics: accuracy, macro_f1, precision_macro, recall_macro;
              and 'report' containing the sklearn classification report string.
    """
    if class_names is None:
        class_names = [str(i) for i in range(model.output_shape[-1])]

    preds_proba = model.predict(X_val, verbose=0)
    preds = preds_proba.argmax(axis=1)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    precision = precision_score(y_val, preds, average='macro', zero_division=0)
    recall = recall_score(y_val, preds, average='macro', zero_division=0)

    print("\nValidation Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

    # Confusion matrix and classification report
    cm = confusion_matrix(y_val, preds)
    print("\nClassification Report:\n")
    report = classification_report(y_val, preds, target_names=class_names, zero_division=0)
    print(report)

    # Save and plot confusion matrix
    plot_confusion_matrix(cm, classes=class_names, normalize=False, title='Confusion matrix (counts)',label_prefix=label_prefix)
    plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Confusion matrix (normalized)',label_prefix=label_prefix)

    metrics = {
        'accuracy': acc,
        'macro_f1': f1,
        'precision_macro': precision,
        'recall_macro': recall,
        'report': report,
        'confusion_matrix': cm
    }
    return metrics

def evaluate_best_hp_and_save(hp_summary, X_val, y_val, class_names, label_prefix):
    """
    Load best model from a hyperparameter search summary, evaluate on validation, and save metrics.

    Args:
        hp_summary (dict): Summary dict returned by hyperparameter_search (contains best_model_info).
        X_val, y_val: Validation arrays.
        class_names: list of class labels.
        label_prefix: prefix used to name saved files for clarity.

    Returns:
        dict: Metrics for the best model or None if not available.
    """
    if not hp_summary or not hp_summary.get('best_model_info'):
        print("No best model info found in hyperparameter summary.")
        return None

    best_path = hp_summary['best_model_info']['model_path']
    print(f"\nLoading best model from {best_path} ...")
    best_model = load_model(best_path)
    metrics = evaluate_on_validation(best_model, X_val, y_val, label_prefix, class_names=class_names)

    # Save report to disk
    report_path = os.path.join(RESULTS_fOLDER, f'{label_prefix}_best_model_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Best model path: {best_path}\n\n")
        f.write(json.dumps(hp_summary['best_model_info'], indent=2))
        f.write("\n\nValidation Classification Report:\n")
        f.write(metrics['report'])

    print(f"Saved best-model report to {report_path}")
    return metrics

def predict_test_and_save(model, tokenizer, max_len=66):
    """
    Run prediction on test.csv, attach predicted labels and probabilities and save CSV.

    Args:
        model (tensorflow.keras.Model): Trained Keras model for inference.
        tokenizer: Fitted tokenizer for converting texts.
        max_len (int): Sequence length for padding/truncation.

    Returns:
        pd.DataFrame or None: DataFrame with predictions (if test exists), else None.
    """
    if not os.path.exists(TEST_FILE):
        print(f"Test file was not found at {TEST_FILE}. Skipping test inference.")
    else:
        test_df = pd.read_csv(TEST_FILE)
        test_texts, test_labels = load_data(TEST_FILE, "Test Data")
        test_processed = preprocess_texts(test_texts, "Test Data")
        test_sequences = transform_and_pad(test_processed, tokenizer, MAX_SEQUENCE_LENGTH, "Test Data")

        preds_proba = model.predict(test_sequences, verbose=0)
        preds = preds_proba.argmax(axis=1)
        probs = preds_proba.max(axis=1)
        if test_labels is not None: # maybe we got the true labels of test and we can compare to predictions
            acc = accuracy_score(test_labels, preds)
            print (f"Got accuracy of {acc} on test dataset!")

        out_df = test_df.copy().reset_index(drop=True)
        out_df['predicted_label'] = preds
        out_df['predicted_confidence'] = probs

        output_path = os.path.join(os.path.dirname(TEST_FILE),'test_predictions.csv')
        out_df.to_csv(output_path, index=False)
        print(f"Saved test predictions to {output_path}")

        return out_df

if __name__ == "__main__":
    # Run the main data preparation pipeline
    data = main_data_preparation()

    print("\n" + "=" * 70)
    print("Data preparation complete! Variables are ready to use.")
    print("=" * 70)

    # ----------------------------------------------------
    # BASELINE TRAINING
    # ----------------------------------------------------

    X_train = data['X_train']
    y_train = data['y_train']
    X_val   = data['X_val']
    y_val   = data['y_val']
    embedding_matrix = data['embedding_matrix']
    vocab_size        = data['vocab_size']
    embedding_dim     = data['embedding_dim']
    max_len           = data['max_sequence_length']
    num_classes       = data['num_classes']
    class_names = ["sadness","joy","love","anger","fear", "suprise"]

    # Limit the number of models to keep runtime reasonable (set max_models=None to run all combos)
    hp_summary_lstm = hyperparameter_search('lstm', X_train, y_train, X_val, y_val,
                                            vocab_size, embedding_dim, embedding_matrix,
                                            max_len, num_classes, PARAM_GRID_LSTM, max_models=6,
                                            results_filename='hp_results_lstm.json')

    hp_summary_gru = hyperparameter_search('gru', X_train, y_train, X_val, y_val,
                                           vocab_size, embedding_dim, embedding_matrix,
                                           max_len, num_classes, PARAM_GRID_GRU, max_models=6,
                                           results_filename='hp_results_gru.json')

    hp_best_metrics_lstm = evaluate_best_hp_and_save(hp_summary_lstm, X_val, y_val, class_names, 'lstm_hp')
    hp_best_metrics_gru = evaluate_best_hp_and_save(hp_summary_gru, X_val, y_val, class_names, 'gru_hp')

    tokenizer = data.get('tokenizer', None)

    # Choose model for final test predictions by KIND PARAM
    selected_model_path = None
    if BEST_MODEL_KIND == 'lstm' :
        if hp_summary_lstm and hp_summary_lstm.get('best_model_info'):
            selected_model_path = hp_summary_lstm['best_model_info']['model_path']
    elif BEST_MODEL_KIND == 'gru' :
        if hp_summary_gru and hp_summary_gru.get('best_model_info'):
            selected_model_path = hp_summary_gru['best_model_info']['model_path']

    if selected_model_path is not None:
        print(f"\nUsing model for test predictions: {selected_model_path}")
        final_model = load_model(selected_model_path)
        preds_df = predict_test_and_save(final_model, tokenizer, max_len=max_len)
    else:
        print("No model found for test predictions. Place a trained model in ./models or run HP search first.")

    print("\nAll additional steps finished.")