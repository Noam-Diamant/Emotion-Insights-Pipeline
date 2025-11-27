"""
Data Preparation and GloVe Embedding Setup for Emotion Detection
This script handles text preprocessing, tokenization, and GloVe embedding matrix creation.
"""

import numpy as np
import pandas as pd
import re
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
MAX_SEQUENCE_LENGTH = 66  # Maximum length in dataset - no truncation
EMBEDDING_DIM = 100
GLOVE_FILE = './data/glove.6B.100d.txt'
TRAIN_FILE = './data/train.csv'
VALIDATION_FILE = './data/validation.csv'


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
    label_column = 'label' if 'label' in df.columns else df.columns[1]

    texts = df[text_column].values
    labels = df[label_column].values

    print(f"Number of samples: {len(texts)}")
    print(f"Number of unique emotions: {len(np.unique(labels))}")
    print(f"Emotion distribution: {np.bincount(labels)}")

    return texts, labels


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


def load_glove_embeddings(glove_file):
    """
    Load pre-trained GloVe word embeddings.

    Args:
        glove_file: Path to the GloVe embeddings file

    Returns:
        dict: Dictionary mapping words to embedding vectors
    """
    print("\n" + "=" * 70)
    print("STEP 5: Loading GloVe Embeddings")
    print("=" * 70)

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


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
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
    embeddings_index = load_glove_embeddings(GLOVE_FILE)

    # ========================================================================
    # STEP 6: Create Embedding Matrix
    # ========================================================================
    embedding_matrix = create_embedding_matrix(
        tokenizer, embeddings_index, vocab_size, EMBEDDING_DIM
    )

    # ========================================================================
    # Save Processed Data
    # ========================================================================
    save_processed_data(
        train_sequences, train_labels, val_sequences, val_labels,
        embedding_matrix, tokenizer
    )

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


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the main data preparation pipeline
    data = main()

    print("\n" + "=" * 70)
    print("Data preparation complete! Variables are ready to use.")
    print("=" * 70)
