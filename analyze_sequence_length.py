"""
Quick analysis to determine optimal MAX_SEQUENCE_LENGTH
"""
import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load data
df = pd.read_csv('./data/train.csv')
text_column = 'text' if 'text' in df.columns else df.columns[0]
texts = df[text_column].values

# Preprocess
processed_texts = [preprocess_text(str(text)) for text in texts]

# Tokenize
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(processed_texts)
sequences = tokenizer.texts_to_sequences(processed_texts)

# Analyze sequence lengths
sequence_lengths = [len(seq) for seq in sequences]

print("=" * 70)
print("SEQUENCE LENGTH ANALYSIS")
print("=" * 70)
print(f"\nTotal samples: {len(sequence_lengths)}")
print(f"\nSequence Length Statistics:")
print(f"  Min length: {np.min(sequence_lengths)}")
print(f"  Max length: {np.max(sequence_lengths)}")
print(f"  Mean length: {np.mean(sequence_lengths):.2f}")
print(f"  Median length: {np.median(sequence_lengths):.2f}")
print(f"  Std deviation: {np.std(sequence_lengths):.2f}")

print(f"\nPercentiles:")
percentiles = [50, 75, 90, 95, 99, 100]
for p in percentiles:
    value = np.percentile(sequence_lengths, p)
    coverage = (np.array(sequence_lengths) <= value).sum() / len(sequence_lengths) * 100
    print(f"  {p}th percentile: {value:.0f} words (covers {coverage:.1f}% of samples)")

print(f"\nRecommendations:")
print(f"  - For 90% coverage: MAX_SEQUENCE_LENGTH = {int(np.percentile(sequence_lengths, 90))}")
print(f"  - For 95% coverage: MAX_SEQUENCE_LENGTH = {int(np.percentile(sequence_lengths, 95))}")
print(f"  - For 99% coverage: MAX_SEQUENCE_LENGTH = {int(np.percentile(sequence_lengths, 99))}")

# Distribution
print(f"\nSequence Length Distribution:")
bins = [0, 20, 40, 60, 80, 100, 150, 200, max(sequence_lengths)]
for i in range(len(bins)-1):
    count = sum(1 for l in sequence_lengths if bins[i] < l <= bins[i+1])
    pct = count / len(sequence_lengths) * 100
    print(f"  {bins[i]+1:3d}-{bins[i+1]:3d} words: {count:5d} samples ({pct:5.2f}%)")
