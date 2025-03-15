import pandas as pd
import random
import numpy as np
from tqdm import tqdm

# ===============================
# 1. Load the Generated CSV Data
# ===============================
# For demonstration, we load a sample subset.
df = pd.read_csv("synthetic_transactions.csv", nrows=10000)
print("Data loaded. Shape:", df.shape)

# =============================================
# 2. Simulate Evolving Fraud Tactics (RL Style)
# =============================================
def simulate_evolving_fraud(df, modification_rate=0.05):
    n_mod = int(len(df) * modification_rate)
    indices = random.sample(list(df.index), n_mod)
    for idx in indices:
        # For simulation, force a record to become 'fraudulent' and increase the amount.
        df.at[idx, 'behavioral_history'] = 'fraudulent'
        df.at[idx, 'amount'] *= random.uniform(1.1, 1.5)
    return df

df = simulate_evolving_fraud(df, modification_rate=0.05)
print("Evolving fraud tactics simulated.")

# =========================================
# 3. Inject Adversarial Noise to the Data
# =========================================
def inject_adversarial_noise(df, noise_rate=0.05):
    n_noisy = int(len(df) * noise_rate)
    indices = random.sample(list(df.index), n_noisy)
    for idx in indices:
        noise = df.at[idx, 'amount'] * random.uniform(-0.1, 0.1)
        df.at[idx, 'amount'] += noise
    return df

df = inject_adversarial_noise(df, noise_rate=0.05)
print("Adversarial noise injected.")

# ============================================================
# 4. Multi-Modal Feature Engineering (Text-based Processing)
# ============================================================
# 4.A Prepare a Text Corpus for Tokenization & NER.
# We combine 'merchant' and 'session_metadata' into one text field.
df['text_features'] = df['merchant'] + " " + df['session_metadata']

# 4.A.1 Tokenization Using SentencePiece
import sentencepiece as spm

# Save the corpus to a temporary file.
with open("corpus.txt", "w", encoding="utf8") as f:
    for text in df['text_features']:
        f.write(text + "\n")

# Train SentencePiece model (using a small vocab size for demo)
spm.SentencePieceTrainer.Train("--input=corpus.txt --model_prefix=spm_model --vocab_size=500 --model_type=bpe")
sp = spm.SentencePieceProcessor(model_file="spm_model.model")

# Tokenize the text_features
df['sp_tokens'] = df['text_features'].apply(lambda x: sp.EncodeAsPieces(x))
print("SentencePiece tokenization complete. Sample tokens:")
print(df['sp_tokens'].head())

# 4.A.2 Named Entity Recognition (NER) Using spaCy
import spacy
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

df['ner_entities'] = df['text_features'].apply(extract_entities)
print("NER extraction complete. Sample entities:")
print(df['ner_entities'].head())

# 4.A.3 Word Embeddings with Word2Vec (using gensim)
from gensim.models import Word2Vec

# Train Word2Vec on the SentencePiece tokens.
tokenized_texts = df['sp_tokens'].tolist()
w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=50, window=5, min_count=1, workers=4)
print("Word2Vec model trained.")
# Example: Print embedding for a sample word (if it exists)
sample_word = "Inc"
if sample_word in w2v_model.wv:
    print(f"Embedding for '{sample_word}':", w2v_model.wv[sample_word])
else:
    print(f"Word '{sample_word}' not in vocabulary.")

# =====================================================
# 5. Multi-Modal Feature Engineering (Structured Features)
# =====================================================
# 5.B.1 Normalize Numerical Features
num_cols = ['age', 'credit_score', 'amount', 'network_latency', 'velocity']
X = df[num_cols].values.astype(np.float32)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
print("Numerical features normalized.")

# 5.B.2 Build an Autoencoder for Structured Data Embeddings using PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

input_dim = X_norm.shape[1]
latent_dim = 3
autoencoder = Autoencoder(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

X_tensor = torch.from_numpy(X_norm)

epochs = 20  # For demo; use more epochs for better training.
for epoch in range(epochs):
    optimizer.zero_grad()
    recon, z = autoencoder(X_tensor)
    loss = criterion(recon, X_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Extract latent embeddings
with torch.no_grad():
    _, latent_embeddings = autoencoder(X_tensor)
latent_embeddings = latent_embeddings.numpy()
print("Autoencoder training complete. Latent embeddings shape:", latent_embeddings.shape)

# 5.B.3 Apply PCA for Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(latent_embeddings)
print("PCA result shape:", pca_result.shape)

# 5.B.4 t-SNE Visualization for Fraud Pattern Clustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_result = tsne.fit_transform(latent_embeddings)

# Plot t-SNE result color-coded by label.
plt.figure(figsize=(8,6))
labels = df['label'].values
unique_labels = np.unique(labels)
colors = ['blue', 'red', 'orange']
for ul, c in zip(unique_labels, colors):
    idx = np.where(labels == ul)
    plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], c=c, label=ul, alpha=0.5)
plt.legend()
plt.title("t-SNE of Structured Embeddings")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()

print("Multi-modal feature engineering complete.")
