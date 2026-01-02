# Distinguishing Suicidal Ideation from Depression: A Two-Stage NLP Pipeline with Unsupervised Label Correction

**Author:** A. Author (Student, FAST NUCES)

## Abstract
This project presents a novel Natural Language Processing (NLP) framework designed to classify the distinct psychological states of **Suicidal Ideation** versus **Depression** in social media text.

Unlike traditional approaches that perform binary classification between "Healthy" and "Suicidal" states, this study addresses the noisy nature of user-generated content (where depression and suicide forums often overlap in intent). The solution employs a two-stage pipeline:

1.  **Unsupervised Label Correction:** Using clustering and confidence thresholds to clean noisy ground-truth labels.
2.  **Supervised Classification:** Training Deep Neural Networks on the refined data to achieve superior accuracy.

---

## Pipeline Architecture
The system follows a sequential workflow:

1.  **Data Collection:** Web-scraping Reddit (`r/depression`, `r/SuicideWatch`).
2.  **Feature Extraction:** Generating embeddings using BERT and Google Universal Sentence Encoder (GUSE).
3.  **Unsupervised Correction (Phase 1):** Dimensionality reduction and clustering to re-label data.
4.  **Supervised Classification (Phase 2):** Training CNN, Bi-LSTM, and Dense Networks on corrected labels.

---

## Methodology

### 1. Feature Extraction (Embeddings)
Raw text is converted into numerical vectors using Transformer-based models:
* **BERT (bert-base-uncased):** Tokenized to max length 512, extracting last hidden states.
* **Google Universal Sentence Encoder (GUSE):** Implemented via TensorFlow Hub for high-dimensional semantic vectors.

### 2. Dimensionality Reduction
To mitigate the curse of dimensionality before clustering, three techniques were evaluated:
* **PCA:** Linear transformation.
* **Deep Autoencoder:** Encoder-Decoder architecture (Input 512 → Enc 2 → Dec 512) trained with MSE loss.
* **UMAP:** Manifold approximation (`n_neighbors=45`, `min_dist=0.7`).

### 3. Clustering & Label Correction
Reduced features are processed by **GMM (Gaussian Mixture Models)**, **K-Means**, and **Sparse Subspace Clustering (EnSC, SSC-OMP)**.

Noisy labels are corrected using a confidence-based thresholding formula:

```math
L_{Final}^{(i)} = \begin{cases} 
L_{Pred}^{(i)} & \text{if } L_{GT}^{(i)} \neq L_{Pred}^{(i)} \text{ and } (P_{Pred}^{(i)} > \tau \text{ or } P_{Pred}^{(i)} < 1-\tau) \\
L_{GT}^{(i)} & \text{otherwise}
\end{cases}
