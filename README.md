# TITAN: Triple-Stream Integrated Attention Network for Fake News Detection

##  Overview

**TITAN (Triple-Stream Integrated Attention Network)** is a
transformer-based fake news detection system designed to classify news
as **Real** or **Fake** using only the **news title/headline**.

The project addresses a practical limitation of many fake-news detection
systems that depend on complete articles or additional information.
TITAN focuses on extracting multiple types of linguistic cues from short
titles by using three specialized streams:

-   **Semantic Stream** -- captures the meaning and contextual
    information in the title.
-   **Emotional Stream** -- captures emotional tone and emotionally
    charged language.
-   **Structural Stream** -- captures writing style and structural
    patterns such as clickbait-like expressions.

These representations are combined using a **Gated Fusion mechanism**,
along with the global **CLS representation** from DeBERTa-v3, before
final classification.

------------------------------------------------------------------------

##  Objectives

The main objectives of the project are:

1.  Detect fake news using only news titles.
2.  Extract contextual representations using a pre-trained
    **DeBERTa-v3** encoder.
3.  Capture semantic, emotional, and structural characteristics
    separately.
4.  Dynamically combine the three feature streams using gated fusion.
5.  Encourage complementary information between streams using orthogonal
    regularization.
6.  Handle class imbalance using focal loss.
7.  Evaluate generalization on two benchmark datasets: **GossipCop** and
    **WELFake**.

------------------------------------------------------------------------

##  Model Architecture

The overall TITAN pipeline is:

``` text
                 News Title
                     │
                     ▼
              Data Preprocessing
                     │
                     ▼
             DeBERTa-v3 Tokenizer
                     │
                     ▼
             DeBERTa-v3 Encoder
                     │
                     ▼
          Contextual Embeddings
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    Semantic      Emotional    Structural
     Stream        Stream        Stream
        │            │            │
        └────────────┼────────────┘
                     ▼
                Gated Fusion
                     │
                     ▼
          Fused Feature Representation
                     │
                     +───────────────┐
                     │               │
                     ▼               ▼
              Classification      CLS Pathway
                  Head                 │
                     │                 │
                     └───────┬─────────┘
                             ▼
                        Concatenation
                             │
                             ▼
                       Softmax Layer
                         │       │
                         ▼       ▼
                       REAL     FAKE
```

------------------------------------------------------------------------

##  How TITAN Works

### 1. Dataset Collection

Two benchmark datasets are used:

### GossipCop

GossipCop is based on the FakeNewsNet repository and contains
entertainment and celebrity-related news.

The project uses only the **title** field.

-   Approximately 5,000 fake news samples
-   Approximately 16,000 real news samples
-   The dataset is significantly imbalanced

### WELFake

WELFake is a larger dataset compiled from multiple news sources.

The project uses only the **title** field.

-   Approximately 37,000 fake news samples
-   Approximately 35,000 real news samples
-   More balanced than GossipCop

------------------------------------------------------------------------

### 2. Data Preprocessing

The raw titles are cleaned before training.

The preprocessing pipeline includes:

-   Removing special characters
-   Removing HTML tags
-   Removing unnecessary symbols
-   Converting text to lowercase
-   Removing extra whitespace
-   Removing samples with missing/null values

The cleaned dataset is divided using stratified sampling:

``` text
80% → Training
10% → Validation
10% → Testing
```

------------------------------------------------------------------------

### 3. Tokenization

The cleaned titles are tokenized using the **DeBERTa-v3 tokenizer** from
Hugging Face Transformers.

Each title is converted into:

-   **Input IDs** -- numerical representation of tokens
-   **Attention Mask** -- identifies valid tokens and padding

The maximum sequence length is **64 tokens**.

Shorter titles are padded, while longer titles are truncated.

------------------------------------------------------------------------

### 4. Contextual Feature Extraction

The tokenized title is passed through the pre-trained **DeBERTa-v3**
encoder.

DeBERTa generates contextual embeddings that represent the relationships
and meaning of words within the title.

The resulting representation becomes the input to the TITAN attention
architecture.

------------------------------------------------------------------------

### 5. Triple-Stream Attention

The contextual representation is projected into three feature subspaces.

#### Semantic Stream

Focuses on the semantic meaning and contextual information present in
the title.

#### Emotional Stream

Focuses on emotional tone and emotionally charged expressions.

#### Structural Stream

Focuses on linguistic and structural patterns in the title, including
patterns associated with headline style and clickbait.

Each stream uses attention to extract stream-specific features.

------------------------------------------------------------------------

### 6. Gated Multi-View Fusion

The three streams are not simply concatenated.

A **gated fusion mechanism** dynamically determines how much information
should be taken from each stream.

Conceptually:

``` text
Semantic Features   ─┐
Emotional Features  ─┼──► Gated Fusion ──► Combined Representation
Structural Features ─┘
```

This allows the model to give different importance to semantic,
emotional, and structural information depending on the input title.

------------------------------------------------------------------------

### 7. CLS Representation

The model also retains the global **CLS representation** generated by
DeBERTa.

The fused triple-stream representation and projected CLS representation
are combined:

``` text
Fused Stream Representation
             +
       CLS Representation
             │
             ▼
        Combined Vector
```

This provides both specialized multi-stream information and a global
contextual representation.

------------------------------------------------------------------------

### 8. Classification Head

The combined representation is passed through fully connected
neural-network layers.

Normalization and dropout are used to improve generalization and reduce
overfitting.

Finally, a **Softmax** layer produces probabilities for:

``` text
Real
Fake
```

The class with the higher probability becomes the final prediction.

------------------------------------------------------------------------

##  Diversity-Aware Regularization

One challenge with multiple feature streams is that they may learn
similar information.

TITAN uses **orthogonal regularization** to encourage the three streams
to learn complementary representations.

The objective is:

``` text
Semantic   → Meaning
Emotional  → Emotion
Structural → Style
```

rather than allowing all three streams to learn redundant features.

The orthogonal regularization weight used in the implementation is:

``` text
λortho = 0.1
```

------------------------------------------------------------------------

##  Focal Loss

GossipCop contains a substantial class imbalance.

To address this, TITAN uses **Focal Loss**, which places greater
emphasis on difficult examples.

The focal-loss parameter used is:

``` text
γ = 3.0
```

The overall training objective combines classification loss and
orthogonal regularization:

``` text
Total Loss = Focal Loss + λortho × Orthogonal Loss
```

------------------------------------------------------------------------

##  Training Configuration

The model is implemented using **Python and PyTorch** and trained with
GPU acceleration.

  Parameter                     Value
  ----------------------------- -----------------
  Framework                     PyTorch
  Transformer                   DeBERTa-v3
  Tokenizer                     DeBERTa-v3
  Optimizer                     AdamW
  Encoder Learning Rate         5 × 10⁻⁶
  Task Layer Learning Rate      5 × 10⁻⁵
  Batch Size                    16
  Maximum Epochs                10 or less
  Maximum Sequence Length       64
  Dropout                       0.5
  Focal Loss γ                  3.0
  Orthogonal Regularization λ   0.1
  GPU                           NVIDIA Tesla T4
  Training Platform             Kaggle

The encoder is partially frozen while the upper layers and task-specific
components are fine-tuned.

Early stopping is used based on validation performance.

Additional optimization techniques include:

-   Gradient clipping
-   Learning-rate scheduling
-   Dropout
-   Batch processing
-   Model checkpointing

------------------------------------------------------------------------

##  Results

TITAN was evaluated on both datasets.

  Dataset            Accuracy    F1 Score
  --------------- ----------- -----------
  **GossipCop**     **85.0%**   **0.848**
  **WELFake**       **96.4%**   **0.964**

### GossipCop Comparison

  Method                    Accuracy    F1 Score
  ---------------------- ----------- -----------
  SVM + TF-IDF                 69.8%       0.670
  CNN (Basic)                  72.4%       0.702
  FakeBERT (Headlines)         78.1%       0.775
  EANN (Text-Only)             78.6%       0.792
  Multi-View (LSTM)            80.4%       0.799
  **TITAN**                **85.0%**   **0.848**

### WELFake Comparison

  Method                      Accuracy    F1 Score
  ------------------------ ----------- -----------
  Random Forest                  86.1%       0.860
  Bi-LSTM (Title Only)           88.5%       0.880
  BERT-Base (Short Text)         91.0%       0.908
  DistilBERT (Headline)          92.3%       0.918
  RoBERTa-Base (Title)           93.8%       0.936
  **TITAN**                  **96.4%**   **0.964**

------------------------------------------------------------------------

##  Ablation Study

An ablation study was performed on GossipCop to understand the
contribution of different components.

  Model                           Accuracy    F1 Score
  --------------------------- ------------ -----------
  **TITAN**                     **85.00%**   **0.842**
  BERT + Full TITAN Head            83.79%      0.8341
  No Fusion + No Ortho + CE         75.97%      0.6560
  CLS-only (No Streams)             74.89%      0.6490

The study indicates that the multi-stream architecture, gated fusion,
and regularization contribute substantially to the final model
performance.

------------------------------------------------------------------------

##  Technologies Used

### Programming

-   Python

### Deep Learning

-   PyTorch
-   DeBERTa-v3

### NLP

-   Hugging Face Transformers
-   Transformer-based tokenization

### Data Processing

-   Pandas
-   NumPy

### Evaluation

-   Scikit-learn
-   Accuracy
-   Precision
-   Recall
-   F1 Score

### Visualization

-   Matplotlib
-   Seaborn

### Training Environment

-   Kaggle
-   NVIDIA Tesla T4 GPU

------------------------------------------------------------------------

##  Suggested Project Structure

``` text
TITAN/
│
├── data/
│   ├── GossipCop/
│   └── WELFake/
│
├── notebooks/
│   └── TITAN_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── tokenizer.py
│   ├── model.py
│   ├── attention_streams.py
│   ├── fusion.py
│   ├── losses.py
│   ├── train.py
│   └── evaluate.py
│
├── checkpoints/
│   └── best_model.pt
│
├── results/
│   ├── metrics/
│   └── plots/
│
├── requirements.txt
└── README.md
```

> The structure above is a suggested organization for the
> implementation. It is not claimed as the exact folder structure used
> in the report.

------------------------------------------------------------------------

##  End-to-End Pipeline

In simple terms:

``` text
1. Collect news datasets
        ↓
2. Extract only titles
        ↓
3. Clean the titles
        ↓
4. Split into train/validation/test
        ↓
5. Tokenize using DeBERTa-v3
        ↓
6. Generate contextual embeddings
        ↓
7. Create three streams
        ↓
8. Semantic attention
9. Emotional attention
10. Structural attention
        ↓
11. Gated fusion
        ↓
12. Add CLS representation
        ↓
13. Classification head
        ↓
14. Focal Loss + Orthogonal Loss
        ↓
15. Backpropagation and optimization
        ↓
16. Evaluate using Accuracy, Precision,
    Recall and F1 Score
        ↓
17. Predict REAL or FAKE
```

------------------------------------------------------------------------

##  Key Innovation

The main idea behind TITAN is that **fake news titles can contain
different types of signals**.

Instead of relying only on a single representation, TITAN separately
models:

``` text
             TITAN
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
  Meaning    Emotion    Style
     │         │         │
     └─────────┼─────────┘
               ▼
         Gated Fusion
               │
               ▼
          Classification
               │
          ┌────┴────┐
          ▼         ▼
        REAL       FAKE
```

This multi-view approach is the central architectural idea of the
project.

------------------------------------------------------------------------

##  Evaluation Metrics

The model is evaluated using:

-   **Accuracy** -- proportion of correctly classified titles.
-   **Precision** -- proportion of predicted positive cases that are
    correct.
-   **Recall** -- ability to identify relevant positive cases.
-   **F1 Score** -- balance between precision and recall.

F1 score is particularly useful when class distributions are imbalanced.

------------------------------------------------------------------------

##  Research Contribution

TITAN combines:

1.  **Pre-trained DeBERTa-v3 contextual representations**
2.  **Triple-stream attention**
3.  **Semantic, emotional, and structural feature extraction**
4.  **Gated multi-view fusion**
5.  **CLS representation pathway**
6.  **Diversity-aware orthogonal regularization**
7.  **Focal loss for class imbalance**

The system is designed specifically for **short-text/title-based fake
news detection**.

------------------------------------------------------------------------

##  Limitations

The project focuses on **title-only classification**. Because the model
does not use the full article, images, author information, source
credibility, or social-network context, it cannot directly verify
whether the underlying claim is factually true.

Therefore, the system should be understood as a **machine-learning
classification model for detecting patterns associated with fake/real
news titles**, rather than a replacement for human fact-checking.

------------------------------------------------------------------------

##  Future Scope

Potential extensions include:

-   Using full article content in addition to titles
-   Incorporating images and other multimodal information
-   Using source and author credibility features
-   Incorporating social-media propagation information
-   Improving robustness across different news domains
-   Testing on additional datasets
-   Developing a real-time fake-news detection application

------------------------------------------------------------------------

## ‍ Project Summary

**TITAN** is a **DeBERTa-v3-based triple-stream attention architecture**
for title-based fake news detection. It processes a news title through
semantic, emotional, and structural attention streams, dynamically
combines these representations using gated fusion, incorporates the CLS
representation, and performs binary classification into **Real** or
**Fake**.

The model is implemented in **Python/PyTorch**, trained using GPU
acceleration, and evaluated on **GossipCop and WELFake** datasets.

------------------------------------------------------------------------

##  Note

Dataset names, architecture components, training settings, and reported
performance values in this README are based on the project report
provided for TITAN.