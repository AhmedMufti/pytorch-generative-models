# Question 1: Machine Translation (English → Urdu)

## Overview
**Problem:** The goal is to build a Neural Machine Translation (NMT) system that translates English sentences into Urdu.

**Dataset:**
- **Source:** [Kaggle English-Urdu Parallel Corpus](https://www.kaggle.com/datasets/zeeshanmulla/englishurdu-parallel-corpus)
- **Size:** Approximately 24,000 sentence pairs.
- **Format:** CSV file containing paired English and Urdu sentences.

**Model Check:**
- **Architecture:** Sequence-to-Sequence (Seq2Seq) with **Vanilla RNNs**.
- **Constraint:** The assignment explicitly forbids the use of LSTMs, GRUs, or Transformers for this specific task to demonstrate understanding of basic RNN mechanics.

## Approach & Methodology

### 1. Data Preprocessing
- **Normalization:** Convert text to lowercase, remove special characters (for English), and clean whitespace.
- **Tokenization:**
  - We build a custom vocabulary for both English and Urdu.
  - Special tokens are added: `<sos>` (Start of Sentence), `<eos>` (End of Sentence), `<pad>` (Padding), and `<unk>` (Unknown).
  - Sentences are converted into sequences of integer indices.

### 2. Model Architecture (Seq2Seq)
The model consists of two main components:
- **Encoder:**
  - Takes the English sentence as input.
  - Passes it through an Embedding layer to get dense vector representations.
  - Uses a **Vanilla RNN** to process the sequence step-by-step.
  - The final hidden state of the RNN captures the context of the entire source sentence.
- **Decoder:**
  - Takes the Encoder's final hidden state as its initial hidden state.
  - Uses a **Vanilla RNN** to generate the Urdu translation one word at a time.
  - Feeds the predicted word (or ground truth during training) as input for the next step.

### 3. Training
- **Loss Function:** Cross Entropy Loss (ignoring padding indices).
- **Optimizer:** Adam optimizer.
- **Teacher Forcing:** A technique used during training where the model is sometimes fed the actual previous ground-truth word instead of its own prediction to stabilize convergence.

### 4. Evaluation
- **Metric:** BLEU Score, a standard metric for machine translation quality.
- **Qualitative:** We manually inspect sample translations to see if the model captures the meaning, even if the grammar isn't perfect (a common limitation of vanilla RNNs on long sequences).

## Execution Steps (Checklist)

1.  **Environment Setup (Cloud):**
    -   Open [Google Colab](https://colab.research.google.com/).
    -   Create a new notebook.
    -   **Important:** Set Runtime to **T4 GPU**.
2.  **Upload Files:**
    -   Upload `solution.py`.
3.  **Run the Script:**
    -   In a code cell, run the following commands:
        ```python
        !pip install kagglehub openpyxl
        !python solution.py
        ```
    -   The script will now *automatically* attempt to download the English-Urdu dataset (which might be an Excel file) using `kagglehub`.
4.  **Monitor Training:**
    -   Watch the console output. You will see "Q1: Machine Translation" followed by Epoch Loss.
    -   The loss should decrease (e.g., starting around 7.0 and dropping).
5.  **Analyze Results:**
    -   Once finished, the script will print a BLEU score (if NLTK is available/installed using `!pip install nltk`).
    -   It will print "Q1 Finished".
6.  **Iterate (Optional):**
    -   **When to stop:** If the loss is still high (e.g., > 4.0) or translations are nonsensical.
    -   **How to improve:** Edit `solution.py` to increase `N_EPOCHS = 2` to `N_EPOCHS = 10` or `20`. Re-run the script.
