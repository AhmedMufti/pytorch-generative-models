# Q1 Demo Prep — Machine Translation (English → Urdu)

## What is Q1 About?

We built a **machine translation system** that takes an English sentence and tries to produce its Urdu translation. Think of it like a very basic Google Translate, but built from scratch using a neural network.

The specific type of model we used is called a **Sequence-to-Sequence (Seq2Seq) model** with a **Vanilla RNN** (basic Recurrent Neural Network — NOT LSTM, NOT GRU, NOT Transformer).

---

## The Big Picture (How Translation Works in Our Code)

```
English sentence → [Tokenizer] → Numbers → [ENCODER RNN] → Context Vector → [DECODER RNN] → Numbers → [Detokenizer] → Urdu sentence
```

There are two main parts:

### 1. ENCODER (reads English)
- Takes in an English sentence word by word
- Processes each word through an RNN cell
- After reading ALL English words, it produces a single **context vector** (a list of 512 numbers)
- This context vector is supposed to capture the ENTIRE meaning of the English sentence

### 2. DECODER (writes Urdu)
- Starts with the context vector from the encoder
- Generates Urdu words one at a time
- At each step, it looks at:
  - The previous word it generated
  - Its hidden state (memory of what it has generated so far)
- Keeps going until it produces an end-of-sentence token or hits max length

**Key limitation:** The ENTIRE English sentence is squeezed into one fixed-size vector (512 numbers). Imagine trying to summarize a full paragraph into a single sentence — you would lose details. That is exactly what happens here.

---

## Step-by-Step: What the Code Does (In Order)

### Task 1: Data Preprocessing
**What:** We load the dataset and clean it up.

- **Dataset:** An Excel file with 9,102 English-Urdu sentence pairs from Kaggle
- **Cleaning:**
  - Removed 9 duplicate pairs
  - Removed any rows with missing values (0 found)
  - **Filtered** to keep only sentences with 15 words or fewer — left with 2,019 pairs
- **Why filter to 15 words?** Vanilla RNN cannot handle long sentences. Its gradients vanish after about 10-15 time steps, so longer sentences are wasted data.

**If asked:** "We preprocessed by removing duplicates and missing values. Since vanilla RNN struggles with long sequences due to vanishing gradients, we also filtered to sentences of 15 words or fewer."

---

### Task 2: Dataset Splits
**What:** We split the 2,019 pairs into three groups:

| Split | Count | % | Purpose |
|-------|-------|---|---------|
| Train | 1,615 | 80% | Model learns from these |
| Validation | 201 | 10% | Monitor overfitting during training |
| Test | 203 | 10% | Final evaluation (model never sees these during training) |

We verified programmatically that there is **zero overlap** between splits.

**If asked:** "We use 80/10/10 split. The model only trains on the training set. Validation is used to track if the model is overfitting. Test is held out completely and only used at the very end."

---

### Task 3: Vocabulary Building
**What:** We create a mapping between words and numbers, because neural networks only understand numbers.

- English vocabulary: 2,438 unique words
- Urdu vocabulary: 3,011 unique words
- Max vocab cap: 5,000 (rare words become `<unk>`)
- 4 special tokens:
  - `<pad>` = 0 (used to make all sentences same length in a batch)
  - `<sos>` = 1 (start of sentence — tells decoder to begin)
  - `<eos>` = 2 (end of sentence — tells decoder to stop)
  - `<unk>` = 3 (unknown word — for rare words not in vocabulary)

**Example:**
```
"I like cats" → [1, 45, 289, 1034, 2]
                <sos> I  like cats <eos>
```

**If asked:** "Each word gets a unique integer ID. We cap the vocabulary at 5,000 to focus on frequent words. Words seen fewer times become unk tokens."

---

### Task 4: Batching and Padding
**What:** We group sentences into batches of 32 for efficient GPU processing.

- Different sentences have different lengths
- We **pad** shorter sentences with `<pad>` (0) so all sentences in a batch have the same length
- Batch shape: `[32, 17]` = 32 sentences, each up to 17 tokens long

**If asked:** "We pad sequences to equal length within each batch so the GPU can process them in parallel. Padding tokens are ignored during loss computation."

---

### Task 5 and 6: Model Architecture
**What:** The actual neural network. Three classes:

#### Encoder (reads English)
```
Input word → Embedding(2438, 256) → RNN(256, 512, 2 layers) → Hidden state
```
- **Embedding:** converts word ID into a 256-dimensional vector (word meaning representation)
- **RNN:** processes the sequence, 512-dimensional hidden state, 2 layers stacked
- **Output:** final hidden state = the "context vector"

#### Decoder (writes Urdu)
```
Previous word → Embedding(3011, 256) → RNN(256, 512, 2 layers) → Linear(512, 3011) → Next word prediction
```
- Same structure but reversed
- **Linear layer** at the end converts the 512-dim hidden state into probabilities over all 3,011 Urdu words
- Picks the highest probability word as the next output

#### Seq2Seq (combines both)
- Connects encoder output to decoder input
- Uses **teacher forcing** during training

**Total parameters:** 4,778,691 (about 4.8 million learnable numbers)

**If asked:** "The encoder reads the English sentence and compresses it into a 512-dimensional vector. The decoder takes that vector and generates Urdu words one at a time. We used 2 RNN layers with 256-dim embeddings and 512-dim hidden states."

---

### What is Teacher Forcing?

During TRAINING only:
- **With teacher forcing:** At each step, feed the CORRECT target word as input to the next step (even if the model predicted wrong)
- **Without teacher forcing:** Feed whatever the model ACTUALLY predicted

We decay teacher forcing from **100% to 50%** over 30 epochs. This means:
- Early training: always feed correct words, so the model learns word mappings
- Late training: sometimes feed its own predictions, so it learns to recover from its own mistakes

**If asked:** "Teacher forcing feeds the ground truth as decoder input during training. We decay it from 100% to 50% so the model gradually learns to rely on its own predictions instead of always getting the correct answer."

---

### Task 7: Hyperparameter Tuning
**What:** We tried 4 different configurations to find the best one.

| Config | emb_dim | hid_dim | layers | lr | dropout | Val Loss |
|--------|---------|---------|--------|----|---------|----------|
| 1 | 128 | 256 | 1 | 0.003 | 0.1 | 7.81 |
| 2 | 256 | 256 | 1 | 0.001 | 0.1 | 7.12 |
| **3 (Best)** | **256** | **512** | **2** | **0.001** | **0.3** | **6.87** |
| 4 | 128 | 512 | 1 | 0.003 | 0.1 | 8.40 |

Each config was trained for 5 quick epochs and the one with lowest validation loss was selected.

**Best: emb=256, hid=512, 2 layers, lr=0.001, dropout=0.3**

**If asked about each hyperparameter:**
- **emb_dim (256):** Size of word embedding vectors. Larger = richer word representations.
- **hid_dim (512):** Size of RNN hidden state. Larger = more memory capacity.
- **n_layers (2):** Stacking 2 RNN layers for more depth.
- **lr (0.001):** Learning rate — how big the weight updates are. Too high = unstable, too low = slow.
- **dropout (0.3):** Randomly turns off 30% of neurons during training to prevent memorization.

---

### Task 5 and 6 Continued: Full Training (30 Epochs)

The best config was trained for 30 full epochs. Here is what happened:

| Metric | Start (Epoch 1) | End (Epoch 30) | What it means |
|--------|-----------------|----------------|---------------|
| Train Loss | 6.23 | 2.92 | Model learned the training data well (loss halved) |
| Val Loss | 6.32 | 8.03 | Model got WORSE on unseen data (overfitting) |
| Teacher Forcing | 100% | 52% | Gradually reduced |

**The training curve (q1_training_curves.png) shows:**
- Blue line (train loss) going DOWN — model is memorizing training examples
- Orange line (val loss) going UP — model cannot generalize to new sentences
- The gap between them = **overfitting**
- Best checkpoint was saved at **Epoch 1** (lowest val loss = 6.32)

**If asked:** "The training loss decreased steadily showing the model is learning, but validation loss increased, indicating overfitting. This is expected with only 1,615 training pairs. The model memorizes training data rather than learning general translation rules."

---

### Task 8: Inference (Greedy + Beam Search)

Two decoding strategies were used on the test set:

#### Greedy Decoding
- At each step, pick the single highest probability word
- Fast but can make irreversible mistakes early

#### Beam Search (k=3)
- Keep the top 3 most promising partial translations at each step
- Explore multiple paths and pick the best complete sentence
- Slower but can find better translations

**Results:**
| Method | BLEU Score |
|--------|-----------|
| Greedy | 0.0162 |
| Beam (k=3) | 0.0180 |

**What is BLEU?**
BLEU measures how many word sequences (1-gram, 2-gram, 3-gram, 4-gram) in the predicted translation match the reference translation.
- BLEU = 1.0 means perfect match
- BLEU = 0.0 means nothing matches
- Our score of about 0.017 = very poor (almost no overlap with reference)

**If asked:** "Greedy just picks the most likely word at each step. Beam search is smarter. It keeps the top k=3 candidates at each step and picks the best full sequence at the end. Beam search gave slightly better BLEU (0.018 vs 0.016), but both are very low."

---

### Task 9: Error Analysis

We analyzed 30 test translations. Results:

```
word_repetition: 30/30
wrong_translation: 0/30
truncated_output: 0/30
unk_tokens: 0/30
acceptable: 0/30
```

**What the model actually outputs:**
For ANY English input, the output is always something like: "and did in in in from in"
(These are the most common Urdu function words: "and", "did", "in", "from")

**Why?** The model has collapsed. It just outputs the most frequent words in Urdu regardless of input. Think of a person who only knows 5 Urdu words and just keeps saying them no matter what you ask.

---

## WHY Did Translation Fail? (Most Important Section for Demo)

### Reason 1: Vanishing Gradients (PRIMARY CAUSE)
**What it is:** In a vanilla RNN, information from early words gets weaker and weaker as the sequence gets longer. By the time the encoder finishes a 15-word sentence, it has almost forgotten the first few words.

**Why it matters:** The "hidden state" that gets passed from encoder to decoder has effectively lost most of the English sentence meaning.

**Analogy:** Imagine playing the telephone game with 15 people. The message at the end barely resembles the original.

**Fix:** LSTM/GRU have "gates" that can selectively remember or forget, preventing this fade-out.

### Reason 2: Fixed Context Vector Bottleneck
**What it is:** The ENTIRE English sentence gets compressed into ONE vector of 512 numbers. Whether the sentence is 3 words or 15 words, it all has to fit into those 512 numbers.

**Why it matters:** You cannot store the meaning of "Tom was about to say something" in 512 numbers precisely enough for the decoder to reconstruct a correct Urdu translation.

**Fix:** Attention mechanism lets the decoder "look back" at each English word individually at each decoding step, instead of relying on just one summary vector.

### Reason 3: Too Little Training Data
**What it is:** We only have 1,615 training sentence pairs. Real translation systems use MILLIONS.

**Why it matters:** The model saw each sentence about 30 times over 30 epochs. That is not enough to learn the complex mapping between two completely different languages (English and Urdu have different word order, grammar, script, everything).

**Evidence:** Train loss halved but val loss nearly doubled = model memorized training data, learned nothing generalizable.

### Reason 4: Exposure Bias
**What it is:** During training, the decoder gets the correct previous word (teacher forcing). During inference, it gets its own (often wrong) previous prediction.

**Why it matters:** The model was never trained to recover from its own mistakes. One wrong word causes a cascade of wrong words.

---

## Possible Future Improvements

| Improvement | What It Does |
|-------------|-------------|
| **Attention Mechanism** | Decoder can look at all encoder outputs, not just the final one |
| **LSTM/GRU** | Gating mechanisms solve vanishing gradients |
| **Transformer** | Self-attention, parallel processing, state-of-the-art for translation |
| **BPE Tokenization** | Break words into subwords, reducing vocabulary and handling unseen words |
| **More Data** | 100k+ sentence pairs instead of 1.6k |

---

## Quick Demo Q&A Cheat Sheet

**Q: What model did you use?**
A: Vanilla RNN Encoder-Decoder (Seq2Seq). The assignment specifically required vanilla RNN, not LSTM or Transformer.

**Q: Why are the translations so bad?**
A: Three main reasons: (1) vanilla RNN suffers from vanishing gradients so it forgets early parts of sentences, (2) the entire sentence is compressed into a single 512-dim vector which is a bottleneck, and (3) we only had 1,615 training pairs which is far too little for translation.

**Q: What is BLEU score and what does 0.016 mean?**
A: BLEU measures how many word sequences in the prediction match the reference. 1.0 is perfect, 0.0 is nothing matches. 0.016 means almost no word overlap. The translations are essentially random common words.

**Q: What does overfitting mean in your results?**
A: Training loss went from 6.23 to 2.92 (model learned training data well), but validation loss went from 6.32 to 8.03 (model performs worse on unseen data). The model memorized rather than generalized.

**Q: What is teacher forcing?**
A: During training, instead of feeding the model its own previous prediction at each step, we feed the actual correct word. We decayed this from 100% to 50% over training so the model gradually learns to work with its own predictions.

**Q: What is beam search?**
A: Instead of just picking the single best word at each step (greedy), beam search keeps the top k=3 candidates and explores multiple paths. It gave slightly better BLEU (0.018 vs 0.016).

**Q: How would you improve the model?**
A: Use attention mechanism so the decoder can look at specific encoder words, switch to LSTM/GRU to handle vanishing gradients, or use a Transformer architecture. Also, a much larger dataset (100k+ pairs) would help significantly.

**Q: Why did you filter sentences to 15 words?**
A: Vanilla RNN gradients vanish after about 10-15 timesteps, so longer sentences would be wasted. The model cannot effectively learn from them. Filtering keeps the data within the model's capacity.

**Q: What is the context vector?**
A: It is the final hidden state of the encoder, a fixed-size vector (512 numbers) that is supposed to represent the entire meaning of the source sentence. The decoder uses only this vector to generate the translation.

**Q: What does dropout do?**
A: During training, it randomly turns off 30% of neurons. This prevents the model from relying too heavily on any single neuron and encourages it to learn more robust features. It is a regularization technique.

**Q: What optimizer did you use?**
A: Adam. It is an adaptive learning rate optimizer that adjusts the learning rate for each parameter individually based on recent gradient history.

**Q: What loss function did you use?**
A: Cross-entropy loss. It measures how far the model's predicted probability distribution over all Urdu words is from the correct answer. Padding tokens are ignored in the loss calculation.

**Q: Why not use LSTM or Transformer?**
A: The assignment specifically required vanilla RNN. The whole point is to see its limitations first-hand and understand why more advanced architectures were invented.
