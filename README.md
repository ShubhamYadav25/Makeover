# Makemore — Character-Level Language Model

A complete educational implementation of the **MLP character-level language model** from Andrej Karpathy's Makemore walkthrough.

The goal is not just to run the model, but to understand **why every line exists**.

---

## 1. What are we building?

We build a tiny language model that learns to generate names one character at a time.

For example, after training it might generate:

```text
anil
rohan
meera
alex
...
```

The important part is that the model does **not** memorize a fixed list of names during generation.

It learns a probability distribution such as:

```text
Given:
    "an"

Predict:

    n → 0.40
    i → 0.25
    a → 0.10
    d → 0.08
    ...
```

It then samples the next character and feeds it back into the model.

---

# 2. Setup

You need Python and PyTorch.

```bash
pip install torch
```

Then:

```bash
python makemore_from_scratch.py
```

On Windows PowerShell:

```powershell
python makemore_from_scratch.py
```

---

# 3. What happens from start to end?

```text
                    ┌────────────────────┐
                    │      WORDS         │
                    │ anna, arjun, ...   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   TOKENIZATION     │
                    │ characters → IDs   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ CONTEXT WINDOWS    │
                    │ previous 3 chars   │
                    │        →           │
                    │ next character     │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ TRAIN / VAL / TEST │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │    EMBEDDING       │
                    │ ID → vector        │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │      FLATTEN       │
                    │ 3×10 → 30 values   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   HIDDEN LAYER     │
                    │   W1 + b1 + tanh   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │      LOGITS        │
                    │ score per char     │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  CROSS ENTROPY     │
                    │      LOSS          │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   BACKPROPAGATION  │
                    │ calculate gradients│
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ GRADIENT DESCENT   │
                    │ update parameters  │
                    └─────────┬──────────┘
                              │
                              └───────┐
                                      │
                                      ▼
                                REPEAT TRAINING
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │       GENERATION       │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │ context = "..."/IDs   │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │       MODEL            │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │       LOGITS           │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │       SOFTMAX          │
                         │ logits → probabilities │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │       SAMPLING         │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │   NEXT CHARACTER       │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │ UPDATE CONTEXT         │
                         │ remove oldest + add    │
                         │ newly generated char   │
                         └────────────┬───────────┘
                                      │
                                      └──────────────┐
                                                     │
                                                     ▼
                                                   REPEAT
                                                     │
                                             "." = END
                                                     │
                                                     ▼
                                             GENERATED NAME
```

---

# 4. Tokenization

## What is tokenization?

Tokenization converts text into units that a neural network can represent numerically.

Here we use **character-level tokenization**.

Example:

```text
"cat"
```

becomes:

```text
['c', 'a', 't']
```

We then create a vocabulary.

Example:

```text
['.', 'a', 'b', 'c', ...]
```

and assign IDs:

```text
'.' → 0
'a' → 1
'b' → 2
'c' → 3
...
```

Therefore:

```text
"cat"
```

might become:

```text
[3, 1, 20]
```

The exact IDs depend on the vocabulary.

### Why do we need it?

PyTorch neural-network operations work with numbers.

We cannot directly do:

```text
"c" × weight
```

but we can do:

```text
3 → embedding → vector → neural network
```

The pipeline is:

```text
text
 ↓
characters
 ↓
integer IDs
 ↓
embeddings
 ↓
neural network
```

The code creates two mappings:

```python
stoi
```

means:

```text
string → integer
```

and:

```python
itos
```

means:

```text
integer → string
```

`itos` is especially important during generation.

The model predicts:

```text
7
```

but we need to convert:

```text
7 → 'h'
```

---

# 5. Context Window

We use:

```python
block_size = 3
```

This means the model looks at the previous three characters.

For:

```text
anna
```

we create:

```text
context       target

...           a
..a           n
.an           n
ann           a
nna           .
```

`.` is a special boundary token.

So the neural network learns:

```text
previous characters → next character
```

---

# 6. Embeddings

Instead of representing a character using a huge one-hot vector, we give every character a small learned vector.

For example:

```text
'a' → [ 0.21, -0.42, 0.73, ... ]
'n' → [-0.11,  0.55, 0.09, ... ]
```

In this implementation:

```python
embedding_size = 10
```

So every character gets a 10-dimensional vector.

The embedding table has shape:

```text
[vocab_size, embedding_size]
```

If vocabulary size is 27:

```text
[27, 10]
```

---

# 7. Hidden Layer

The three character embeddings are concatenated.

If:

```text
block_size = 3
embedding_size = 10
```

then:

```text
3 × 10 = 30
```

values enter the hidden layer.

With:

```text
hidden_size = 100
```

the first layer is:

```text
[batch, 30]
        ×
[30, 100]
        =
[batch, 100]
```

Then we apply:

```python
tanh()
```

The non-linearity allows the model to learn more complex relationships than a purely linear model.

---

# 8. Logits

The hidden layer produces:

```text
[batch, 100]
```

The output layer converts that into:

```text
[batch, vocab_size]
```

If there are 27 possible characters:

```text
[batch, 27]
```

These 27 numbers are **logits**.

Example:

```text
a → 2.4
b → -1.2
c → 0.7
d → 4.1
...
```

The largest score does not necessarily have to be selected during generation.

We can convert logits into probabilities using softmax.

---

# 9. Cross Entropy

The target tells us the correct character.

Suppose:

```text
target = 'd'
```

but the model gives:

```text
a = 0.10
b = 0.05
c = 0.10
d = 0.70
...
```

That's good because the correct answer has high probability.

If instead:

```text
d = 0.01
```

the loss becomes much larger.

The model therefore learns to assign high probability to the correct next character.

Important:

```python
F.cross_entropy(logits, target)
```

expects **raw logits**.

Do not manually softmax first.

---

# 10. Backpropagation

After calculating loss:

```python
loss.backward()
```

PyTorch calculates gradients such as:

```text
dLoss/dC
dLoss/dW1
dLoss/db1
dLoss/dW2
dLoss/db2
```

These tell us how each parameter affects the loss.

Then:

```python
parameter -= learning_rate * gradient
```

moves the parameters toward lower loss.

---

# 11. Mini-Batches

Instead of using the entire dataset for every update:

```python
batch_size = 32
```

we randomly select 32 examples.

Example:

```text
full dataset
     ↓
random 32 examples
     ↓
forward
     ↓
loss
     ↓
backward
     ↓
update
```

Then another random batch is selected.

---

# 12. Train / Validation / Test

We split the examples into:

```text
80% → training
10% → validation
10% → test
```

Training data updates the parameters.

Validation data helps us detect overfitting while developing.

Test data gives a final evaluation.

---

# 13. Generation

Training is:

```text
context → correct answer
```

Generation reverses the direction:

```text
context
   ↓
model
   ↓
probabilities
   ↓
sample
   ↓
next character
   ↓
new context
   ↓
repeat
```

Example:

```text
... → a
..a → n
.an → i
ani → l
nil → .
```

Result:

```text
anil
```

This is called **autoregressive generation**.

---

# 14. Temperature

Generation uses:

```python
softmax(logits / temperature)
```

### Temperature < 1

Example:

```text
temperature = 0.7
```

Makes the distribution sharper.

Result:

```text
more predictable
less random
```

### Temperature = 1

Normal distribution.

### Temperature > 1

Example:

```text
temperature = 1.5
```

Makes the distribution flatter.

Result:

```text
more random
more diverse
```

This same concept is used when generating text from modern LLMs.

---

# 15. Sampling vs Argmax

Argmax:

```python
torch.argmax(probs)
```

always selects the highest probability.

Example:

```text
a = 0.60
b = 0.25
c = 0.10
d = 0.05
```

Argmax always selects:

```text
a
```

Sampling instead might select:

```text
a → 60%
b → 25%
c → 10%
d → 5%
```

This creates diversity.

---

# 16. Tensor Shape Cheat Sheet

With:

```text
B = 32
block_size = 3
embedding_size = 10
hidden_size = 100
vocab_size = 27
```

the forward pass is:

```text
X

[32, 3]

      ↓

Embedding lookup

[32, 3, 10]

      ↓

Flatten

[32, 30]

      ↓

W1

[30, 100]

      ↓

Hidden

[32, 100]

      ↓

W2

[100, 27]

      ↓

Logits

[32, 27]
```

The single most useful debugging question when working with neural networks is:

> "What is the shape of this tensor right now?"

---

# 17. Complete Concept Map

```text
LANGUAGE MODEL
│
├── Dataset
│
├── Tokenization
│   ├── characters
│   ├── vocabulary
│   ├── stoi
│   └── itos
│
├── Context
│   └── previous N characters
│
├── Embedding
│   └── character ID → learned vector
│
├── MLP
│   ├── Linear
│   ├── Bias
│   └── tanh
│
├── Output
│   └── logits
│
├── Loss
│   └── cross entropy
│
├── Training
│   ├── mini-batch
│   ├── backward
│   ├── gradients
│   └── gradient descent
│
├── Evaluation
│   ├── train
│   ├── validation
│   └── test
│
└── Generation
    ├── softmax
    ├── temperature
    ├── sampling
    ├── next character
    └── autoregressive loop
```

---

# 18. Connection to Modern LLMs

The model here is tiny, but the core idea is surprisingly similar to an autoregressive LLM.

```text
MAKEMORE                  MODERN LLM
------------------------------------------------
character                 token
character ID              token ID
embedding                 embedding
MLP                       Transformer
logits                    logits
softmax                   softmax
sampling                  sampling
next character            next token
autoregressive            autoregressive
```

The major difference is that modern LLMs use much more powerful architectures, especially the **Transformer and self-attention**, and operate over token sequences rather than individual characters.

---

# 19. What to Modify

After the basic version works, experiment with:

```python
embedding_size = 10
hidden_size = 100
batch_size = 32
learning_rate = 0.1
steps = 5000
block_size = 3
```

Try:

```text
embedding_size = 20
hidden_size = 200
```

or:

```text
block_size = 5
```

and observe how the model changes.

---

# 20. Next Learning Step

After understanding this implementation, the next important step is to reproduce more of Karpathy's internals manually:

```text
forward pass
      ↓
manual gradients
      ↓
chain rule
      ↓
dL/dlogits
      ↓
dL/dW2
      ↓
dL/dh
      ↓
gradient through tanh
      ↓
dL/dW1
      ↓
embedding gradients
```

Then move into:

```text
BatchNorm
    ↓
better initialization
    ↓
deeper MLP
    ↓
training diagnostics
    ↓
Transformer
    ↓
self-attention
    ↓
GPT-style language model
```

That is the bridge from this small Makemore model toward understanding modern LLMs from first principles.
