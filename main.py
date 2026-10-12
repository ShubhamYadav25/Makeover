"""
Run:
    python main.py

Dependencies:
    pip install torch

The model:
    characters
      -> integer IDs
      -> context windows
      -> embeddings
      -> hidden layer (tanh)
      -> logits
      -> cross entropy
      -> backpropagation
      -> gradient descent
      -> generated names
"""

import random
from pathlib import Path

import torch
import torch.nn.functional as F


# ============================================================
# 0. REPRODUCIBILITY
# ============================================================
# Random operations are used for:
# - parameter initialization
# - train/validation/test split
# - mini-batch selection
# - name generation
#
# Setting a seed makes your run more reproducible.
torch.manual_seed(42)
random.seed(42)


# ============================================================
# 1. DATASET
# ============================================================
"""
CONCEPT: LANGUAGE MODELING DATA

What is it?
-------------
A language model learns to predict the next symbol/token from previous
symbols/tokens.

For example, for:

    "anna"

we can train examples such as:

    context       target
    --------------------
    ...           a
    ..a           n
    .an           n
    ann           a
    nna           .

The "." character is our special boundary token.

Why do we need this?
--------------------
A neural network needs input -> expected output pairs.

Here:
    input  = previous characters
    output = next character

This turns a text-generation problem into a supervised learning problem.

Overview:
----------
    words
      ↓
    context windows
      ↓
    X = inputs
    Y = targets
      ↓
    neural network learns P(next_character | context)

Example:
---------
If block_size = 3:

    "anna"

becomes approximately:

    X                    Y
    [., ., .]            a
    [., ., a]            n
    [., a, n]            n
    [a, n, n]            a
    [n, n, a]            .

After tokenization, the characters will be integer IDs instead of
literal characters.
"""


# A small dataset keeps training fast on a normal PC.
# You can replace this with a much larger names.txt later.
words = [
    "anna",
    "andrew",
    "ananya",
    "arjun",
    "aarav",
    "aditya",
    "aman",
    "amit",
    "anil",
    "abhay",
    "akash",
    "alex",
    "alice",
    "amelia",
    "ben",
    "bella",
    "bob",
    "brandon",
    "bruce",
    "charlie",
    "chris",
    "daniel",
    "david",
    "deepak",
    "dev",
    "divya",
    "emma",
    "eric",
    "ethan",
    "farhan",
    "gaurav",
    "george",
    "harry",
    "ishaan",
    "jack",
    "james",
    "jane",
    "john",
    "julia",
    "karen",
    "karan",
    "kavya",
    "kevin",
    "liam",
    "lisa",
    "lucas",
    "maya",
    "meera",
    "michael",
    "mia",
    "neha",
    "nikhil",
    "noah",
    "oliver",
    "olivia",
    "priya",
    "rahul",
    "raj",
    "riya",
    "rohan",
    "rohit",
    "sam",
    "sara",
    "sarah",
    "shreya",
    "sofia",
    "sophia",
    "steve",
    "tanvi",
    "vansh",
    "vikas",
    "vishal",
    "william",
    "zara",
]


# ============================================================
# 2. TOKENIZATION / VOCABULARY
# ============================================================
"""
CONCEPT: TOKENIZATION

What is tokenization?
---------------------
Tokenization converts text into small units that a neural network can
represent numerically.

In this project we use CHARACTER-LEVEL tokenization.

Example:

    "cat"

becomes:

    ['c', 'a', 't']

Then we create a vocabulary:

    ['.', 'a', 'b', 'c', ...]

and assign each character an integer ID.

Example:

    stoi = {
        '.': 0,
        'a': 1,
        'b': 2,
        'c': 3,
        ...
    }

Then:

    "cat"

becomes:

    [3, 1, 20]

The exact IDs depend on the vocabulary.

Why is it needed?
-----------------
PyTorch tensors contain numbers, not strings.

The neural network cannot directly calculate:

    "c" @ W

but it can calculate:

    3 -> embedding vector -> matrix operations

So the pipeline is:

    text
      ↓
    characters
      ↓
    integer IDs
      ↓
    embeddings
      ↓
    neural network

What output should you imagine?
--------------------------------
For a vocabulary such as:

    ['.', 'a', 'b', 'c']

you might have:

    "cab"

    ['c', 'a', 'b']

    [3, 1, 2]

Then the model can work with [3, 1, 2].

We also create the reverse mapping:

    itos = integer -> string

so:

    3 -> 'c'
    1 -> 'a'
    2 -> 'b'

This is required during generation because the model predicts an integer
ID, and we need to convert it back into a character.
"""


# "." is the special START/END token.
# We sort characters so the mapping is deterministic.
chars = sorted(list(set("".join(words) + ".")))

# string -> integer
stoi = {ch: i for i, ch in enumerate(chars)}

# integer -> string
itos = {i: ch for ch, i in stoi.items()}

vocab_size = len(chars)

print("=" * 70)
print("VOCABULARY / TOKENIZATION")
print("=" * 70)
print("Vocabulary:", chars)
print("Vocabulary size:", vocab_size)
print("Example encoding of 'anna':",
      [stoi[ch] for ch in "anna"])
print("Example decoding:",
      "".join(itos[i] for i in [stoi[ch] for ch in "anna"]))
print()


# ============================================================
# 3. CREATE CONTEXT/TARGET DATASET
# ============================================================
"""
CONCEPT: CONTEXT WINDOW

What is a context window?
-------------------------
The model does not see an entire word and magically know the answer.

We explicitly give it the previous N characters.

Here:

    block_size = 3

means:

    "look at the previous 3 characters"

Example for "anna":

    ... -> a
    ..a -> n
    .an -> n
    ann -> a
    nna -> .

Why?
----
This creates the learning task:

    previous characters -> next character

X contains contexts.
Y contains the correct next character.

Example after integer encoding:

    X = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 14],
        ...
    ]

    Y = [
        1,
        14,
        14,
        ...
    ]

The model's job is:

    X[i] -> Y[i]

Important:
---------
The model will ALWAYS receive exactly block_size characters.

That is why shorter contexts are padded with ".".
"""


block_size = 3

X = []
Y = []

for word in words:

    # Start each word with:
    #
    # [., ., .]
    #
    # Example:
    #
    # "anna"
    #
    # context starts as:
    # [., ., .]
    context = [stoi["."]] * block_size

    # Add "." after the word to teach the model where the word ends.
    #
    # "anna" -> "anna."
    for ch in word + ".":

        # Current context is the input.
        X.append(context)

        # Current character is the expected answer.
        Y.append(stoi[ch])

        # Slide the context window forward.
        #
        # If:
        #
        # [., ., .] -> a
        #
        # next context becomes:
        #
        # [., ., a]
        context = context[1:] + [stoi[ch]]


X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)

print("=" * 70)
print("CONTEXT DATASET")
print("=" * 70)
print("X shape:", tuple(X.shape))
print("Y shape:", tuple(Y.shape))

print("\nFirst 5 examples:")
for i in range(min(5, len(X))):
    context_text = "".join(itos[int(j)] for j in X[i])
    target_text = itos[int(Y[i])]
    print(f"  {context_text!r} -> {target_text!r}")
print()


# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================
"""
CONCEPT: TRAIN / VALIDATION / TEST

Why split the data?
-------------------
If we train and evaluate on exactly the same examples, the model can
memorize them.

We therefore separate examples:

    training set
        ↓
    used to update parameters

    validation set
        ↓
    used to check generalization during development

    test set
        ↓
    final unbiased-ish evaluation

Typical idea:

    80% train
    10% validation
    10% test
"""


N = len(X)

generator = torch.Generator().manual_seed(42)
perm = torch.randperm(N, generator=generator)

n_train = int(0.8 * N)
n_val = int(0.1 * N)

train_idx = perm[:n_train]
val_idx = perm[n_train:n_train + n_val]
test_idx = perm[n_train + n_val:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

print("=" * 70)
print("DATA SPLIT")
print("=" * 70)
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))
print()


# ============================================================
# 5. MODEL ARCHITECTURE / HYPERPARAMETERS
# ============================================================
"""
CONCEPT: HYPERPARAMETERS

Hyperparameters are choices WE make before/during training.

Examples:
    block_size
    embedding_size
    hidden_size
    learning_rate
    batch_size
    number of training steps

They are not learned by gradient descent.

Our architecture:

    character ID
         ↓
    embedding vector
         ↓
    concatenate 3 embeddings
         ↓
    linear layer
         ↓
    tanh
         ↓
    linear layer
         ↓
    logits for every character

Example shapes:

    X       = [B, 3]

    C[X]    = [B, 3, 10]

    flatten = [B, 30]

    hidden  = [B, 100]

    logits  = [B, vocab_size]

where:
    B = batch size
"""


embedding_size = 10
hidden_size = 100
batch_size = 32

# Learning rate controls how large each gradient-descent step is.
learning_rate = 0.1

# Number of parameter updates.
steps = 5000


# ============================================================
# 6. MODEL PARAMETERS
# ============================================================
"""
CONCEPT: PARAMETERS

These are the numbers the model actually LEARNS.

C:
    embedding table

W1, b1:
    hidden layer

W2, b2:
    output layer

Training repeatedly changes these values to reduce loss.
"""


# ------------------------------------------------------------
# Embedding table
# ------------------------------------------------------------
"""
C shape:

    [vocab_size, embedding_size]

Example:

    vocabulary = 27 characters
    embedding_size = 10

    C = [27, 10]

If:

    character ID = 3

then:

    C[3]

returns a 10-dimensional vector.

This is the learned representation of that character.
"""

C = torch.randn(
    vocab_size,
    embedding_size
)


# ------------------------------------------------------------
# First linear layer
# ------------------------------------------------------------
"""
We have:

    block_size = 3
    embedding_size = 10

So after concatenating 3 embeddings:

    3 × 10 = 30 values

Therefore W1:

    [30, hidden_size]

The hidden layer learns combinations of the characters in the context.
"""

W1 = torch.randn(
    block_size * embedding_size,
    hidden_size
) * 0.1

b1 = torch.zeros(hidden_size)


# ------------------------------------------------------------
# Output layer
# ------------------------------------------------------------
"""
The hidden layer has hidden_size neurons.

We need one output score (logit) for EVERY possible next character.

Therefore:

    W2 = [hidden_size, vocab_size]

Example:

    hidden_size = 100
    vocab_size = 27

    W2 = [100, 27]

The result is 27 logits.

One logit per possible next character.
"""

W2 = torch.randn(
    hidden_size,
    vocab_size
) * 0.1

b2 = torch.zeros(vocab_size)


parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True


print("=" * 70)
print("MODEL")
print("=" * 70)
print("C :", tuple(C.shape))
print("W1:", tuple(W1.shape))
print("b1:", tuple(b1.shape))
print("W2:", tuple(W2.shape))
print("b2:", tuple(b2.shape))

total_parameters = sum(p.numel() for p in parameters)
print("Total trainable parameters:", total_parameters)
print()


# ============================================================
# 7. FORWARD PASS
# ============================================================
"""
CONCEPT: FORWARD PASS

The forward pass means:

    input
      ↓
    calculate prediction
      ↓
    calculate loss

For one batch:

    X
    ↓
    embedding lookup
    ↓
    flatten
    ↓
    linear + tanh
    ↓
    linear
    ↓
    logits
    ↓
    cross entropy
    ↓
    loss

The model has not learned anything from the current batch until
backpropagation + parameter update happen.
"""


def forward(xb):
    """
    Run the model forward.

    Input:
        xb shape = [B, block_size]

    Example:
        [32, 3]

    Output:
        logits shape = [B, vocab_size]

    Example:
        [32, 27]
    """

    # --------------------------------------------------------
    # EMBEDDING LOOKUP
    # --------------------------------------------------------
    #
    # xb:
    #     [B, 3]
    #
    # C:
    #     [vocab_size, embedding_size]
    #
    # C[xb]:
    #     [B, 3, embedding_size]
    #
    # Example:
    #     [32, 3, 10]
    #
    emb = C[xb]

    # --------------------------------------------------------
    # FLATTEN
    # --------------------------------------------------------
    #
    # [B, 3, 10]
    #
    # becomes:
    #
    # [B, 30]
    #
    # because:
    #
    # 3 contexts × 10 dimensions = 30
    #
    emb = emb.view(
        emb.shape[0],
        -1
    )

    # --------------------------------------------------------
    # HIDDEN LAYER
    # --------------------------------------------------------
    #
    # emb @ W1 + b1
    #
    # [B, 30] @ [30, 100]
    #
    # -> [B, 100]
    #
    # tanh introduces NON-LINEARITY.
    #
    # Without a non-linearity, stacking linear operations would still
    # effectively behave like one linear transformation.
    #
    h = torch.tanh(
        emb @ W1 + b1
    )

    # --------------------------------------------------------
    # OUTPUT LAYER
    # --------------------------------------------------------
    #
    # [B, 100] @ [100, vocab_size]
    #
    # -> [B, vocab_size]
    #
    # These are LOGITS.
    #
    # They are raw scores, NOT probabilities.
    #
    logits = h @ W2 + b2

    return logits


# ============================================================
# 8. CROSS ENTROPY / LOSS
# ============================================================
"""
CONCEPT: LOGITS AND CROSS ENTROPY

Suppose the model outputs:

    logits =
    [2.1, 0.5, -1.0, 3.7, ...]

These are raw scores.

Softmax conceptually converts them to probabilities:

    [0.12, 0.02, 0.01, 0.80, ...]

If the correct character is index 3, we want high probability at index 3.

Cross entropy penalizes the model when it gives low probability to the
correct answer.

Important:
----------
PyTorch's F.cross_entropy expects RAW LOGITS.

Do NOT do:

    softmax(logits)
    F.cross_entropy(probabilities, target)

Instead:

    F.cross_entropy(logits, target)

PyTorch internally performs a numerically stable log-softmax operation.
"""


def calculate_loss(xb, yb):
    logits = forward(xb)

    # logits:
    #     [B, vocab_size]
    #
    # yb:
    #     [B]
    #
    # cross_entropy returns one scalar loss.
    loss = F.cross_entropy(logits, yb)

    return loss


# ============================================================
# 9. TRAINING LOOP
# ============================================================
"""
CONCEPT: BACKPROPAGATION + GRADIENT DESCENT

One training step:

    1. Pick a random mini-batch
    2. Forward pass
    3. Calculate loss
    4. Calculate gradients with backward()
    5. Update parameters
    6. Repeat

Mathematically:

    parameter = parameter - learning_rate * gradient

If:

    gradient > 0

we decrease the parameter.

If:

    gradient < 0

we increase the parameter.

The gradient tells us which direction increases/decreases loss.
"""


print("=" * 70)
print("TRAINING")
print("=" * 70)

for step in range(steps):

    # --------------------------------------------------------
    # MINI-BATCH
    # --------------------------------------------------------
    #
    # Instead of using every example at once, select random examples.
    #
    # If batch_size = 32:
    #
    # xb -> [32, 3]
    # yb -> [32]
    #
    # Mini-batches make training faster and add useful stochasticity.
    #
    ix = torch.randint(
        0,
        len(X_train),
        (batch_size,)
    )

    xb = X_train[ix]
    yb = Y_train[ix]

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------
    loss = calculate_loss(xb, yb)

    # --------------------------------------------------------
    # CLEAR OLD GRADIENTS
    # --------------------------------------------------------
    #
    # PyTorch accumulates gradients by default.
    #
    # Therefore every training iteration must clear the old gradients.
    #
    for p in parameters:
        p.grad = None

    # --------------------------------------------------------
    # BACKPROPAGATION
    # --------------------------------------------------------
    #
    # PyTorch walks backward through the computation graph and computes:
    #
    #     dLoss/dC
    #     dLoss/dW1
    #     dLoss/db1
    #     dLoss/dW2
    #     dLoss/db2
    #
    loss.backward()

    # --------------------------------------------------------
    # GRADIENT DESCENT
    # --------------------------------------------------------
    #
    # torch.no_grad() means:
    #
    # "Do not build another autograd graph for this parameter update."
    #
    with torch.no_grad():

        for p in parameters:
            p -= learning_rate * p.grad

    if step % 500 == 0:
        print(
            f"step={step:5d} | "
            f"train_loss={loss.item():.4f}"
        )


# ============================================================
# 10. EVALUATION
# ============================================================
"""
CONCEPT: EVALUATION

After training, we calculate loss on:

    train
    validation
    test

We do NOT update parameters during evaluation.

torch.no_grad():
    saves memory
    avoids unnecessary gradient calculation
"""


def evaluate(X_data, Y_data):
    with torch.no_grad():
        return calculate_loss(X_data, Y_data).item()


train_loss = evaluate(X_train, Y_train)
val_loss = evaluate(X_val, Y_val)
test_loss = evaluate(X_test, Y_test)

print()
print("=" * 70)
print("FINAL EVALUATION")
print("=" * 70)
print(f"Train loss:      {train_loss:.4f}")
print(f"Validation loss: {val_loss:.4f}")
print(f"Test loss:       {test_loss:.4f}")
print()


# ============================================================
# 11. GENERATION
# ============================================================
"""
CONCEPT: AUTOREGRESSIVE GENERATION

Training:

    context -> correct next character

Generation:

    context -> predict next character
                      ↓
                selected character
                      ↓
                becomes context
                      ↓
                   repeat

Example:

    ... -> a
    ..a -> n
    .an -> i
    ani -> l
    nil -> .

    generated name = "anil"

The model generates ONE character at a time.

This is called AUTOREGRESSIVE generation.

The same high-level idea is used by modern autoregressive LLMs:

    previous tokens
          ↓
       predict
          ↓
    next token
          ↓
    append token
          ↓
       repeat
"""


def generate_name(temperature=1.0, max_length=20):
    """
    Generate one new name.

    temperature:
        < 1.0 -> more deterministic
        = 1.0 -> normal sampling
        > 1.0 -> more random

    max_length:
        safety limit so a broken/random model cannot generate forever.
    """

    # Start with block_size boundary tokens.
    context = [stoi["."]] * block_size

    generated = []

    # We are only doing inference.
    # There is no reason to calculate gradients.
    with torch.no_grad():

        for _ in range(max_length):

            # Convert context to a batch of one example.
            #
            # [block_size]
            #      ↓
            # [1, block_size]
            #
            X_new = torch.tensor(
                [context],
                dtype=torch.long
            )

            # Run model.
            logits = forward(X_new)

            # ------------------------------------------------
            # TEMPERATURE
            # ------------------------------------------------
            #
            # logits / temperature
            #
            # Low temperature:
            #     differences between logits become stronger
            #     -> sharper distribution
            #
            # High temperature:
            #     differences become weaker
            #     -> flatter distribution
            #
            scaled_logits = logits / temperature

            # Convert logits to probabilities.
            probs = F.softmax(
                scaled_logits,
                dim=1
            )

            # ------------------------------------------------
            # SAMPLE
            # ------------------------------------------------
            #
            # Instead of always taking argmax, sample according
            # to the probability distribution.
            #
            # Example:
            #
            # a = 0.60
            # b = 0.25
            # c = 0.10
            # d = 0.05
            #
            # Sampling can select:
            #     a most often
            #     b sometimes
            #     c occasionally
            #     d rarely
            #
            ix = torch.multinomial(
                probs,
                num_samples=1
            ).item()

            # "." means END OF WORD.
            if ix == stoi["."]:
                break

            # Convert integer ID back to character.
            generated.append(itos[ix])

            # Slide context window.
            #
            # Old:
            #     [., ., a]
            #
            # New character:
            #     n
            #
            # New:
            #     [., a, n]
            #
            context = context[1:] + [ix]

    return "".join(generated)


# ============================================================
# 12. GENERATE EXAMPLES
# ============================================================

print("=" * 70)
print("GENERATED NAMES")
print("=" * 70)

for temperature in [0.7, 1.0, 1.3]:

    print(f"\nTemperature = {temperature}")

    for _ in range(10):
        print("  ", generate_name(temperature=temperature))

print()


# ============================================================
# 13. SHAPE / FLOW SANITY CHECK
# ============================================================
"""
CONCEPT: TENSOR SHAPES

For batch_size B=4, block_size=3, embedding_size=10,
hidden_size=100 and vocab_size=27:

    X
    [4, 3]

       ↓ embedding lookup

    [4, 3, 10]

       ↓ flatten

    [4, 30]

       ↓ W1 [30, 100]

    [4, 100]

       ↓ W2 [100, 27]

    [4, 27]

    logits

This is one of the most important things to understand when reading
neural-network code.
"""


with torch.no_grad():

    demo_x = X_train[:4]

    demo_emb = C[demo_x]
    demo_flat = demo_emb.view(demo_emb.shape[0], -1)
    demo_h = torch.tanh(demo_flat @ W1 + b1)
    demo_logits = demo_h @ W2 + b2

print("=" * 70)
print("TENSOR SHAPE SANITY CHECK")
print("=" * 70)
print("X:       ", tuple(demo_x.shape))
print("Embedding:", tuple(demo_emb.shape))
print("Flatten:  ", tuple(demo_flat.shape))
print("Hidden:   ", tuple(demo_h.shape))
print("Logits:   ", tuple(demo_logits.shape))
print()

print("=" * 70)
print("DONE Makemore MLP pipeline completed.")
print("=" * 70)
