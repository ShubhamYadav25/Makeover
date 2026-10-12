# Makemore — Start to End Flow

```mermaid
flowchart TD
    A[Names / Text Dataset] --> B[Character Tokenization]
    B --> C[Vocabulary]
    C --> D[stoi: character to ID]
    C --> E[itos: ID to character]

    D --> F[Context Windows]
    F --> G[X: Context IDs]
    F --> H[Y: Next Character ID]

    G --> I[Train / Validation / Test Split]
    H --> I

    I --> J[Embedding Lookup]
    J --> K[Flatten Embeddings]
    K --> L[Linear Layer: W1 + b1]
    L --> M[tanh Activation]
    M --> N[Output Layer: W2 + b2]
    N --> O[Logits]

    O --> P[Cross Entropy Loss]
    H --> P

    P --> Q[Backpropagation]
    Q --> R[Gradients]
    R --> S[Gradient Descent]
    S --> T[Updated Parameters]
    T --> J

    T --> U[Evaluation]
    U --> V[Train Loss]
    U --> W[Validation Loss]
    U --> X[Test Loss]

    T --> Y[Generation]
    Y --> Z[Start Context: ...]
    Z --> AA[Embedding + MLP]
    AA --> AB[Logits]
    AB --> AC[Temperature]
    AC --> AD[Softmax]
    AD --> AE[Sampling]
    AE --> AF[Next Character]
    AF --> AG[Update Context]
    AG --> AA

    AF --> AH{Is '.'?}
    AH -- No --> AG
    AH -- Yes --> AI[Generated Name]
```

## Compact view

```text
                         TRAINING
                            │
                            ▼
                    ┌───────────────┐
                    │     TEXT      │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  TOKENIZATION │
                    │ chars → IDs   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ CONTEXT WINDOW│
                    │ 3 chars → 1   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   EMBEDDING   │
                    │ ID → vector   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     MLP       │
                    │ Linear+tanh   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    LOGITS     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ CROSS ENTROPY │
                    │     LOSS      │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ BACKPROPAGATE │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ UPDATE PARAMS │
                    └───────┬───────┘
                            │
                            └─────────────┐
                                          │
                                          ▼
                                       REPEAT


                        GENERATION
                            │
                            ▼
                       "... ..."
                            │
                            ▼
                         MODEL
                            │
                            ▼
                         LOGITS
                            │
                            ▼
                       TEMPERATURE
                            │
                            ▼
                         SOFTMAX
                            │
                            ▼
                        SAMPLING
                            │
                            ▼
                    NEXT CHARACTER
                            │
                            ▼
                    UPDATE CONTEXT
                            │
                            └─────────────┐
                                          │
                                          ▼
                                        REPEAT
                                          │
                                  "." = STOP
                                          │
                                          ▼
                                   GENERATED NAME
```
