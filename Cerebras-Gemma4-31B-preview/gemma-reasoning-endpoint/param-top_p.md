# Parameter: top_p (Nucleus Sampling)

`top_p` controls how the model selects the next token during generation.

## How it works

At each generation step, the model ranks all tokens by probability from highest to lowest, then keeps only the smallest set of tokens whose cumulative probability adds up to `p`. Everything outside that set is excluded from sampling.

**Example with `top_p: 0.9`:**
- The model sums token probabilities until it hits 90%
- Only those tokens are candidates for the next word
- The remaining 10% (low-probability, often weird or off-topic tokens) are cut off

## Practical effect

| Value | Behavior |
|---|---|
| `1.0` | All tokens are candidates — no filtering |
| `0.9` | Cuts the long tail of unlikely tokens, keeps output coherent |
| `0.5` | Very conservative — only high-confidence tokens survive |

## vs temperature

- **`temperature`** scales the probability distribution — higher values spread it out, making generation more creative and varied
- **`top_p`** filters which tokens are even eligible *after* that scaling

They operate in sequence: temperature reshapes the distribution first, then `top_p` cuts off the bottom of it.

## Usage in this project

```python
"parameters": {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,        # cuts low-probability tail after temperature scaling
    "enable_reasoning": True,
    "reasoning_depth": 6,
}
```

Set in `test_reasoning.py` Test 4 (multimodal) and available across all payloads.
