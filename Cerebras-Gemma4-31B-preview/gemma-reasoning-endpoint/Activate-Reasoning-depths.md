# Activate Reasoning Depths

`reasoning_depth` controls how many iterative reasoning passes the model takes before committing to a final answer — essentially how many times it revisits and refines its thinking.

With `reasoning_depth: 6`, the process looks roughly like this:

```
Pass 1 — initial understanding: "The question is about X, I need to consider Y and Z..."
Pass 2 — expand: "Building on that, Y connects to A and B..."
Pass 3 — challenge: "But wait, my assumption about A might be wrong because..."
Pass 4 — refine: "Correcting that, the better approach is..."
Pass 5 — synthesize: "Pulling it all together, the key insight is..."
Pass 6 — finalize: "My conclusion is..."
─────────────────────────────────────────
Final answer (what the user sees)
```

Each pass is a full forward pass through the model — so depth 6 is literally 6× the compute of depth 1, but produces significantly more thorough reasoning before the answer lands.
