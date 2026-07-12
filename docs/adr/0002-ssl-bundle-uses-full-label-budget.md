# SSL Model Bundles export the 100%-label-budget checkpoint

_Note: ADR 0001 is reserved by [PRD 0001](../prd/0001-backend.md) for "Backend owns all preprocessing" (not yet written); this is the next number._

Each SSL variant (Data2Vec, SimCLR, DINOv2) pretrains an encoder, then runs a label-budget sweep — fine-tuning encoder+head jointly at 1%, 5%, 10%, 25%, and 100% of labeled data — to demonstrate SSL's data efficiency. That sweep is the project's core experiment, but it produces five candidate checkpoints per variant with no single one earmarked for serving.

We export only the **100%-budget** fine-tune as that variant's Model Bundle, not whichever budget scored highest on test F1. This keeps every SSL bundle trained on the same full label set as the three supervised bundles (CNN, CNN-LSTM, Transformer), so the App's model picker compares architectures under identical training conditions rather than mixing in a data-efficiency variable. The trade-off: the deployed SSL bundles don't showcase the sweep's actual headline finding (SSL's advantage at low label budgets) — that result lives in MLflow metrics only, not in what the Backend serves.
