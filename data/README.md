# Public dataset

This folder contains the curated public-facing artifacts derived from the project's raw data.

Files included:

- `cleaned_public_dataset.csv` — cleaned CSV with only `text`, `label`, `source` (suitable for public release after PII removal).
- `tweet_ids.csv` — mapping of `tweet_id,label` for the Twitter dump. Use these IDs to hydrate tweets via the Twitter/X API rather than distributing tweet text.

Purpose:

- Provide a privacy-conscious dataset for research while preserving reproducibility via tweet IDs for Twitter-derived content.

Redistribution notes:

- The repository does not include full raw Twitter dumps. Hydrating tweet IDs must follow Twitter/X developer policies.
