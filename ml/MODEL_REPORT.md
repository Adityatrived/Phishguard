# PhishGuard — Model Comparison Report

Trained on `dataset.csv` (130k URLs, balanced), 80/20 stratified train/test split, seed=42.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.8571 | 0.8562 | 0.8583 | 0.8573 |
| Random Forest | 0.9471 | 0.9630 | 0.9299 | 0.9462 |
| XGBoost | 0.9528 | 0.9663 | 0.9382 | 0.9521 |

**Winner: XGBoost** (highest F1 on the held-out test set).

Confusion matrix `[[TN, FP], [FN, TP]]`: [[12575, 425], [803, 12197]]

Top features for the winning model:

- `is_https`: 0.2612
- `brand_edit_distance`: 0.1058
- `is_url_shortener`: 0.0683
- `has_at_symbol`: 0.0647
- `longest_digit_run`: 0.0587
- `num_subdomains`: 0.0537
- `hostname_length`: 0.0519
- `path_length`: 0.0400
