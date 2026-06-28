# Stock Market Direction Prediction from News Headlines

Predicts whether the Dow Jones Industrial Average (DJIA) rose or fell on a
given day using **only that day's news headlines** — no price history, no
technical indicators. The motivating question: most people don't have time to
research individual stocks, but they do read the news. Can the headlines from a
single day predict which way the market moved?

## Approach

The dataset pairs each trading day with ~25 top news headlines and a label for
whether the DJIA rose (`1`) or fell (`0`) that day.

I started with a hand-engineered approach: over 100 "feature functions" that
scored each headline by weighting specific words and phrases (sentiment,
topic keywords, length, etc.), fed into a classifier. This capped out around
**55% accuracy** — barely above chance — which told me that word-level rules
weren't capturing enough of the actual meaning in a headline.

So I rebuilt the pipeline around **semantic embeddings**. Each day's headlines
are encoded into a dense vector with a pretrained sentence transformer
(`paraphrase-MiniLM-L6-v2`), which captures meaning rather than just keyword
presence. A `RandomForestClassifier` is then trained on those embeddings. This
moved test accuracy meaningfully above the feature-function baseline.

## Results

| Approach                          | Test accuracy |
| --------------------------------- |  |
| Hand-engineered feature functions | ~55% (≈ chance) |
| Sentence embeddings +  random forest (unregularized)| 44% — severe overfitting (100% train)|
| Sentence embeddings + random forest (regularized)| ~48% — at chance|

After systematic regularization (limiting tree depth and leaf size, increasing data and tree count), the model stopped overfitting but did not exceed chance on held-out data. This suggests the daily headlines in this dataset carry no reliable signal for next-day DJIA direction — consistent with the efficient-market hypothesis.

## Setup

1. Install Python 3.8+.
2. Install dependencies:
   ```
   pip install scikit-learn sentence-transformers numpy matplotlib tqdm
   ```
3. Download `Combined_News_DJIA.csv` from the
   ["Daily News for Stock Market Prediction" dataset on Kaggle](https://www.kaggle.com/datasets/aaron7sun/stocknews)
   and place it in the project directory.
4. Run:
   ```
   python machineLearning.py
   ```

The first run computes and caches the headline embeddings (the slow step);
later runs reuse the cache and are much faster. Confusion matrices for train
and test performance are displayed on completion.

## How it works

- `get_data()` — reads the CSV, cleans the raw byte-string headlines, and
  returns each day either as one joined string or a list of headlines.
- `run_model()` — embeds the headlines (caching to `.npz`), trains the random
  forest, and reports train/test accuracy with confusion matrices.
- `display_accuracy()` — renders a labeled confusion matrix.
