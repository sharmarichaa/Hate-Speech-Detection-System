# Hate Speech Detection on Twitter

Classifies tweets into three categories using classic NLP and machine learning:

- **Hate speech**
- **Offensive language**
- **Neither**

> **Content warning:** the dataset and some examples contain slurs and offensive language.

## Overview

The project cleans raw tweets, converts them to Bag-of-Words features, trains and compares four scikit-learn models, and evaluates them with metrics suited to an imbalanced dataset (macro-F1, per-class precision and recall).

## Dataset

- **Source:** Davidson et al. (2017), *Automated Hate Speech Detection and the Problem of Offensive Language* ([original repository](https://github.com/t-davidson/hate-speech-and-offensive-language)).
- **Size:** 24,783 tweets.
- **Labels:** 0 = hate speech (1,430), 1 = offensive language (19,190), 2 = neither (4,163).
- **Imbalance:** hate speech is only about 5.8% of the data, so accuracy alone is misleading.

The file `twitter.csv` is the labeled dataset. Check the original repository's license before redistributing it.

## Method

1. **Cleaning:** lowercase; remove URLs, @mentions, HTML entities, the `rt` marker, punctuation, and words containing digits; remove English stopwords; Snowball stemming.
2. **Split:** 67% train / 33% test, stratified by class.
3. **Features:** `CountVectorizer` (Bag-of-Words), fit on training data only through a scikit-learn `Pipeline`, so there is no data leakage.
4. **Models compared:** Decision Tree (baseline), Decision Tree (balanced), Logistic Regression (balanced), Multinomial Naive Bayes.
5. **Selection:** best model by macro-F1.

## Results

Test set of 8,179 tweets:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Logistic Regression (balanced) | 0.868 | 0.732 |
| Decision Tree (balanced) | 0.863 | 0.701 |
| Decision Tree (baseline) | 0.879 | 0.696 |
| Multinomial Naive Bayes | 0.873 | 0.590 |

Best model (Logistic Regression), per class:

| Class | Precision | Recall |
|---|---|---|
| Hate speech | 0.32 | 0.57 |
| Offensive language | 0.96 | 0.88 |
| Neither | 0.81 | 0.93 |

Exact numbers may vary slightly by library version. The notebook prints the full report.

## Project structure

```
.
├── hate_speech_detection.ipynb   # full pipeline
├── twitter.csv                   # dataset
├── requirements.txt
└── README.md
```

## Setup and usage

```bash
git clone <your-repo-url>
cd <your-repo-name>
python -m venv venv
venv\Scripts\activate        # Windows (use: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
jupyter notebook hate_speech_detection.ipynb
```

Run all cells. The first run downloads the NLTK stopword list (internet required). The notebook saves the best model as `hate_speech_model.joblib`.

Predict on new text with the notebook's helper:

```python
predict_text("have a great day everyone")
# -> 'no hate speech or offensive language'
```

## Limitations

- Bag-of-Words ignores word order and context (sarcasm, reclaimed slurs, quoted speech).
- Hate speech is a small minority class, so its precision and recall are much lower than the other classes.
- The data is from 2017 and English only; language and slang drift will reduce real-world performance.
- Labels are crowd-sourced and noisy.
- **Do not use this model for automated moderation without human review.**

## Possible improvements

- TF-IDF and n-gram features.
- Transformer models (e.g., BERT / HateBERT) fine-tuned on this data.
- Resampling or threshold tuning to improve hate-speech recall.
- A more recent, multilingual dataset.

## Reference

Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language.* Proceedings of ICWSM.
