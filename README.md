# Virada AI — Financial Sentiment & Market Signals

Live site: https://financial-project-phi.vercel.app/

Virada AI classifies financial news as **negative / neutral / positive** with a custom neural network and turns that sentiment into market signals for US and Kazakhstan stocks (KSPI, HSBK, KAP, NVDA, AAPL, TSLA and others).

## Repository layout

| Path | What it is |
|------|------------|
| `index.html`, `attention.js`, `evidence.js` | Static website (deployed on Vercel, no build step) |
| `plots/` | Training curves, confusion matrix, ablation and price-prediction charts used on the site |
| `*_analysis.json` | Per-ticker analysis data |
| `train_big_neural_sentiment.py` | Training script: embeddings → CNN → BiGRU → attention pooling → classifier |
| `predict_big_neural_sentiment.py`, `predict_demo.py` | Inference |
| `train.py`, `financial_analysis.py` | Training pipeline and text preprocessing |
| `backtest.py`, `backtest_results.json`, `backtest_chart.png` | Strategy backtest |
| `build_site_data.py`, `build_multilingual_data.py`, `apply_multilingual_patch.py` | Data prep for the site and EN/RU/KK support |
| `fix_*.py` | Small maintenance scripts |

## Datasets

1. Financial PhraseBank
2. Twitter financial news sentiment

## Results

Accuracy ≈ 75%, weighted F1 ≈ 0.74 (custom network).

## Run

```bash
pip install torch datasets scikit-learn tqdm joblib numpy nltk
python train_big_neural_sentiment.py
python predict_big_neural_sentiment.py
```

Website locally: open `index.html` or run `python -m http.server` in the repo root.

Model weights (`*.pt`, `*.joblib`) and raw data are not committed (see `.gitignore`).

## Author

Zhanibek
