# Steam Game Recommender System

Group project for Machine Learning Application course.

## Data

Download from Kaggle: https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam

Place the raw CSV files in `data/raw/`:
```
data/raw/games.csv
data/raw/users.csv
data/raw/recommendations.csv
```

## Project Structure

```
ml/
├── data/
│   ├── raw/          # downloaded CSV files (not tracked by git)
│   └── processed/    # generated parquet files after preprocessing
├── src/
│   ├── data_processing.py       # Step 1: cleaning, scoring, temporal split
│   ├── evaluate.py              # unified evaluation functions
│   ├── popularity_baseline.py   # popularity baseline
│   ├── itemknn.py               # standard ItemKNN
│   ├── itemknn_idf.py           # IDF-weighted ItemKNN (best single model)
│   ├── svd_batched.py           # truncated SVD baseline
│   ├── svd_knn_overlap.py       # candidate complementarity analysis
│   ├── pipeline.py              # full KNN+SVD+UUA+Ranker pipeline
│   ├── uua_agent.py             # User Understanding Agent
│   ├── ranker_agent.py          # LLM Ranker Agent
│   ├── cold_start.py            # cold-start branch
│   └── ablation_fair.py         # fair ablation experiment
├── cache/            # neighbour index / SVD factor cache (auto-generated)
├── results/
│   ├── val/          # validation results
│   └── test/         # test results
└── figures/          # EDA plots
```

## Reproduction Steps

**1. Install dependencies**
```bash
pip install pandas numpy scipy scikit-learn pyarrow tqdm openai
```

**2. Preprocess data**
```bash
cd ml
python src/data_processing.py
```

**3. Run recall models**
```bash
python src/itemknn_idf.py      # best single model
python src/svd_batched.py      # SVD baseline
python src/svd_knn_overlap.py  # overlap analysis
```

**4. Run full pipeline (requires OpenRouter API key)**
```bash
set OPENROUTER_API_KEY=your_key_here
python src/pipeline.py --mode full --n_users 100
```

**5. Cold-start evaluation**
```bash
python src/cold_start.py --mode evaluate --n_users 500
```

## Notes

- All model caches are saved to `cache/` and reused on subsequent runs.
- Test set results are in `results/test/`. The test set should not be used for any parameter tuning.
- LLM calls use `openai/gpt-4o` via OpenRouter at `temperature=0.0`.
