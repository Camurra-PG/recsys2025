# recsys2025 — Retailrocket Evaluation Harness

Scores Universal Behavioral Profiles on downstream tasks: a small classifier is trained per task on top of frozen user embeddings, then evaluated on a held-out window.

Fork of the official RecSys Challenge 2025 harness ([Synerise/recsys2025](https://github.com/Synerise/recsys2025)), extended to run on the [Retailrocket dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset). Retailrocket ships no evaluation targets, so they are constructed here.

Profiles are produced by a separate repository: [Replication-of-Beyond-Model-Size](https://github.com/Camurra-PG/Replication-of-Beyond-Model-Size--Narrative-Driven-Universal-Modeling).

Upstream documentation on the Challenge entry format and model architecture is kept in [`README_UPSTREAM.md`](README_UPSTREAM.md).

## Structure

```
.
├── build_retailrocket_eval_dataset.py   # builds the full target structure from raw Retailrocket CSVs
├── training_pipeline/
│   ├── train.py                         # training entry point
│   ├── data_module.py                   # train/validation split by client ID
│   ├── target_calculators.py            # churn / active / propensity labels
│   ├── metric_calculators.py            # Macro-AUROC, Micro-AUROC, Hit-Rate@K
│   └── model.py                         # task head (unchanged from upstream)
├── data_utils/
│   └── split_data.py                    # upstream time-window splitter (Challenge data only)
├── validator/
│   └── run.py                           # upstream entry-format validator
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/Camurra-PG/recsys2025.git
cd recsys2025
pip install -r requirements.txt
```

Datasets are not included — get Retailrocket from [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

## Usage

### 1. Build the evaluation targets

`--target-days` counts backwards from the dataset's last event (2015-09-18) to place the observation cutoff. `--min-history-events` is the minimum history a client needs to be included at all.

```bash
python build_retailrocket_eval_dataset.py \
  --raw-dir  data/retailrocket_raw \
  --out-dir  data/retailrocket_eval \
  --target-days 75 \
  --min-history-events 5 \
  --valid-ratio 0.2 \
  --seed 42
```

Produces:

```
data/retailrocket_eval/
├── product_properties.parquet      # empty dummy — DataDir validation requires the file
├── input/
│   └── relevant_clients.npy        # 38,547 clients
└── target/
    ├── train_target.parquet        # churn / active / propensity columns
    ├── validation_target.parquet
    ├── active_clients.npy
    ├── propensity_category.npy
    ├── propensity_sku.npy
    └── propensity_new_sku.npy
```

The split is stratified: clients carrying a propensity label are divided separately from those without, so rare classes cannot land entirely on one side by chance.

### 2. Train and score

```bash
python -m training_pipeline.train \
  --data-dir       data/retailrocket_eval \
  --embeddings-dir embeddings/qwen3-8b \
  --tasks churn active propensity_category \
  --log-name qwen_rr_seed42 \
  --accelerator gpu --devices 0 \
  --disable-relevant-clients-check
```

### 3. Run several seeds

With ~21 positive churn examples in the validation fold, one seed is not a point estimate — observed churn AUROC for the same model ranges from 0.32 to 0.84 across seeds.

```bash
for s in 42 7 123 2024; do
  python -m training_pipeline.train \
    --data-dir data/retailrocket_eval \
    --embeddings-dir embeddings/qwen3-8b \
    --tasks churn active --seed $s \
    --log-name qwen_rr_seed$s \
    --accelerator gpu --devices 0 \
    --disable-relevant-clients-check
done
```

## Differences from upstream

Retailrocket breaks several assumptions the upstream harness makes about the Challenge data.

| | Challenge | Retailrocket |
|---|---|---|
| Churn labels | implicit, via presence in a target table | explicit nullable column, history buyers only |
| Propensity labels | one row per purchase event | one row per client, list-valued columns |
| `active_clients.npy` | clients with ≥1 purchase in history | clients with ≥1 event in the target window |
| Train/validation | two consecutive time windows | 80/20 client-level split |
| `product_properties.parquet` | real metadata | absent; dummy file required |

The `active_clients.npy` row is the one to watch: same filename, different meaning. The churn population is therefore filtered on `churn.notna()` instead.

Two metrics were added next to the upstream Macro-AUROC. **Micro-AUROC** pools all (client, class) pairs before scoring, so each class counts in proportion to its examples; a large Macro–Micro gap means the Macro figure rests on classes too sparse to support it. **Hit-Rate@K** (K=3) records whether a client's true class appears in the top 3 — with eight classes a random ranker scores 0.375.

## Task coverage

What can be evaluated is limited by positive examples in the validation fold, not by profile quality.

| Task | Labeled | Validation | Status |
|---|---|---|---|
| Churn | 3,585 | ~700 | robust |
| Active | 5,481 | ~1,095 | robust |
| Category propensity | ~69 | ~9–14 | borderline |
| SKU propensity | ~26 | ~1 | not robust |
| New-SKU propensity | ~18 | 0 | not determinable |
| Conversion, propensity price | — | — | no Retailrocket equivalent |

Lowering `--min-history-events` to admit 188,704 clients raises category coverage from ~11 to ~12 labeled validation clients, since the clients it adds are the least active ones.

## Licence

Upstream licence applies to the inherited code. Datasets keep their own terms and are not redistributed here.
