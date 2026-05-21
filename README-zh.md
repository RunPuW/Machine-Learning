# Steam 游戏个性化推荐系统

基于混合召回 + 大语言模型重排的 Steam 游戏推荐系统。
使用 IDF 加权 ItemKNN 和 SVD 进行候选召回，结合 GPT-4o 驱动的 UUA Agent 和 Ranker Agent 完成语义精排

---

## 项目简介

本系统分为两个阶段：

1. **召回层**：ItemKNN 和 SVD 分别从 37000+ 款游戏中各自召回候选，合并成约 45 个候选游戏
2. **重排层**：UUA Agent 分析用户历史生成偏好画像，Ranker Agent 结合画像对候选进行语义重排，并生成每条推荐的自然语言解释

数据来源：[Kaggle - Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)

---

## 项目结构

```
ml/
├── data/
│   ├── raw/            # 从 Kaggle 下载的原始 CSV 文件（不上传 git）
│   └── processed/      # 预处理后生成的 parquet 文件（自动生成）
│
├── src/
│   ├── data_processing.py      # 第一步：数据清洗、评分构造、时间切分
│   ├── evaluate.py             # 统一评估函数（所有模型共用）
│   ├── popularity_baseline.py  # 全局热门基准模型
│   ├── itemknn.py              # 标准 ItemKNN
│   ├── itemknn_idf.py          # IDF 加权 ItemKNN（最优单模型）
│   ├── svd_batched.py          # 截断 SVD 基准模型
│   ├── svd_knn_overlap.py      # KNN 和 SVD 候选互补性诊断
│   ├── pipeline.py             # 完整 pipeline（KNN+SVD 召回 → UUA → Ranker）
│   ├── uua_agent.py            # 用户理解 Agent（生成偏好画像）
│   ├── ranker_agent.py         # LLM 精排 Agent（语义重排 + 解释生成）
│   ├── cold_start.py           # 冷启动分支（历史不足 5 条的用户）
│   └── ablation_fair.py        # 公平消融实验（同一批用户，四组对比）
│
├── cache/              # 邻居索引和 SVD 因子缓存（自动生成，可复用）
├── results/
│   ├── val/            # 验证集评估结果
│   └── test/           # 测试集评估结果
└── figures/            # EDA 可视化图表
```

---

## 环境安装

```bash
pip install pandas numpy scipy scikit-learn pyarrow tqdm openai python-pptx
```

建议 Python 版本：3.10 及以上

---

## 数据下载

1. 前往 Kaggle 数据集页面下载：
   https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam

2. 解压后将以下三个文件放入 `data/raw/` 目录：

```
data/raw/
├── games.csv
├── users.csv
└── recommendations.csv
```

---

## 完整复现步骤

### 第一步：数据预处理

清洗原始数据，构造评分特征（score_A / score_B），按时间切分为 train / val / test。

```bash
cd ml
python src/data_processing.py
```

运行后会在 `data/processed/` 下生成：
- `interactions_core.parquet`
- `train_interactions.parquet`
- `val_interactions.parquet`
- `test_interactions.parquet`
- `games_cleaned.csv`

---

### 第二步：召回层基准模型

按顺序运行，结果保存在 `results/val/`。

```bash
# 全局热门基准
python src/popularity_baseline.py

# 标准 ItemKNN（比较 score_A 和 score_B，K=50/100/200）
python src/itemknn.py

# IDF 加权 ItemKNN（最优单模型，K=200/500）
python src/itemknn_idf.py

# 截断 SVD（n_factors=50/100/200）
python src/svd_batched.py

# KNN 和 SVD 候选互补性诊断（分析是否值得融合）
python src/svd_knn_overlap.py
```

---

### 第三步：完整 Pipeline（需要 LLM API）

设置 OpenRouter API Key（使用 GPT-4o）：

```bash
# Windows
set OPENROUTER_API_KEY=你的key

# Mac/Linux
export OPENROUTER_API_KEY=你的key
```

运行完整推荐链路：

```bash
# 消融实验：ML only（不调用 LLM，可跑大样本）
python src/pipeline.py --mode ml_only --n_users 500

# 消融实验：Ranker only（fusion 候选 + Ranker，不用 UUA）
python src/pipeline.py --mode ranker_only --n_users 100

# 完整链路：fusion + UUA + Ranker
python src/pipeline.py --mode full --n_users 100
```

---

### 第四步：公平消融实验

在同一批 50 个用户上，分四组对比各组件贡献。

```bash
python src/ablation_fair.py
```

四组对比：
- **A**：纯 KNN 直出
- **B**：KNN+SVD 融合，无 LLM
- **C**：融合 + Ranker（无 UUA）
- **D**：融合 + UUA + Ranker（完整系统）

---

### 第五步：冷启动评估

适用于训练历史不足 5 条的用户，在 test 集上单独评估。

```bash
# 伪验证协议（不花 API，用于调参）
python src/cold_start.py --mode pseudo_val --truncate 0 --n_users 200
python src/cold_start.py --mode pseudo_val --truncate 4 --n_users 100

# 真实冷启动评估（调用 LLM）
python src/cold_start.py --mode evaluate --n_users 500
```

---

## 主要实验结果

### 召回层（验证集，1,037,090 用户）

| 模型 | NDCG@10 | Recall@10 | Hit@10 |
|---|---|---|---|
| Popularity | 0.0170 | 0.0322 | 0.0430 |
| SVD score_A, n=100 | 0.0344 | 0.0609 | 0.0815 |
| ItemKNN score_B, K=200 | 0.0550 | 0.0960 | 0.1224 |
| **IDF-KNN score_B, K=500** | **0.0577** | **0.1014** | **0.1295** |

### 冷启动（测试集，500 用户）

| 方法 | NDCG@10 | Recall@10 |
|---|---|---|
| Popularity | 0.0178 | 0.034 |
| Level 2 LLM | 0.0235 | 0.040 |

---

## 注意事项

- `cache/` 目录下的 `.npz` 文件为自动生成的邻居索引缓存，首次运行后可复用，无需重复计算
- 所有模型的调参均在验证集（val）上进行，测试集（test）仅用于最终报告，不参与调参
- LLM 相关实验（pipeline、cold_start、ablation_fair 的 C/D 组）需要有效的 OpenRouter API Key
- 预计 LLM 费用：完整消融约 $2，冷启动 500 用户约 $4

GitHub: https://github.com/RunPuW/Machine-Learning
