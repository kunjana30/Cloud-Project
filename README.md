# Credit Risk Modelling with Synthetic Data on Google Cloud

Can synthetic data replace or augment real customer data when training credit default models? This project answers that question with an end-to-end workflow on **Google Cloud (Cloud Storage and Vertex AI)** and **Google Colab**: it generates synthetic credit data, evaluates it, trains models under four scenarios, explains and fairness-checks them, and deploys the best model as a prediction endpoint.

> University project (Cloud Computing).

---

## Table of Contents
1. [Motivation](#motivation)
2. [Research Questions](#research-questions)
3. [Dataset](#dataset)
4. [Architecture](#architecture)
5. [Methodology](#methodology)
6. [Repository Structure](#repository-structure)
7. [Requirements](#requirements)
8. [Setup and Usage](#setup-and-usage)
9. [Configuration Reference](#configuration-reference)
10. [Outputs](#outputs)
11. [Results](#results)
12. [Cost Control and Cleanup](#cost-control-and-cleanup)
13. [Troubleshooting](#troubleshooting)
14. [Limitations](#limitations)
15. [Future Work](#future-work)
16. [References](#references)
17. [License and Author](#license-and-author)

---

## Motivation

Credit risk models estimate the probability that a borrower will fail to repay. Building them well is hard for three reasons:

- **Privacy.** Customer records are protected by regulation (for example GDPR) and by internal data-sharing rules, which makes moving real data to vendors, universities or cloud environments slow and risky.
- **Class imbalance.** Defaulters are a minority. Models trained on imbalanced data tend to miss the very customers they are meant to flag.
- **Fairness and auditability.** Credit decisions must be explainable and must not discriminate on attributes such as sex.

Synthetic data, meaning artificial records that imitate the statistical patterns of real data without corresponding to real individuals, is a promising way to address the first two problems. This project tests how well it works in practice, and what it costs in accuracy, privacy and fairness.

## Research Questions

1. **RQ1:** How closely does a model trained only on synthetic data approach one trained on real data (train on synthetic, test on real)?
2. **RQ2:** Does adding synthetic defaulters to real data beat real-only training and the classical SMOTE oversampling technique?
3. **RQ3:** Which generator gives the best balance of utility and privacy?
4. **RQ4:** Does synthetic data reduce or amplify gender-based differences in model outcomes?

## Dataset

**UCI "Default of Credit Card Clients"** (Yeh and Lien, 2009).

| Property | Value |
|---|---|
| Customers | 30,000 |
| Predictors | 23 (credit limit, sex, education, marital status, age, 6 months of repayment status `PAY_0`-`PAY_6`, 6 months of bill amounts, 6 months of payment amounts) |
| Target | `default` (1 = defaults next month) |
| Default rate | about 22% |
| Train / test split | 21,000 / 9,000 (stratified 70/30) |

**The dataset is not included in this repository.** Download it from the UCI Machine Learning Repository (search for "Default of Credit Card Clients") and check its license terms. The file contains an extra label row (`X1`...`X23`, `Y`) above the real column names; the notebook detects and handles this automatically, so you can use the file as downloaded.

## Architecture

```
                        Google Cloud project
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │   Cloud Storage bucket                                       │
 │   ├── data/       raw CSV                                    │
 │   ├── synthetic/  generated datasets                         │
 │   ├── results/    metrics tables                             │
 │   └── model/      model.joblib                               │
 │          ▲  │                                                │
 │          │  ▼                                                │
 │   Google Colab notebook ──► Vertex AI Experiments (runs)     │
 │   (data prep, generators,                                    │
 │    training, evaluation)  ──► Vertex AI Model Registry       │
 │                           ──► Vertex AI Endpoint (optional)  │
 └──────────────────────────────────────────────────────────────┘
```

| Component | Service | Role |
|---|---|---|
| Storage | Cloud Storage (IaaS-style object storage) | Raw data, synthetic data, results, model file |
| Development | Google Colab | Runs the notebook on Google-managed VMs |
| Experiment tracking | Vertex AI Experiments (PaaS) | Logs parameters and metrics for every run |
| Model management | Vertex AI Model Registry | Versioned model storage |
| Serving | Vertex AI Endpoint | Online prediction API (optional) |
| Cost control | Cloud Billing budgets | Spending alerts |

## Methodology

1. **Load and clean** the data (fix the label row, drop `ID`, standardise column names, convert to numeric).
2. **Split** into training and test sets with stratification. The test set is never used to fit generators or models, which prevents data leakage.
3. **Generate synthetic data** with three Synthetic Data Vault (SDV) generators fitted on the training set only:
   - Gaussian Copula (statistical)
   - CTGAN (conditional GAN)
   - TVAE (variational autoencoder)
4. **Evaluate the synthetic data**
   - *Fidelity:* SDMetrics quality score (with a built-in fallback score), distribution plots, correlation heatmaps
   - *Utility:* train on synthetic, test on real (TSTR)
   - *Privacy:* distance to closest record (DCR) against the real training set, compared with the real test set as a reference, plus a count of exact copies
5. **Train models** (Logistic Regression, Random Forest, XGBoost) in four scenarios, all tested on the same real test set:

   | Scenario | Training data |
   |---|---|
   | A | Real data only (baseline) |
   | B | Synthetic data only (one run per generator) |
   | C | Real data plus synthetic defaulters from the best generator, enough to balance the classes |
   | D | Real data balanced with SMOTE |

6. **Metrics:** ROC-AUC, Gini (`2 x AUC - 1`), KS statistic, PR-AUC, precision, recall, F1, plus training time per run. Gini and KS are the standard credit-industry measures.
7. **Explainability:** SHAP summary plot for the best tree-based model.
8. **Fairness:** Fairlearn selection rate and recall by sex, compared across the real-only, best and synthetic-only models.
9. **Deployment (optional):** save `model.joblib`, register it in the Vertex AI Model Registry with a matching prebuilt serving container, deploy to an endpoint and test it.

## Repository Structure

```
.
├── credit_risk_synthetic_gcp.ipynb   # the complete notebook
├── all_results.csv                   # metrics for every scenario and model
├── model.joblib                      # best trained model
├── report/                           # project report / case study (optional)
├── screenshots/                      # Google Cloud and results screenshots (optional)
├── .gitignore
└── README.md
```

Suggested `.gitignore`:

```
# data and generated files
default_of_credit_card_clients.csv
synthetic_*.csv
mlruns/
mlflow.db
# credentials
*.json
.env
```

> Do not commit the raw dataset, credentials, key files or your project ID if you prefer to keep it private.

## Requirements

- A Google account (a personal Gmail is easiest, since managed university or work accounts often restrict project creation)
- For the Google Cloud mode: a Google Cloud project with billing linked (the free trial is enough)
- Python 3 (provided by Colab)

Python libraries (installed by the first notebook cell):

```
sdv  imbalanced-learn  xgboost  fairlearn  shap
google-cloud-aiplatform  google-cloud-storage
```

pandas, NumPy, SciPy, scikit-learn and matplotlib are preinstalled in Colab.

## Setup and Usage

The notebook runs in two modes, chosen by `USE_GCS` in the configuration cell.

### Option A: Local mode in Colab (no Google Cloud project needed)

1. Download the dataset from UCI.
2. Open `credit_risk_synthetic_gcp.ipynb` in [Google Colab](https://colab.research.google.com) (File > Upload notebook).
3. Run the install cell. If Colab shows **Restart session**, click it and continue from the next cell without rerunning the install.
4. Leave `USE_GCS = False`, run the configuration cell, and upload the CSV when asked.
5. Run the remaining cells top to bottom.

### Option B: Google Cloud mode (Cloud Storage and Vertex AI)

1. **Create a project and budget.** Create a Google Cloud project, link billing, and create a budget alert under Billing.
2. **Enable APIs** (APIs & Services > Library): Vertex AI API, Compute Engine API, Notebooks API, Cloud Storage.
3. **Create a bucket** in one region (for example `asia-south1`) with public access prevention on, create a `data/` folder, and upload the dataset CSV.
4. **Open the notebook** in Colab and run the install cell (restart if prompted).
5. **Edit the configuration cell** (see below) and set `USE_GCS = True`. Run it and sign in with the Google account that owns the project when the popup appears. You should see `Connected to Google Cloud project: ...`.
6. **Run the cells** top to bottom. Data is read from your bucket, runs are logged to Vertex AI Experiments, and outputs are saved back to the bucket.
7. **Optional deployment:** set `DEPLOY_ENDPOINT = True` in Section 12 only when you are ready to deploy. Delete the endpoint afterwards.

## Configuration Reference

Set these in the configuration cell near the top of the notebook:

| Variable | Meaning |
|---|---|
| `USE_GCS` | `False` = local mode (upload CSV). `True` = use Cloud Storage and Vertex AI |
| `PROJECT_ID` | Your Google Cloud **project ID** (not the project name) |
| `REGION` | Region of your bucket, for example `asia-south1` |
| `BUCKET` | Bucket name only, without `gs://` |
| `DATA_FILE` | Path of the CSV inside the bucket, default `data/default_of_credit_card_clients.csv` |
| `EPOCHS` | Training epochs for CTGAN and TVAE (default 100; lower it on CPU) |
| `best_gen` | Generator used for Scenario C (chosen automatically by fidelity score; override manually if you like) |
| `IMAGE_OVERRIDE` | Serving container address for model registration, if the automatic guess fails |
| `DEPLOY_ENDPOINT` | `False` by default; set `True` to create a billed endpoint |

## Outputs

| File | Description |
|---|---|
| `all_results.csv` | One row per scenario and model with all metrics and training time |
| `model.joblib` | Best tree-based model (Random Forest or XGBoost, chosen automatically by Gini) |
| `synthetic_gaussian_copula.csv`, `synthetic_ctgan.csv`, `synthetic_tvae.csv` | Generated datasets |
| Plots in the notebook | Default-rate charts, distribution comparisons, correlation heatmaps, Gini and recall bar charts, SHAP summary |
| Vertex AI Experiments | Comparable runs for every scenario and model (Google Cloud mode) |

**Using the saved model:**

```python
import joblib, pandas as pd
model = joblib.load("model.joblib")

columns = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
           "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
           "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
customer = pd.DataFrame([[50000,2,2,1,35,2,2,2,0,0,0,30000,29000,28000,20000,15000,10000,
                          1500,1500,1000,1000,500,500]], columns=columns)

print("Default probability:", model.predict_proba(customer)[0][1])
```

The deployed endpoint returns **default probabilities** (for example 0.19 or 0.72), not hard 0/1 labels. Choose a cutoff that suits your risk appetite; 0.5 is only a default.

## Results

*Fill in from your own `all_results.csv`.*

| Scenario | Best model | ROC-AUC | Gini | KS | Recall | F1 |
|---|---|---|---|---|---|---|
| A: real only | | | | | | |
| B: synthetic only (best generator) | | | | | | |
| C: real + synthetic defaulters | | | | | | |
| D: real + SMOTE | | | | | | |

**Synthetic data quality**

| Generator | Quality score | Median DCR | Exact copies | Generation time |
|---|---|---|---|---|
| Gaussian Copula | | | | |
| CTGAN | | | | |
| TVAE | | | | |

**Key findings:** *(summarise RQ1-RQ4 here, including any negative results, for example "synthetic-only models reached X% of the real baseline Gini" or "synthetic defaulters did/did not beat SMOTE")*

A deployed endpoint test returned probabilities of 0.19, 0.72 and 0.26 for three test customers whose actual outcomes were 0, 1 and 0. That confirms the serving path works. It is a demonstration, not an accuracy measurement; the reported metrics come from the full 9,000-row test set.

## Cost Control and Cleanup

A Vertex AI endpoint bills every hour while deployed, even when idle.

- Set a **budget alert** in Billing before you begin.
- Deploy an endpoint only for your demo, then clean up:
  ```python
  endpoint.undeploy_all()
  endpoint.delete()
  ```
- Stop or delete any Workbench instance and Compute Engine VMs you created.
- When the project is finished, download your outputs and use **IAM & Admin > Settings > Shut down** to stop all charges.
- Colab itself is not billed to your Google Cloud credits, but disconnect the runtime when you are done (**Runtime > Disconnect and delete runtime**).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Edit PROJECT_ID and BUCKET first` | The values still hold placeholders, or the safety check was edited. Use your real project ID and bucket name |
| `403 Permission denied` | Signed in with a different account than the project owner, or billing is not linked |
| `404 Not Found` for the CSV | Check the bucket name and that the file is at `data/default_of_credit_card_clients.csv` |
| `Vertex AI API has not been used` | Enable it under APIs & Services > Library, wait a minute and rerun |
| Metadata, `fit` or `evaluate_quality` errors | SDV changed its API between versions. The notebook tries each known form; if one still fails, check `pip show sdv` and open an issue with the version and error |
| CTGAN/TVAE very slow | Use a GPU runtime, lower `EPOCHS`, or add `batch_size=1000` |
| Model registration fails | Set `IMAGE_OVERRIDE` to a prebuilt scikit-learn or XGBoost container whose version matches the library used for training |
| Colab asks to restart after install | Restart the session and continue without rerunning the install cell |

## Limitations

- One public dataset from one market and year (Taiwan, 2005); results may not transfer to other portfolios or loan products.
- Single-table generators only; transaction histories or customer relationships are not modelled.
- Privacy is assessed empirically (distance to closest record and exact-copy counts), not with formal guarantees such as differential privacy.
- A fixed 0.5 threshold is used for precision, recall and F1.
- Fairness is checked for one attribute (sex) and a few metrics.
- Deployment is a small-scale demonstration.

## Future Work

- Differentially private generators (for example DP-CTGAN) with a formal privacy budget
- Membership-inference attack tests
- Multi-table synthetic data for transaction histories
- Testing on a second dataset such as *Give Me Some Credit*
- Threshold tuning and cost-sensitive evaluation
- Vertex AI Model Monitoring for drift detection
- Managing the pipeline with Vertex AI Pipelines

## References

- Xu, L., Skoularidou, M., Cuesta-Infante, A., Veeramachaneni, K. (2019). *Modeling Tabular Data using Conditional GAN.* NeurIPS.
- Patki, N., Wedge, R., Veeramachaneni, K. (2016). *The Synthetic Data Vault.* IEEE DSAA.
- Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
- Yeh, I-C., Lien, C-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications.
- Lundberg, S., Lee, S-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Bird, S. et al. (2020). *Fairlearn: A toolkit for assessing and improving fairness in AI.* Microsoft Research.

## License and Author

**License:** [add your license, for example MIT]. The dataset has its own license from UCI; check it before redistributing.

**Author:** [Your Name], [University / Course], 121326030006

Contact: [email or LinkedIn]
