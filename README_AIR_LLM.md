::: {align="center"}
# AIR-LLM

### Adaptive Missing-Value Imputation via LLM-Guided Recommendation and Empirical Strategy Selection

**An evidence-driven framework for context-aware, feature-level
imputation strategy selection**

![Research](https://img.shields.io/badge/Research-Tabular%20Imputation-1f6feb?style=for-the-badge)
![LLM](https://img.shields.io/badge/LLM-Guided%20Recommendation-7c3aed?style=for-the-badge)
![Evaluation](https://img.shields.io/badge/Selection-Empirical%20%2B%20Multi--Objective-0f766e?style=for-the-badge)
![Conference](https://img.shields.io/badge/ICMLDE%205.0-2026-2563eb?style=for-the-badge)

**Mohd Amzad · Faiyaz Ahmad\* · Bashir Alam**\
Department of Computer Engineering, Jamia Millia Islamia, New Delhi,
India\
**Corresponding author:** Faiyaz Ahmad · <fahmad1@jmi.ac.in>
:::

------------------------------------------------------------------------

## Contents

-   [Overview](#overview)
-   [Research at a glance](#research-at-a-glance)
-   [Motivation and research gap](#motivation-and-research-gap)
-   [Key contributions](#key-contributions)
-   [Framework architecture](#framework-architecture)
-   [Methodology](#methodology)
-   [Mathematical formulation](#mathematical-formulation)
-   [Datasets and experimental
    design](#datasets-and-experimental-design)
-   [Evaluation metrics](#evaluation-metrics)
-   [Results](#results)
-   [Ablation study](#ablation-study)
-   [Limitations and future work](#limitations-and-future-work)
-   [Suggested repository structure](#suggested-repository-structure)
-   [Getting started](#getting-started)
-   [Reproducibility checklist](#reproducibility-checklist)
-   [Citation and references](#citation-and-references)
-   [Contact](#contact)

## Overview

Missing values are a persistent challenge in real-world tabular data.
Inappropriate imputation can distort feature distributions, weaken
relationships among variables, and reduce the reliability of downstream
machine-learning models. The effectiveness of an imputation method
depends on dataset properties, feature characteristics, missingness
mechanisms, and the downstream task.

**AIR-LLM (Adaptive Imputation via LLM-Guided Recommendation and
Empirical Strategy Selection)** formulates missing-value imputation as a
**context-aware strategy-selection problem**. Rather than applying one
fixed algorithm to every feature, AIR-LLM profiles the data, uses a
large language model (LLM) to recommend candidate strategies,
empirically evaluates those candidates using masked observed values, and
selects a feature-specific strategy through normalised multi-objective
utility.

> **Central principle:** The LLM recommends candidate strategies;
> empirical evidence determines the final selection. The LLM does not
> directly generate missing values in the proposed framework.

The paper evaluates AIR-LLM on Adult Income, Bank Marketing, and
Diabetes 130-US Hospitals datasets under MCAR, MAR, and MNAR conditions.
It reports an average reconstruction **MAE of 0.1275 ± 0.006** and
downstream-performance gains of **up to 20% over statistical baselines**
in the reported experiments.

## Research at a glance

  Aspect                 Description
  ---------------------- ---------------------------------------------------------
  Research task          Adaptive missing-value imputation strategy selection
  Framework              AIR-LLM
  Data                   Structured / tabular
  Decision granularity   Feature-specific
  Recommendation         LLM-guided candidate ranking
  Validation             Empirical mask-and-recover evaluation
  Selection              Normalised multi-objective utility
  Missingness            MCAR, MAR, MNAR
  Datasets               Adult Income, Bank Marketing, Diabetes 130-US Hospitals
  Reported average MAE   **0.1275 ± 0.006**
  Reported gain          **Up to 20% over statistical baselines**

## Motivation and research gap

Conventional imputation approaches include statistical,
machine-learning, deep-learning, and generative methods. However, their
performance depends on the data context, and a method that works well
for one dataset or feature may not be optimal for another.

  -----------------------------------------------------------------------
  Research challenge                  AIR-LLM response
  ----------------------------------- -----------------------------------
  Fixed methods are not               Selects strategies at feature level
  context-adaptive                    using contextual profiles

  Missingness mechanisms differ       Incorporates MCAR, MAR, and MNAR
                                      context

  Imputation can distort              Evaluates distributional fidelity
  distributions and dependencies      and dependency preservation

  LLM recommendations may not reflect Treats recommendations as
  measured performance                candidates, not ground truth

  A single metric is insufficient     Uses multi-objective utility across
                                      five criteria
  -----------------------------------------------------------------------

## Key contributions

1.  **Context-aware formulation:** Reframes imputation as a
    strategy-selection problem rather than applying one universal
    method.
2.  **LLM-guided recommendation:** Uses structured dataset, feature,
    missingness, and task profiles to rank candidate strategies.
3.  **Empirical validation:** Tests recommendations through
    mask-and-recover experiments using known observed values.
4.  **Multi-objective selection:** Combines reconstruction,
    distribution, dependency, predictive, and efficiency evidence.
5.  **Feature-level adaptation:** Selects a suitable strategy for each
    incomplete feature based on empirical utility.

## Framework architecture

The AIR-LLM workflow is:

**Dataset profiling → Missingness and feature characterisation → LLM
recommendation → Candidate evaluation → Utility-based selection →
Confidence assessment → Final imputation**

``` mermaid
flowchart TD
    A[Incomplete tabular dataset] --> B[Dataset profiling]
    B --> C[Missingness characterisation]
    C --> D[Feature-level profiling]
    D --> E[LLM recommendation engine]
    E --> F[Ranked candidate strategies]
    F --> G[Mask observed values]
    G --> H[Apply candidate methods]
    H --> I[Empirical evaluation]
    I --> J[Normalise evaluation criteria]
    J --> K[Composite utility calculation]
    K --> L[Feature-specific strategy selection]
    L --> M[Confidence assessment]
    M --> N[Final imputation]
    N --> O[Completed dataset]
```

## Methodology

### 1. Dataset profiling

For dataset (D={X_1,X_2,`\ldots`{=tex},X_p}), AIR-LLM constructs a
profile containing:

-   Dataset size and structure.
-   Feature types.
-   Feature distributions.
-   Inter-feature dependencies.
-   Missingness characteristics.
-   Downstream-task properties.

The dataset profile is represented as:

\[
P_D=\[P\_{`\text{size}`{=tex}},P\_{`\text{type}`{=tex}},P\_{`\text{distribution}`{=tex}},
P\_{`\text{dependency}`{=tex}},P\_{`\text{missingness}`{=tex}},P\_{`\text{task}`{=tex}}\]
\]

The overall missingness rate is:

\[ MR_D=`\frac{\sum_{i=1}^{n}\sum_{j=1}^{p}M_{ij}}{np}`{=tex} \]

where (M\_{ij}=1) indicates a missing value and (0) indicates an
observed value.

### 2. Missingness characterisation

AIR-LLM considers three missingness mechanisms:

  -----------------------------------------------------------------------
  Mechanism                           Definition
  ----------------------------------- -----------------------------------
  **MCAR** --- Missing Completely At  Missingness is independent of
  Random                              observed and unobserved values.

  **MAR** --- Missing At Random       Missingness depends on observed
                                      variables, not on the unobserved
                                      values after conditioning on
                                      observed data.

  **MNAR** --- Missing Not At Random  Missingness remains dependent on
                                      unobserved values after accounting
                                      for observed variables.
  -----------------------------------------------------------------------

These mechanisms form part of the context used for recommendation and
empirical evaluation. MNAR is particularly challenging because the
missingness process depends on values that are unavailable.

### 3. Feature characterisation

For every incomplete feature (X_j), AIR-LLM builds a feature-level
profile:

\[ P_j=\[T_j,M_j,D_j,R_j,C_j,Y_j\] \]

representing feature type, missingness, distribution, dependencies,
contextual information, and downstream-task relevance.

The feature missingness rate is:

\[ MR_j=`\frac{\sum_{i=1}^{n}M_{ij}}{n}`{=tex} \]

The paper identifies Pearson/Spearman correlation, Cramér's V, and
mutual information for dependency characterisation.

### 4. Candidate strategy pool

The heterogeneous candidate pool described in the paper includes:

  -----------------------------------------------------------------------
  Family                              Candidate methods
  ----------------------------------- -----------------------------------
  Statistical                         Mean, median, mode, constant,
                                      random-sample

  Machine learning                    KNN, Iterative Imputer / MICE,
                                      Random Forest, Gradient Boosting,
                                      MissForest

  Matrix-based                        SoftImpute, matrix factorisation

  Deep learning                       Autoencoders, VAE / MIWAE

  Generative                          GAIN and other generative
                                      approaches

  Diffusion                           Tabular diffusion methods

  LLM-based                           LLM-dependent imputation approaches
  -----------------------------------------------------------------------

Only strategies compatible with the feature type and experimental
environment are evaluated.

### 5. LLM recommendation

The recommendation context combines dataset, missingness, feature, and
task profiles:

\[ P=\[P_D,P_M,P_F,P_T\] \]

The LLM uses this structured context to produce a ranked list of
candidate strategies with confidence and a short rationale. It does not
directly generate the missing values.

### 6. Empirical mask-and-recover evaluation

LLM recommendations are treated as hypotheses and tested empirically.

1.  Select a subset of observed values in an incomplete feature.
2.  Artificially mask those values.
3.  Apply a candidate strategy using the remaining feature information.
4.  Compare reconstructed values with their known originals.
5.  Measure performance across the evaluation criteria.

The masked subset is represented as:

\[ X_j^{mask}`\subset `{=tex}X_j^{obs} \]

and the candidate reconstruction as:

\[ `\hat{X}`{=tex}*j^{mask}=s_k(X_j^{obs},X*{-j}) \]

This provides a measurable basis for comparing candidates.

### 7. Multi-objective adaptive selection

Candidates are assessed across five dimensions:

1.  Reconstruction accuracy.
2.  Distributional fidelity.
3.  Dependency preservation.
4.  Downstream predictive utility.
5.  Computational efficiency.

The normalised composite utility is:

\[ U\_{jk}=w_1A\_{jk}+w_2D\_{jk}+w_3R\_{jk}+w_4P\_{jk}+w_5E\_{jk} \]

subject to:

\[ `\sum`{=tex}\_{m=1}\^{5}w_m=1 \]

Error-based metrics are converted into benefit scores; benefit-based
metrics are normalised directly.

For feature (X_j), the selected strategy is:

\[
s_j\^\*=`\underset{s_k\in S_j}{\arg\max}`{=tex};U(s_k`\mid `{=tex}P_j)
\]

The final decision therefore depends on empirical utility, not the LLM
ranking alone.

### 8. Confidence estimation and explainability

The paper estimates selection confidence using LLM confidence, the
empirical utility margin, and stability across repeated evaluations:

\[
C_j=`\alpha `{=tex}C\_{LLM}+`\beta `{=tex}C\_{margin}+`\gamma `{=tex}C\_{stab}
\]

where:

\[ `\alpha`{=tex}+`\beta`{=tex}+`\gamma=1`{=tex} \]

and:

\[ C\_{margin}=`\frac{U_1-U_2}{|U_1|+\epsilon}`{=tex} \]

Here, (U_1) and (U_2) are the highest and second-highest strategy
utilities. If confidence falls below threshold (`\tau`{=tex}), the
framework treats the selection as uncertain and triggers additional
candidate evaluation.

The explainability record includes data evidence, LLM rationale,
empirical evidence, and the final decision.

### 9. Final imputation

The selected feature-specific strategy is applied to the original
missing values. The intended output is a completed tabular dataset
accompanied by evidence supporting the strategy-selection decision.

## Mathematical formulation

The central AIR-LLM decision is:

\[ `\boxed{s_j^*=\underset{s_k\in S_j}{\arg\max}\;U(s_k\mid P_j)}`{=tex}
\]

This expresses the core contribution: select an imputation strategy for
each incomplete feature according to its contextual profile and
empirically measured multi-objective utility.

## Datasets and experimental design

The paper evaluates three heterogeneous datasets from the UCI Machine
Learning Repository.

  Dataset                         Instances   Features Task
  --------------------------- ------------- ---------- ---------------------------
  Adult Income                       32,560         15 Binary classification
  Bank Marketing                     45,211         17 Binary classification
  Diabetes 130-US Hospitals         101,766         48 Multiclass classification
  **Total**                     **179,537**        --- ---

### Dataset sources

-   [Adult Income --- UCI](https://doi.org/10.24432/C5XW20)
-   [Bank Marketing --- UCI](https://doi.org/10.24432/C5K306)
-   [Diabetes 130-US Hospitals --- UCI](https://doi.org/10.24432/C5230J)

### Data splitting

  Split          Proportion
  ------------ ------------
  Training              70%
  Validation            15%
  Test                  15%

The paper uses stratified splitting to preserve class ratios.

### Missingness experiment matrix

  Dimension                        Configuration
  -------------------------------- -------------------------
  Datasets                         3
  Mechanisms                       MCAR, MAR, MNAR
  Missingness rates                10%, 20%, 30%, 40%, 50%
  Repetitions                      5
  Total dataset-level conditions   **225**

\[ 3`\times3`{=tex}`\times5`{=tex}`\times5`{=tex}=225 \]

Different random seeds are used across repetitions. Values masked for
validation are retained as ground truth for mask-and-recover evaluation.

## Evaluation metrics

  -----------------------------------------------------------------------
  Dimension               Metrics in the paper    Purpose
  ----------------------- ----------------------- -----------------------
  Reconstruction accuracy MAE, RMSE               Error between
                                                  reconstructed and
                                                  original observed
                                                  values

  Distributional fidelity Wasserstein distance,   Preservation of feature
                          Jensen--Shannon         distributions
                          divergence              

  Dependency preservation Correlation, mutual     Preservation of feature
                          information             relationships

  Predictive utility      Macro F1, ROC AUC       Downstream predictive
                                                  performance

  Computational           Execution time          Computational cost
  efficiency                                      
  -----------------------------------------------------------------------

No single metric determines the final strategy; the criteria are
normalised and combined in the composite utility.

## Results

### Quantitative reconstruction performance

  Method                         MAE                RMSE Selection
  ------------- -------------------- ------------------- --------------
  Mean / Mode         0.2002 ± 0.009       0.251 ± 0.010 Fixed
  KNN                 0.2032 ± 0.008       0.744 ± 0.015 Fixed
  MICE                0.1707 ± 0.007       0.214 ± 0.009 Fixed
  MissForest          0.1356 ± 0.006       0.170 ± 0.005 Fixed
  GAIN                0.1340 ± 0.005       0.168 ± 0.006 Fixed
  VAE / MIWAE         0.1620 ± 0.007       0.257 ± 0.012 Fixed
  Diffusion           0.1197 ± 0.004       0.150 ± 0.005 Fixed
  **AIR-LLM**     **0.1275 ± 0.006**   **0.161 ± 0.005** **Adaptive**

### Key findings

-   AIR-LLM reports average reconstruction **MAE = 0.1275 ± 0.006** and
    **RMSE = 0.161 ± 0.005**.
-   The paper reports up to **20% improvement over statistical
    baselines** and approximately **5--7% improvement over robust
    machine-learning and generative approaches** in its adaptive
    comparisons.
-   The strongest fixed baseline in raw MAE and RMSE is Diffusion.
-   The paper reports that the best solutions were found in **68% of
    feature-level evaluations**.
-   AIR-LLM's contribution is adaptive, feature- and context-dependent
    selection; it does not claim the lowest raw reconstruction error in
    every comparison.

> Interpret the results in the context of the paper's experimental
> protocol. Fixed-baseline reconstruction scores and the value of
> adaptive strategy selection measure related but distinct aspects of
> the framework.

## Ablation study

The paper examines the contribution of LLM recommendation and empirical
validation.

  -----------------------------------------------------------------------
  Ablation                Reported result         Interpretation
  ----------------------- ----------------------- -----------------------
  Without LLM guidance    **12% loss in overall   Random candidate
                          performance**           selection weakens
                                                  context-sensitive
                                                  recommendation.

  Without empirical       **9% increase in        Relying on LLM
  validation              reconstruction-error    confidence alone makes
                          variation**             selection less stable.
  -----------------------------------------------------------------------

The ablation findings support the combined use of LLM reasoning for
candidate recommendation and empirical evaluation for final selection.

## Limitations and future work

The paper identifies the following limitations:

-   **MNAR conditions:** Missingness dependent on unobserved values
    remains challenging.
-   **Computational overhead:** Repeated candidate evaluations and
    mask-and-recover experiments increase computational cost.
-   **Confidence calibration:** Confidence estimation under difficult
    missingness conditions requires refinement.
-   **Scale and heterogeneity:** Further evaluation is needed on larger
    and more diverse datasets.

Future directions include larger heterogeneous datasets,
resource-constrained scenarios, streaming settings, efficient local
LLMs, improved confidence calibration, and automatic candidate
expansion.

## Suggested repository structure

The following is a **suggested layout**, not a claim that all files
already exist. Keep the README aligned with the actual files committed
to your repository.

``` text
AIR-LLM/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── configs/
│   ├── dataset_config.yaml
│   ├── experiment_config.yaml
│   └── strategy_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   ├── 00_environment_and_configuration.ipynb
│   ├── 01_dataset_loading.ipynb
│   ├── 02_preprocessing_and_splitting.ipynb
│   ├── 03_missingness_generation.ipynb
│   ├── 04_dataset_and_feature_profiling.ipynb
│   ├── 05_candidate_recommendation.ipynb
│   ├── 06_empirical_candidate_evaluation.ipynb
│   ├── 07_adaptive_strategy_selection.ipynb
│   ├── 08_comparative_evaluation.ipynb
│   └── 09_final_analysis.ipynb
├── src/
│   ├── profiling/
│   ├── missingness/
│   ├── recommendation/
│   ├── strategies/
│   ├── evaluation/
│   ├── selection/
│   └── explainability/
├── results/
│   ├── metrics/
│   ├── tables/
│   ├── figures/
│   └── logs/
├── paper/
│   └── AIR_LLM_Paper.pdf
└── docs/
    ├── methodology.md
    └── experimental_protocol.md
```

## Getting started

The paper defines the research framework and experimental protocol, but
does not specify a complete version-pinned software environment or a
universal command-line entry point. The commands below are setup
examples; adapt them to the actual implementation before publishing.

### 1. Clone the repository

Replace the placeholder with your GitHub username or organisation:

``` bash
git clone https://github.com/YOUR-USERNAME/AIR-LLM.git
cd AIR-LLM
```

### 2. Create a Python environment

``` bash
python -m venv .venv
```

Activate it:

**Windows**

``` bash
.venv\Scripts\activate
```

**Linux / macOS**

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

If a tested `requirements.txt` is included:

``` bash
pip install -r requirements.txt
```

The paper does not specify package versions. Publish a validated
dependency file based on the packages actually used by the
implementation.

### 4. Configure the experiments

Before running experiments:

-   Obtain datasets from their official sources and follow their terms.
-   Configure dataset paths and target columns.
-   Set the train/validation/test split.
-   Configure missingness mechanisms, rates, and random seeds.
-   Check candidate-strategy compatibility with feature types.
-   Configure LLM provider/model credentials securely, if required by
    the implementation.
-   Set the utility criteria, normalisation procedure, and weights.

Do not commit API keys, credentials, restricted data, or large raw
datasets to GitHub.

### 5. Run the implementation

Add the actual entry-point command after committing and testing it. For
example, if your project exposes a script:

``` bash
# Example only — replace with the real script and arguments.
python path/to/experiment_entrypoint.py --config configs/experiment_config.yaml
```

For a notebook-based release, document the correct notebook execution
order and required inputs.

## Reproducibility checklist

For a reproducible research release, document:

-   Dataset source and version.
-   Preprocessing and target-column definitions.
-   Train, validation, and test split proportions.
-   Missingness mechanism, rate, and random seed.
-   Candidate strategy registry and compatibility rules.
-   LLM provider/model, prompt template, and generation settings.
-   Utility criteria, normalisation, and weights.
-   Confidence threshold and additional-evaluation policy.
-   Software versions, hardware, runtime, and execution commands.
-   Per-run metrics, aggregate mean ± standard deviation, and failure
    logs.

The paper reports five repetitions and mean ± standard deviation. It
describes Friedman and Wilcoxon signed-rank tests with Holm correction
where needed, using significance level (`\alpha=0.05`{=tex}).

**Reproduction status:** Do not claim that results have been
independently reproduced until the corresponding experiments have been
rerun from the public repository and their outputs verified.

## Citation and references

If you use AIR-LLM in academic work, please cite the research paper.

### Provisional BibTeX

The supplied paper copy identifies the title, authors, and *Procedia
Computer Science*, but does not provide final volume, page range, or DOI
details. Update this entry when the final publisher record is available.

``` bibtex
@article{amzad_airllm,
  title   = {Adaptive Missing-Value Imputation via LLM-Guided Recommendation and Empirical Strategy Selection},
  author  = {Amzad, Mohd and Ahmad, Faiyaz and Alam, Bashir},
  journal = {Procedia Computer Science},
  year    = {2025},
  note    = {International Conference on Machine Learning and Data Engineering, ICMLDE 5.0}
}
```

### Paper details

-   **Title:** Adaptive Missing-Value Imputation via LLM-Guided
    Recommendation and Empirical Strategy Selection
-   **Authors:** Mohd Amzad, Faiyaz Ahmad, Bashir Alam
-   **Affiliation:** Department of Computer Engineering, Jamia Millia
    Islamia, New Delhi, India
-   **Corresponding author:** fahmad1@jmi.ac.in
-   **Publication venue stated in the paper:** Procedia Computer
    Science; peer review under ICMLDE 5.0, 2026

### Selected references

1.  Van Buuren, S. & Groothuis-Oudshoorn, K. (2011). *mice: Multivariate
    imputation by chained equations in R*. Journal of Statistical
    Software.
2.  Stekhoven, D. J. & Bühlmann, P. (2012). *MissForest: Non-parametric
    missing value imputation for mixed-type data*. Bioinformatics.
3.  Mattei, P. A. & Frellsen, J. (2019). *MIWAE: Deep generative
    modelling and imputation of incomplete data sets*. ICML.
4.  Yoon, J., Jordon, J. & van der Schaar, M. (2018). *GAIN: Missing
    data imputation using generative adversarial nets*. ICML.
5.  Zheng, S. & Charoenphakdee, N. (2023). *Diffusion models for missing
    value imputation in tabular data*. NeurIPS.
6.  Feitelberg, J. et al. (2025). *TabImpute: Accurate and fast
    zero-shot missing-data imputation with a pre-trained transformer*.
    ICML.
7.  He, X. et al. (2025). *LLM-Forest: Ensemble learning of LLMs with
    graph-augmented prompts for data imputation*. Findings of ACL.

Refer to the paper's bibliography for the complete reference list.

## Contact

**Research group:** Department of Computer Engineering, Jamia Millia
Islamia, New Delhi, India.

For research questions, methodology discussions, or academic
collaboration:

-   **Faiyaz Ahmad (Corresponding Author):** <fahmad1@jmi.ac.in>

For code questions and implementation issues, use the GitHub **Issues**
tab once enabled for the repository.

------------------------------------------------------------------------

::: {align="center"}
### AIR-LLM

**Profile the data. Recommend candidates. Validate empirically. Select
adaptively.**

*An evidence-driven approach to context-aware tabular data imputation.*
:::
