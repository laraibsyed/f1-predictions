# 🏎️ Formula 1 Race Predictor (AI-Powered)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model](https://img.shields.io/badge/Model-XGBoost_Ranker-orange)
![Status](https://img.shields.io/badge/Status-V1_Baseline_Complete-green)

## 📖 Overview
This project applies **Machine Learning (Learning to Rank)** to predict the finishing order of Formula 1 races. Unlike standard classification models that predict a simple "Win/Loss," this system uses **Pairwise Ranking** to determine if Driver A is likely to finish ahead of Driver B in a specific race context.

The model is trained on historical data (2018–2024) and tested on the 2025 season, achieving a **Winner Prediction Accuracy (Precision @ 1) of ~42%**, which matches or exceeds historical "Pole-to-Win" conversion rates.

## 📊 Key Results

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Winner Accuracy** | **41.67%** | Precision @ 1 (Predicted Winner = Actual Winner) |
| **Baseline Strategy** | ~40% | "Always bet on Pole Position" |
| **Random Guess** | 5% | 1 out of 20 drivers |

### What Matters to the Model?
The model discovered that **Grid Position** is the primary indicator of success, followed by **Team Strength** (specifically using Haas as a baseline anchor).

![Feature Importance Plot](assets/feature_importance.png)
*(Figure 1: XGBoost Feature Importance showing GridPosition as the dominant factor)*

---

## 🛠️ The Tech Stack

* **Language:** Python
* **Libraries:** `pandas`, `xgboost`, `scikit-learn`, `matplotlib`, `seaborn`
* **Algorithm:** XGBoost Ranker (`objective='rank:pairwise'`)

---

## 🧠 Model Architecture

### 1. Data Pipeline & Preprocessing
The raw data is transformed to prevent **Data Leakage** (the model seeing the future) and handle the **Cold Start** problem for new teams.

* **Target Inversion:** F1 scores are "Lower is Better" (1st place). The model maximizes scores, so the target was flipped: `Target = 21 - Position`.
* **The "DNF" Penalty:** Retired cars ("R", "NC") are mapped to position `20` to heavily penalize crashes.
* **Team Lineage Mapping:** Consolidated historical team names to handle rebrands:
    * *Toro Rosso / AlphaTauri* $\rightarrow$ `RB`
    * *Renault* $\rightarrow$ `Alpine`
    * *Force India / Racing Point* $\rightarrow$ `Aston Martin`
* **Future Proofing (2026):** Implemented proxy mapping for new manufacturers:
    * *Audi* $\rightarrow$ Mapped to `Sauber` lineage.
    * *Cadillac* $\rightarrow$ Mapped to `Haas` (New entrant baseline).

### 2. Feature Engineering
We engineered features that capture "Form" and "Class" without revealing the race result.

* **`SeasonStrength`:** Cumulative points scored by a driver *before* the race starts.
* **`DriverConfidence`:** Rolling average of finishing positions over the last 3 races (calculated using a `transform` bubble to prevent data leakage between drivers).
* **`TeamId`:** One-Hot Encoded to capture car performance hierarchy.

### 3. Model Logic
We use **Query Groups** to tell the XGBoost Ranker which drivers are competing in the same race. The model learns to sort the list of 20 drivers rather than predicting independent regression scores.

---

## 🔍 Visual Analysis

### Correlation Heatmap
The heatmap confirms the negative correlation between **Grid Position** and our **Inverted Target Score** (Lower Grid = Higher Score). It also highlights the "Team Hierarchy" automatically learned by the model.

![Correlation Heatmap](assets/correlation_matrix.png)
*(Figure 2: Feature Correlation Matrix showing the relationships between Grid, Points, and Teams)*

---

## 🔮 Future Roadmap (V2)
Teammate Delta: A feature measuring the performance gap between a driver and their teammate to isolate driver skill from car performance.
Circuit History: Adding "Course Specialist" features (e.g., Perez at Street Tracks).
Weather Complexity: Upgrading the binary Rain feature to granular forecast data.
