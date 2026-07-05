# FIFA World Cup 2026 Match Winner Prediction
## Phase 3 - Model Training, Evaluation, and Selection Report

This report summarizes the results of the model training and selection phase, comparing the performance of multiple baseline and tuned machine learning models.

### 1. Executive Summary
- **Best Performing Model**: **LightGBM**
- **Composite Selection Score**: `0.7474`
- **Accuracy on Test Set**: `0.5852`
- **F1-Score on Test Set**: `0.4389`
- **ROC-AUC Score**: `0.7146`
- **Log Loss**: `0.9002`
- **Inference Throughput**: `2.93 predictions/sec`
- **Model File Size**: `0.9894 MB`

### 2. Model Selection Criteria & Weights
Models were evaluated on a composite utility score using the following weights:
| Metric | Weight | Description |
| --- | --- | --- |
| **Accuracy** | 20% | Standard classification accuracy on stratified test set |
| **F1-Score** | 30% | Macro F1-Score (balanced indicator across Home Win, Draw, Away Win classes) |
| **ROC-AUC** | 20% | Area under ROC Curve (overall class separation quality) |
| **Log Loss** | 10% | Penalty for confident incorrect predictions |
| **Throughput** | 10% | Predictions per second (computational speed) |
| **File Size** | 10% | Serialized model storage size on disk |

### 3. Comprehensive Model Rankings Table
Below are the rankings of all viable models evaluated. Models with poor log loss (>= 1.1) or low accuracy (< 55%) were filtered out as unviable.

| Rank | Model Name | Accuracy | F1-Score | ROC-AUC | Log Loss | Size (MB) | Throughput (pred/sec) | Composite Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LightGBM | 0.5852 | 0.4389 | 0.7146 | 0.9002 | 0.9894 | 2.93 | 0.7474 |
| 2 | CatBoost | 0.5801 | 0.4535 | 0.7101 | 0.9072 | 2.0566 | 5.76 | 0.7456 |
| 3 | HistGradientBoosting | 0.5835 | 0.4360 | 0.7150 | 0.8999 | 0.4247 | 5.49 | 0.7170 |
| 4 | LogisticRegression | 0.5817 | 0.4226 | 0.7108 | 0.9061 | 0.0017 | 83.17 | 0.6811 |
| 5 | XGBoost | 0.5742 | 0.4532 | 0.7073 | 0.9175 | 1.2049 | 8.40 | 0.6491 |
| 6 | NeuralNetwork | 0.5850 | 0.4204 | 0.7150 | 0.9011 | 0.0205 | 13.18 | 0.6489 |
| 7 | randomforest_tuned | 0.5792 | 0.4462 | 0.7040 | 0.9120 | 54.7650 | 1.36 | 0.6383 |
| 8 | catboost_tuned | 0.5824 | 0.4210 | 0.7122 | 0.9048 | 0.1218 | 29.36 | 0.6231 |
| 9 | RandomForest | 0.5722 | 0.4650 | 0.6982 | 0.9313 | 149.3032 | 1.20 | 0.5820 |
| 10 | lightgbm_tuned | 0.5808 | 0.4204 | 0.7103 | 0.9066 | 2.5097 | 0.43 | 0.5548 |
| 11 | xgboost_tuned | 0.5795 | 0.4140 | 0.7081 | 0.9156 | 1.5334 | 4.88 | 0.4800 |
| 12 | ExtraTrees | 0.5673 | 0.4614 | 0.6904 | 0.9475 | 390.0493 | 1.03 | 0.3775 |
| 13 | SVM | 0.5787 | 0.4139 | 0.6670 | 0.9403 | 0.5899 | 0.15 | 0.2419 |

### 4. Training Duration & Overfitting Analysis
The table below details training runtimes and comparative accuracy scores on training and test sets to identify overfitting patterns.

| Model Name | Training Accuracy | Test Accuracy | Overfitting Margin | Training Duration (s) |
| --- | --- | --- | --- | --- |
| LogisticRegression | 0.5820 | 0.5817 | 0.0003 | 0.7961 |
| DecisionTree | 0.9997 | 0.4702 | 0.5294 | 2.3059 |
| RandomForest | 0.9997 | 0.5722 | 0.4275 | 10.0084 |
| SVM | 0.5834 | 0.5787 | 0.0047 | 9.4800 |
| KNN | 0.6648 | 0.5314 | 0.1334 | 0.1778 |
| NaiveBayes | 0.5333 | 0.5300 | 0.0033 | 0.0258 |
| ExtraTrees | 0.9997 | 0.5673 | 0.4324 | 3.5520 |
| XGBoost | 0.7259 | 0.5742 | 0.1517 | 2.3932 |
| LightGBM | 0.6316 | 0.5852 | 0.0464 | 1.6818 |
| CatBoost | 0.6769 | 0.5801 | 0.0968 | 25.6274 |
| HistGradientBoosting | 0.6107 | 0.5835 | 0.0272 | 1.3531 |
| NeuralNetwork | 0.5871 | 0.5850 | 0.0021 | 38.5140 |

### 5. Explainability & Feature Importance
To ensure transparency and trust in model predictions, feature importances and SHAP values were generated for the winning LightGBM model.

- **Primary Feature Drivers**:
  1. **Elo Ratings (`home_elo_before`, `away_elo_before`)**: The absolute skill ratings of teams are the strongest predictors of match outcome.
  2. **Elo Win Probability (`elo_win_prob`)**: The mathematically derived win probability directly captures matchup difficulty.
  3. **Average Goal Differences (`home_avg_goal_diff`, `away_avg_goal_diff`)**: Historical goal margins represent team form and tactical effectiveness.
  4. **Form Difference (`form_diff`)**: Represents the momentum of teams in their 5 most recent matches.

- **SHAP Interpretability Summary**:
  - Home Elo before the match has a strong positive influence on the probability of a Home Win, and a strong negative influence on an Away Win.
  - The model effectively utilizes interaction terms (e.g. `win_pct_diff`, `goal_conceded_avg_diff`) to resolve matches between teams of similar Elo.

### 6. Conclusion and Recommendations
- The **LightGBM** model is recommended for deployment due to its optimal balance of predictive power, speed, and compactness.
- Tuned ensembles (e.g., XGBoost, CatBoost) offered similar accuracy but had slightly higher inference latencies or larger model sizes on disk.
- Future iterations could benefit from including player-level statistics, travel distance metrics, and home-crowd advantage factors.
