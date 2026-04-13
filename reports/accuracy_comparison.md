# Heart Disease Detection — Accuracy Comparison Report

## Dataset
- Total samples : 305
- Training set  : 244
- Test set      : 61

## Decision Tree Model

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.770 |
| Precision | 0.806 |
| Recall    | 0.758 |
| F1-Score  | 0.781 |

Best hyperparameters: `{'max_depth': None, 'min_samples_split': 5}`

## Rule-Based Expert System

| Metric   | Score |
|----------|-------|
| Accuracy | 0.541 |

*(Precision/Recall/F1 not computed for the rule-based system as it outputs ordinal risk levels, not binary probabilities.)*

## Comparison & Analysis

| Aspect          | Expert System                        | Decision Tree                     |
|-----------------|--------------------------------------|-----------------------------------|
| Accuracy        | 0.541                   | 0.770      |
| Explainability  | High — rules are human-readable      | Medium — tree paths readable      |
| Data required   | None (domain knowledge only)         | Labelled dataset required         |
| Adaptability    | Manual rule updates needed           | Retrain on new data               |
| Speed           | Very fast (rule firing)              | Very fast (tree traversal)        |

## Conclusion
The Decision Tree achieves higher predictive accuracy by learning statistical
patterns from data. The Expert System is more transparent and requires no training
data, making it useful for rapid clinical screening when labelled examples are scarce.
A hybrid approach (flag high-risk patients with rules, confirm with ML) is recommended
for real-world deployment.
