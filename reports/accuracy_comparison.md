# Heart Disease Detection — Accuracy Comparison Report

## Dataset
- Total samples : 305
- Training set  : 244
- Test set      : 61

## Decision Tree Model (with Feature Selection)

### Selected Features (9 out of 13)
- **cp**: 0.3327
- **ca**: 0.1363
- **thal**: 0.0851
- **sex**: 0.0710
- **restecg**: 0.0641
- **age**: 0.0626
- **oldpeak**: 0.0601
- **chol**: 0.0541
- **trestbps**: 0.0498

### Performance Metrics

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.820 |
| Precision | 0.824 |
| Recall    | 0.848 |
| F1-Score  | 0.836 |

Best hyperparameters: `{'max_depth': 5, 'min_samples_split': 10}`

**Feature Selection Strategy:** Selected top 9 features based on preliminary Decision Tree 
importance scores, capturing 90% of cumulative importance. This reduces model complexity 
and potential overfitting while maintaining predictive power.

## Rule-Based Expert System

| Metric   | Score |
|----------|-------|
| Accuracy | 0.541 |

*(Precision/Recall/F1 not computed for the rule-based system as it outputs ordinal risk levels, not binary probabilities.)*

## Comparison & Analysis

| Aspect          | Expert System                        | Decision Tree (Feature Selected)  |
|-----------------|--------------------------------------|-----------------------------------|
| Accuracy        | 0.541                   | 0.820      |
| Features used   | 10 (manually selected)               | 9 (data-driven)        |
| Explainability  | High — rules are human-readable      | Medium — tree paths readable      |
| Data required   | None (domain knowledge only)         | Labelled dataset required         |
| Adaptability    | Manual rule updates needed           | Retrain on new data               |
| Speed           | Very fast (rule firing)              | Very fast (tree traversal)        |

## Conclusion
The Decision Tree with feature selection achieves higher predictive accuracy by learning statistical
patterns from data while using only the most informative features (9/13). 
This reduces model complexity and improves generalization. The Expert System is more transparent 
and requires no training data, making it useful for rapid clinical screening when labelled examples 
are scarce. A hybrid approach (flag high-risk patients with rules, confirm with ML) is recommended
for real-world deployment.
