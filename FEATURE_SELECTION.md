# Feature Selection Implementation

## Overview
This project now implements **automatic feature selection** for the Decision Tree model based on feature importance scores. This reduces model complexity, improves generalization, and speeds up predictions.

## How It Works

### 1. Two-Stage Training Process

**Stage 1: Feature Importance Extraction**
```python
# Train a preliminary model to get feature importances
prelim_model = DecisionTreeClassifier(max_depth=5, random_state=42)
prelim_model.fit(X_train, y_train)
feature_importances = prelim_model.feature_importances_
```

**Stage 2: Train Final Model on Selected Features**
```python
# Select top N features (90% cumulative importance)
selected_features = top_features_by_importance(feature_importances)

# Train final model on selected features only
final_model.fit(X_train[selected_features], y_train)
```

### 2. Feature Selection Strategy

The algorithm selects features based on **cumulative importance**:
- Calculate importance scores for all features
- Sort features by importance (descending)
- Select features until cumulative importance ≥ 90%
- Keep minimum of 5 features (safety threshold)

**Example:**
```
Feature          Importance    Cumulative
ca               0.25          0.25
cp               0.20          0.45
thal             0.18          0.63
oldpeak          0.15          0.78
thalach          0.12          0.90  ← Stop here (90% reached)
age              0.05          0.95
chol             0.03          0.98
...
```
Result: Select top 5 features (ca, cp, thal, oldpeak, thalach)

## Benefits

### 1. Reduced Overfitting
- Fewer features = simpler model
- Less likely to memorize noise in training data
- Better generalization to new patients

### 2. Improved Interpretability
- Focus on most important clinical indicators
- Easier to explain predictions to medical staff
- Aligns with domain knowledge

### 3. Faster Predictions
- Fewer features to process
- Faster tree traversal
- Important for real-time clinical systems

### 4. Data Efficiency
- Requires fewer measurements per patient
- Reduces cost of data collection
- Useful when some tests are expensive/unavailable

## Implementation Details

### Modified Files

**1. `main.py`**
- Added feature selection before model training
- Saves selected features with the model
- Updated comparison report to show selected features

**2. `ml_model/train_model.py`**
- Standalone training script with feature selection
- Can be run independently: `python ml_model/train_model.py`

**3. `ml_model/predict.py`**
- Loads selected features from saved model
- Filters input data to selected features only
- Backward compatible with old model format

**4. `ui/app.py`**
- Displays which features are being used
- Handles both old and new model formats
- Shows feature selection info in UI

### Model Storage Format

The model is now saved as a dictionary:
```python
{
    "model": trained_decision_tree,
    "selected_features": ["ca", "cp", "thal", "oldpeak", "thalach"],
    "feature_importances": {"ca": 0.25, "cp": 0.20, ...}
}
```

## Usage

### Training with Feature Selection
```bash
# Run full pipeline (includes feature selection)
python main.py

# Or train standalone
python ml_model/train_model.py
```

### Making Predictions
```python
from ml_model.predict import predict

# Only need to provide selected features
patient = {
    "ca": 2,
    "cp": 1,
    "thal": 3,
    "oldpeak": 0.52,
    "thalach": 0.31,
    # Other features will be ignored if not selected
}

result = predict(patient)
print(f"Prediction: {result['label']}")
print(f"Features used: {result['features_used']}")
```

### Viewing Selected Features
The UI now displays which features were selected:
```
Using 5 selected features: ca, cp, thal, oldpeak, thalach
```

## Comparison: Before vs After

| Aspect | Without Feature Selection | With Feature Selection |
|--------|---------------------------|------------------------|
| Features used | 13 (all) | ~5-8 (top features) |
| Model complexity | Higher | Lower |
| Overfitting risk | Higher | Lower |
| Training time | Slower | Faster |
| Prediction time | Slower | Faster |
| Interpretability | Moderate | Higher |
| Accuracy | Baseline | Similar or better |

## Tuning Feature Selection

You can adjust the selection strategy in the code:

**Change cumulative importance threshold:**
```python
# More aggressive (fewer features)
n_features = (cumulative_importance <= 0.80).sum() + 1

# More conservative (more features)
n_features = (cumulative_importance <= 0.95).sum() + 1
```

**Set fixed number of features:**
```python
# Always use top 7 features
selected_features = feature_importances.head(7).index.tolist()
```

**Set minimum importance threshold:**
```python
# Only keep features with importance > 0.05
selected_features = feature_importances[feature_importances > 0.05].index.tolist()
```

## Expected Results

Typical feature selection results on the heart disease dataset:

**Most Important Features (usually selected):**
- `ca` - Number of major vessels (0-3)
- `cp` - Chest pain type
- `thal` - Thalassemia type
- `oldpeak` - ST depression
- `thalach` - Maximum heart rate

**Less Important Features (often excluded):**
- `fbs` - Fasting blood sugar
- `restecg` - Resting ECG
- `slope` - ST segment slope
- `sex` - Gender

This aligns with medical literature where vessel blockage, chest pain, and exercise capacity are strong heart disease indicators.

## Backward Compatibility

The code maintains backward compatibility:
- Old models (just the classifier) still work
- New models (dict with features) are preferred
- UI handles both formats automatically
- No need to retrain if you don't want feature selection

## Next Steps

Potential improvements:
1. **Recursive Feature Elimination (RFE)** - Iteratively remove least important features
2. **Cross-validation for feature selection** - More robust feature selection
3. **Multiple selection methods** - Compare different strategies
4. **Feature interaction analysis** - Consider feature combinations
5. **Domain expert validation** - Verify selected features make clinical sense
