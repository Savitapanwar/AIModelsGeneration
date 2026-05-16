# AIModelsGeneration
The code to generate the models like Linear Regression

---

## LinearRegressionOLS.ipynb

### 📋 Overview
A comprehensive implementation of Linear Regression models (Simple and Multiple) using Ordinary Least Squares (OLS) methodology on the Boston Housing dataset. This notebook demonstrates the complete ML workflow from data exploration to model evaluation.

### 🎯 What is Performed

#### 1. **Data Loading & Exploration**
   - Load Boston Housing dataset from external CSV
   - Display dataset head and statistical summary
   - Explore data shape and basic statistics

#### 2. **Exploratory Data Analysis (EDA)**
   - Histogram plots of all features to visualize distributions
   - Scatter plot analysis (RM vs MEDV relationship)
   - Boxplot to identify outliers in target variable
   - Correlation heatmap to understand feature relationships

#### 3. **Simple Linear Regression (SLR)**
   - Model: `MEDV ~ RM` (House Price vs Average Rooms)
   - Train-test split (70-30 ratio, random_state=42)
   - Model fitting using scikit-learn's LinearRegression
   - Visualization of regression line fit

#### 4. **Multiple Linear Regression (MLR)**
   - Model: `MEDV ~ RM + LSTAT + PTRATIO`
   - Features:
     - `rm`: Average number of rooms
     - `lstat`: % lower status population
     - `ptratio`: Pupil-teacher ratio
   - Model evaluation with R² score (0.651)
   - Coefficient analysis and interpretation

#### 5. **Model Evaluation**
   - R² Score calculation
   - Predicted vs Actual values comparison
   - Coefficient interpretation table

---

### 🏗️ Visual Architecture & Process Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    BOSTON HOUSING DATASET                        │
│                        (506 samples)                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  LOAD & EXPLORE │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
        ┌───────────┤ EXPLORATORY DA  │──────────┐
        │           │    (EDA)         │          │
        │           └──────────────────┘          │
        │                                          │
    ┌───▼────┐   ┌──────────┐   ┌─────────┐  ┌──▼─────┐
    │Histogram│   │ Scatter  │   │ Boxplot │  │Heatmap │
    │ Plots   │   │  Plots   │   │(Outlier)│  │(Corr)  │
    └────────┘   └──────────┘   └─────────┘  └────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
    ┌───▼──────────────┐            ┌────────────▼────┐
    │ SIMPLE LINEAR    │            │ MULTIPLE LINEAR │
    │  REGRESSION      │            │  REGRESSION     │
    │   (SLR)          │            │    (MLR)        │
    └───┬──────────────┘            └────────┬────────┘
        │                                    │
    ┌───▼──────────────┐            ┌────────▼────────┐
    │ Features: RM     │            │ Features:       │
    │ Train-Test Split │            │ RM+LSTAT+PTRATIO│
    │ Model Fit        │            │ Train-Test Split│
    │ Predictions      │            │ Model Fit       │
    │ Line Visualization           │ Predictions     │
    └───┬──────────────┘            └────────┬────────┘
        │                                    │
        └────────────────┬───────────────────┘
                         │
                 ┌───────▼────────┐
                 │ MODEL EVALUATION│
                 │  - R² Score     │
                 │  - Coefficients │
                 │  - Predictions  │
                 │    Table        │
                 └────────────────┘
```

---

### 📊 Dataset Information

| Attribute | Details |
|-----------|---------|
| **Dataset** | Boston Housing |
| **Samples** | 506 |
| **Features** | 14 (including target) |
| **Target** | MEDV (Median home value in $1000s) |
| **Train-Test Split** | 70%-30% |
| **Random State** | 42 |

---

### 🔑 Key Findings

**Multiple Linear Regression Results:**
- **R² Score**: 0.651 (65.1% variance explained)
- **Coefficients**:
  - `rm`: +4.46 (more rooms → higher price)
  - `lstat`: -0.61 (lower status → lower price)
  - `ptratio`: -0.86 (higher ratio → lower price)

---

### 📦 Libraries Used
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `matplotlib`: Visualization
- `seaborn`: Statistical visualization
- `scikit-learn`: Machine learning models & metrics
- `statsmodels`: Statistical modeling

---

### 🚀 Usage
1. Open the notebook in Google Colab or Jupyter
2. Run cells sequentially from top to bottom
3. Observe EDA visualizations
4. Compare SLR vs MLR model performance
5. Analyze predictions and coefficients

---

### 📝 Notes
- Data is loaded from: `https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv`
- Models implemented using Ordinary Least Squares (OLS)
- Suitable for understanding ML regression fundamentals
