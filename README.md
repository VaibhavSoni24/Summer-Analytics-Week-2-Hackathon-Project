# Summer Analytics 2026 - Week 2 Hackathon
### E-Commerce Conversion Prediction Challenge
**Consulting & Analytics Club, IIT Guwahati**

---

## Overview

Binary classification challenge to predict whether an e-commerce user **converts** (makes a purchase) based on anonymized user-level observations - covering demographics, browsing behavior, device and traffic data, purchase history, and marketing signals.

- **Evaluation Metric:** F1 Score
- **Target Variable:** `Converted` (1 = Converted, 0 = Not Converted)
- **OOF F1 Score Achieved:** `0.5577`

---

## Repository Structure

```
├── Deliverables/
│   ├── notebook.ipynb        # Full reproducible solution notebook
│   ├── submission.csv        # Final predictions for private_test.csv
│   ├── report.docx           # One-page methodology summary in docx format
│   └── report.pdf            # One-page methodology summary in pdf format
│
├── Week2-hackathon-datasetsacd318d/
│   ├── train.csv                       # Labeled training data (10,000 rows)
│   ├── public_test.csv                 # Labeled validation data (3,000 rows)
│   ├── private_test.csv                # Unlabeled test data for final evaluation (3,000 rows)
│   ├── SA2026_Starter_Notebook.ipynb   # Starter notebook
│   └── sample_submission.csv           # Required submission format
│
└── HackerEarth Hackathon8fc332d.pdf    # Official problem statement
```

---

## Dataset

| File | Rows | Has Labels |
|---|---|---|
| train.csv | 10,000 | Yes |
| public_test.csv | 3,000 | Yes |
| private_test.csv | 3,000 | No |

**Features:** `Age`, `Income`, `City_Tier`, `Device_Type`, `Traffic_Source`, `Pages_Viewed`, `Products_Viewed`, `Time_On_Site`, `Previous_Purchases`, `Discount_Seen`, `Browser_Version`, `Campaign_Code`

Missing values in `Age` (11.4%), `Income` (7.6%), and `Time_On_Site` (14.2%) — handled via median imputation.

---

## Methodology

### 1. Data Strategy
`train.csv` and `public_test.csv` (which includes labels) were merged into a combined 13,000-row training set to maximize available signal.

### 2. EDA Insights
Correlation analysis identified `Pages_Viewed` (r = 0.308) and `Products_Viewed` (r = 0.306) as the dominant predictors. `Discount_Seen` (r = 0.112) and `Previous_Purchases` (r = 0.105) provide secondary signal. `Age`, `Browser_Version`, and `Campaign_Code` showed near-zero raw correlation.

### 3. Feature Engineering
19 new features were engineered from the original 12, totaling **31 features**:

| Group | Features |
|---|---|
| Engagement Interactions | `Pages_x_Products`, `Pages_sq`, `Products_sq`, `Total_Engagement` |
| Discount Amplification | `Discount_x_Pages`, `Discount_x_Products`, `Discount_x_Prev` |
| Time Depth | `Time_per_Page`, `Time_per_Product` |
| Loyalty Flags | `Loyal_User` (>2 purchases), `New_User` (0 purchases) |
| Channel Flags | `Is_Paid`, `Is_Email`, `Is_Social`, `Is_Mobile`, `Is_Desktop` |
| Threshold Flags | `High_Pages` (>=10), `High_Products` (>=5) |
| Geographic | `Tier1_High_Income` |

### 4. Models & Ensemble

Three tree-based classifiers were trained with class imbalance handling:

| Model | CV F1 (5-fold) | Ensemble Weight |
|---|---|---|
| XGBoost (`scale_pos_weight`) | 0.5236 ± 0.0121 | 35% |
| LightGBM (`is_unbalance=True`) | 0.5296 ± 0.0060 | 35% |
| RandomForest (`class_weight='balanced'`) | 0.5471 ± 0.0098 | 30% |
| **Ensemble (threshold-tuned)** | **0.5577** | --- |

Final predictions use a weighted soft-vote ensemble of all three models.

### 5. Threshold Tuning
Default threshold of 0.5 is suboptimal for imbalanced data. Out-of-fold (OOF) probabilities were used to sweep thresholds from 0.30 to 0.70 and select the value maximizing F1. Optimal threshold: **0.40**.

---

## Reproducing the Results

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm

# Place data files in the same directory as notebook.ipynb, then run all cells
# Output: submission.csv
```

The notebook (`Deliverables/notebook.ipynb`) is fully self-contained and reproducible end-to-end.

---

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
lightgbm
```

---

## Submission Format

```
User_ID,Converted
103001,0
103002,1
...
```

---

*Summer Analytics 2026 | Consulting & Analytics Club, IIT Guwahati*
