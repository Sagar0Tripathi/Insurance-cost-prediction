# 🏥 Medical Insurance Charges Prediction

## 📌 Problem Statement
ACME Insurance Inc. offers affordable health insurance to thousands of customers all over the United States.  

As the lead data scientist at ACME, **you are tasked with creating an automated system to estimate the annual medical expenditure for new customers**, using information such as:
- Age  
- Sex  
- BMI  
- Number of children  
- Smoking status  
- Region of residence  

These estimates will be used to determine the **annual insurance premium**.  
Due to regulatory requirements, the system must also be **explainable** (we need to justify why a prediction is made).  

Dataset used: [Medical Charges Dataset](https://github.com/stedy/Machine-Learning-with-R-datasets) (~1300 records).

---

## ⚙️ Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Checked dataset shape, column info, and missing values.  
- Explored distributions of **age, BMI, children, charges**.  
- Visualized relationships:
  - Scatterplots (charges vs age, charges vs BMI, smoker vs non-smoker).
  - Boxplots (children vs charges, sex vs charges, smoker vs charges).
  - Correlation heatmap.

### 2. Data Preprocessing
- Train/Validation/Test split → **60/20/20** (random_state=49).  
- Standardized numerical features (**age, BMI, children**).  
- One-hot encoded categorical features (**sex, smoker, region**).  
- Created clean input matrices for training models.

### 3. Model Training
- **Plain Linear Regression (OLS)** → baseline model.  
- **Log-Linear Regression** → trained on log(charges).  
- **RidgeCV** → L2 regularization.  
- **LassoCV** → L1 regularization.  

### 4. Model Evaluation
Metrics used:
- **MAE (Mean Absolute Error)**  
- **RMSE (Root Mean Squared Error)**  
- **R² (Coefficient of Determination)**  

Results (Validation set):
- **Plain OLS performed best** (R² ≈ 0.75).  
- **Log-transform underperformed** (R² ≈ 0.63).  
- **Ridge & Lasso** performed nearly identical to OLS (no gain, dataset is small and features are strong).  

Final Test performance (Plain OLS):  
- **MAE ≈ 4.1k**  
- **RMSE ≈ 5.9k**  
- **R² ≈ 0.77**

### 5. Interpretability
- Extracted coefficients to explain feature importance:
  - 🚬 **Smoker: +23.9k** (dominant factor)  
  - 👤 **Age: +3.5k**, **BMI: +2.2k**  
  - 👶 **Children: +660** per child  
  - ⚧️ **Sex (male): +150** (small effect)  
  - 🌍 **Region (non-Northeast): -500 to -1400**  

- Created a **bar chart of coefficients** to visualize importance.  
- Conclusion: **Smoking, Age, and BMI** are the strongest predictors.

### 6. Model Saving
- Used `joblib` to save:
  - Trained OLS model (`linear_model.pkl`).  
  - Preprocessing objects (`scaler.pkl`, `encoder.pkl`).  
- Allows re-use of the model for new predictions.

---

## 💻 Usage

### Install requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
