# Credit Risk Scoring & Loan Default Prediction

## Project Overview
This project predicts whether a customer is likely to default on a loan based on
their personal, financial, and credit history information. It's a beginner-friendly
data analyst project built around a simple, end-to-end machine learning workflow:
data cleaning, exploratory data analysis, model building, and evaluation.

Two notebooks are included:
- **Credit_Risk_Scoring.ipynb** — the full pipeline (cleaning, EDA, modeling, evaluation, feature importance)
- **EDA_and_Analytics.ipynb** — a dedicated visualization and analytics notebook that digs deeper into the dataset with grouped summaries and additional charts

## Tools Used
- Python 3
- Jupyter Notebook
- pandas, numpy (data handling)
- matplotlib, seaborn (visualization)
- scikit-learn (machine learning)

## Dataset Description
The dataset (`loan_data.csv`) contains ~1000 synthetic customer records with the following columns:

| Column | Description |
|---|---|
| Age | Customer's age |
| Income | Annual income |
| LoanAmount | Requested loan amount |
| CreditScore | Customer's credit score |
| EmploymentStatus | Employed / Self-Employed / Unemployed / Retired |
| LoanTerm | Loan repayment term in months |
| PreviousDefaults | Number of previous loan defaults |
| DebtToIncomeRatio | Ratio of debt to income |
| MaritalStatus | Single / Married / Divorced |
| Education | Highest education level |
| Default | Target variable (1 = defaulted, 0 = did not default) |

## Workflow
1. **Load & explore data** – shape, data types, summary statistics, missing values, duplicates
2. **Preprocess data** – handle missing values, remove duplicates, label-encode categorical columns, select features, split into train/test sets
3. **Exploratory Data Analysis** – 8+ visualizations exploring distributions and relationships in the data
4. **Model building** – train Logistic Regression and Decision Tree classifiers
5. **Model evaluation** – Accuracy, Precision, Recall, F1 Score, Confusion Matrix, Classification Report
6. **Feature importance** – identify which features drive predictions
7. **Conclusion** – summarize findings and business insights

## How to Run
1. Make sure Python 3 and Jupyter are installed.
2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```
4. Open `Credit_Risk_Scoring.ipynb` to run the full pipeline, and/or
   `EDA_and_Analytics.ipynb` for the visualization-focused notebook.
5. Run all cells from top to bottom.

## Results
- **Logistic Regression:** ~77% accuracy
- **Decision Tree:** ~75% accuracy
- **Top predictors:** Credit Score and Previous Defaults were consistently the strongest
  predictors of loan default, followed by Debt-to-Income Ratio and Income.

## Future Improvements
- Try additional models such as Random Forest or Gradient Boosting.
- Use cross-validation for more robust accuracy estimates.
- Perform hyperparameter tuning.
- Collect or incorporate real-world loan data for more realistic patterns.
