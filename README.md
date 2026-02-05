AI-ML-Internship-Tasks
📌 Overview

This repository contains my AI/ML internship projects.
It includes tasks on data exploration, visualization, regression, classification, and other ML concepts.

Each task has:

Python script (.py)

Dataset (or dataset link if too large)

Output plots saved in outputs/ folder

README-style markdown summary with objectives, steps, models, and results

📂 Folder Structure
AI-ML-Internship-Tasks/
│
├── Task-1_Iris_Exploration/
│   ├── iris_exploration.py
│   ├── outputs/
│   └── README.md
│
├── Task-2_Stock_Prediction/
│   ├── stock_prediction.py
│   ├── outputs/
│   └── README.md
│
├── Task-3_Heart_Disease/
│   ├── heart_disease_prediction.py
│   ├── outputs/
│   └── README.md
│
├── Task-4_...
├── Task-5_...
└── Task-6_...

🛠 Tools & Libraries

Python 3.x

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

yfinance (for stock prediction)

📋 Tasks Summary
Task 1 – Iris Dataset Exploration & Visualization

Objective: Explore, analyze, and visualize the Iris dataset to understand feature distributions and relationships.

Key Steps:

Load dataset (seaborn.load_dataset("iris"))

Inspect dataset (.head(), .info(), .describe())

Visualize:

Scatter plot → Sepal relationships

Histogram → Feature distributions

Box plot → Outliers detection

Save plots in outputs/ folder

Insights: Setosa is clearly separable; Petal features are strong predictors.

Task 2 – Stock Price Prediction

Objective: Predict next-day stock closing price using historical data.

Key Steps:

Load historical stock data using yfinance

Prepare features: Open, High, Low, Volume

Create target: Next-day Close price

Train Linear Regression model

Evaluate with MAE

Plot actual vs predicted prices and save in outputs/

Insights: Model captures general trend; minor deviations during high volatility.

Task 3 – Heart Disease Prediction

Objective: Predict risk of heart disease using patient health data.

Key Steps:

Load dataset (heart.csv)

Handle missing values (median for numeric, mode for categorical)

Encode categorical features using LabelEncoder

Create binary target column (target)

EDA: Target distribution, Correlation heatmap

Train Logistic Regression model

Evaluate: Accuracy, Confusion Matrix, ROC-AUC score

Visualize ROC curve & Feature importance

Insights: Accuracy ~0.80, ROC-AUC ~0.87; Key features: cp, thal, ca, oldpeak

🚀 How to Run

Clone repository:

git clone https://github.com/<your-username>/AI-ML-Internship-Tasks.git


Navigate to the task folder:

cd Task-1_Iris_Exploration


Create virtual environment (optional but recommended):

python -m venv .venv


Activate environment:

Windows: .venv\Scripts\activate

Linux/WSL: source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt


Run the Python script:

python iris_exploration.py


(Repeat for other tasks using respective .py files)

👩‍💻 Author

Sehrish Shafiq

LinkedIn: https://www.linkedin.com/in/sehrish-shafiq
