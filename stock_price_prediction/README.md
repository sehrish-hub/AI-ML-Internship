# 📈 Task 2: Stock Price Prediction 

## 🎯 Objective
Use historical stock data to predict the next day's closing price using Python and machine learning.

---

## 📊 Dataset
- Source: Yahoo Finance
- Accessed via: `yfinance` Python library
- Example Stock: Apple (AAPL)  
> You can switch to Tesla (TSLA), Google (GOOGL), or any other stock by changing the stock symbol in `main.py`.

---

## 🛠 Tools & Libraries
- Python
- Pandas
- Matplotlib
- Scikit-learn
- yfinance

---

## 🧠 Model
- Linear Regression
- Features:
  - Open
  - High
  - Low
  - Volume
- Target:
  - Next day Close price

---

## 🔍 Steps Performed
1. Loaded stock data using `yfinance`.
2. Created next-day closing price target.
3. Trained Linear Regression model.
4. Evaluated predictions using Mean Absolute Error (MAE).
5. Visualized actual vs predicted prices and saved plots in `outputs/`.


---

## 📈 Output
- `outputs/stock_prediction.png` – Actual vs Predicted Closing Prices.

---

## ✅ Conclusion
The Linear Regression model provides a basic short-term prediction of stock prices.  
For improved performance, you can experiment with:
- Random Forest Regressor
- XGBoost
- LSTM or other deep learning models

---

## 👩‍💻 Author
**Sehrish Shafiq**




## 🚀 How to Run

1. Clone the repository or download the project folder.
2. Create a virtual environment:
```bash
python -m venv .venv
Activate the environment:

Windows

.venv\Scripts\activate


Linux / WSL

source .venv/bin/activate
Install dependencies:

pip install -r requirements.txt


Run the script:

python main.py


To switch stock, change the stock_symbol variable in main.py:

stock_symbol = "TSLA"  # Tesla
stock_symbol = "GOOGL" # Google

