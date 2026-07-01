# Fraud Detection App

Predicts whether a credit card transaction is **fraudulent or legitimate** using machine learning.

---

## Requirements

- Python 3.10 or higher
- pip

---

## Setup

**1. copy these files to your system:**
```
app.py
best_model.pkl
features.pkl
requirements.txt
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the app:**
```bash
python -m streamlit run app.py --server.port 8501
```

**4. Open in browser:**
```
http://localhost:8501
```

---

## How to Use

Fill in the form:

| Field | What to enter |
|---|---|
| Transaction Amount | How much was spent |
| Hour / Day / Month | When the transaction happened |
| Customer Age | Age of the cardholder |
| Category Encoded | Merchant category as a number (0–13) |
| Gender | M or F |
| City Population | Population of the customer's city |
| Customer Lat/Long | GPS coordinates of customer's home |
| Merchant Lat/Long | GPS coordinates of the merchant |

Click **Predict Fraud Risk** → get result instantly.

---

## Output

- ✅ **Legitimate** — transaction looks normal
- 🚨 **Fraudulent** — transaction flagged as suspicious
- Shows fraud probability % and distance between customer and merchant

---

## Category Encoded Values

| Number | Category |
|---|---|
| 0 | entertainment |
| 1 | food_dining |
| 2 | gas_transport |
| 3 | grocery_net |
| 4 | grocery_pos |
| 5 | health_fitness |
| 6 | home |
| 7 | kids_pets |
| 8 | misc_net |
| 9 | misc_pos |
| 10 | personal_care |
| 11 | shopping_net |
| 12 | shopping_pos |
| 13 | travel |

---

## Files Needed

| File | Purpose |
|---|---|
| `app.py` | Main web application |
| `best_model.pkl` | Trained ML model |
| `features.pkl` | Feature names used by model |
| `requirements.txt` | Python dependencies |
