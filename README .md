# Housing Price Predictor

A machine learning project that predicts house prices based on property features like area, location, bedrooms, and furnishing status. Built entirely from scratch using only NumPy, Pandas, and Matplotlib — no sklearn.

---

## Problem Statement

Buying or renting a house is one of the biggest financial decisions a person makes. Yet most people have no reliable way to know if a listed price is fair or inflated. This project builds a price prediction model using key property features so anyone can get an estimated market value instantly.

---

## Features Used

| Feature | Description |
|---|---|
| `area_sqft` | Total area of the house in square feet |
| `bedrooms` | Number of bedrooms (1–5) |
| `bathrooms` | Number of bathrooms (1–4) |
| `floors` | Number of floors (1–3) |
| `age_years` | Age of the house in years |
| `parking` | Number of parking spots (0–2) |
| `location_enc` | Location type: 0=Rural, 1=Suburban, 2=Urban, 3=Prime |
| `furnished_enc` | 0=Unfurnished, 1=Semi-furnished, 2=Fully furnished |
| `road_access` | Whether the house has main road access (1=Yes, 0=No) |

---

## ML Models (all built from scratch)

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | Rs 13.2L | Rs 16.5L | 0.94 |
| Decision Tree | Rs 3.6L | Rs 4.9L | 0.99 |
| **Random Forest ★** | **Rs 3.3L** | **Rs 4.3L** | **1.00** |

**Best model: Random Forest** — an ensemble of 30 decision trees using bootstrap sampling (bagging). No sklearn used anywhere.

---

## Project Structure

```
housing/
├── housing.py              # main script — all code in one file
├── data/
│   └── housing_data.csv    # auto-generated dataset (800 records)
├── model/
│   ├── model.pkl           # trained model (saved after running)
│   └── meta.json           # model metadata (R2, RMSE, size)
├── charts/
│   ├── analysis.png        # EDA chart (6 panels)
│   └── evaluation.png      # model evaluation chart
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib
```

### 2. Run the full pipeline

```bash
python housing.py
```

This will:
- Generate the dataset
- Show EDA charts
- Train all 3 models and compare them
- Save the best model
- Optionally run the interactive predictor

### 3. Run only the predictor (after training once)

```bash
python housing.py --predict
```

### 4. Run only the EDA charts

```bash
python housing.py --eda
```

---

## Interactive Predictor

When you run `--predict`, the program asks you 9 questions about a house:

```
  Area in sqft (e.g. 1200)           : 1500
  Bedrooms (1-5)                     : 3
  Bathrooms (1-4)                    : 2
  Number of floors (1-3)             : 2
  Age of house in years              : 5
  Parking spots (0/1/2)              : 1
  Location type (0-3)                : 2
  Furnished status (0-2)             : 1
  Main road access? (1=Yes / 0=No)  : 1
```

**Sample output:**

```
  ┌──────────────────────────────────────────┐
  │         PRICE ESTIMATE                   │
  ├──────────────────────────────────────────┤
  │  Area       : 1500 sqft
  │  Bedrooms   : 3   Bathrooms: 2
  │  Location   : Urban
  │  Furnished  : Semi-furnished
  │  Age        : 5 years
  ├──────────────────────────────────────────┤
  │  Estimated Price : Rs 54.30 Lakhs
  │                  = Rs 54,30,000
  └──────────────────────────────────────────┘
  Likely range: Rs 48.87L  —  Rs 59.73L
```

---

## EDA Charts (one page, 6 panels)

1. **Price Distribution** — histogram of all house prices
2. **Area vs Price** — scatter plot coloured by location type
3. **Avg Price by Location** — Rural / Suburban / Urban / Prime comparison
4. **Avg Price by Bedrooms** — how bedrooms affect price
5. **Age vs Price** — trend line showing how older houses lose value
6. **Price by Furnished Status** — boxplot comparison

---

## Key Findings from the Data

- **Location is the biggest price driver** — a Prime location house costs 2.5× more than the same house in a Rural area
- **Area has the strongest linear correlation** with price
- **Older houses** (20+ years) show a clear downward price trend
- **Furnishing** adds roughly Rs 1–3 Lakhs to the estimated value
- **Random Forest outperforms** both Linear Regression and a single Decision Tree by capturing non-linear patterns

---

## Dependencies

```
pandas
numpy
matplotlib
```

No sklearn, no tensorflow, no external ML libraries.

---

## Course

Fundamentals of AI/ML — VITyarthi BYOP Capstone Project
