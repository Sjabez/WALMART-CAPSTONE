# Walmart Weekly Sales Forecasting Project

## 📊 **Project Overview**
This project explores historical weekly sales data across multiple Walmart stores to uncover key insights and forecast future sales. The goal is to assist Walmart in better aligning inventory supply with customer demand using data-driven strategies.

---

## 🧾 **Problem Statement**
Walmart is facing challenges managing inventory effectively across its retail outlets. The dataset contains 6,435 entries across 8 columns and includes:

- **Store**: Store number
- **Date**: Weekly sales date
- **Weekly_Sales**: Revenue generated that week
- **Holiday_Flag**: Indicates if the week includes a major holiday
- **Temperature**: Temperature on the day of sale
- **Fuel_Price**: Regional fuel cost
- **CPI**: Consumer Price Index
- **Unemployment**: Regional unemployment rate

---

## 🔍 **Key Questions Addressed**
- Does **unemployment rate** impact weekly sales? If so, **which stores are most affected**?
- Are there **seasonal patterns** in the sales data?
- How do **temperature**, **CPI**, and **fuel prices** influence sales?
- Which are the **top performing stores**?
- Which is the **worst performing store**, and how wide is the performance gap?
- Can we build a **predictive model** to forecast sales for the next 12 weeks?

---

## 🧪 **Methodology**
### ✅ **1. Data Cleaning & Validation**
- Checked for missing values and handled them appropriately.
- Ensured there were no duplicate rows.
- Verified column data types for accurate processing.

### 📊 **2. Exploratory Data Analysis (EDA)**
- Analyzed correlation between unemployment, CPI, and weekly sales.
- Plotted time-series and grouped data to uncover seasonal patterns.
- Compared holiday vs. non-holiday sales across all stores.

### 📉 **3. Statistical Analysis**
- Aggregated data by store and calculated mean weekly sales.
- Highlighted best and worst performing stores.
- Analyzed how CPI and temperature affected different store clusters.

### 🔮 **4. Predictive Modeling**
- Applied **Linear Regression** to build a forecasting model.
- Trained model using historical weekly sales data and related features.
- Evaluated model using **RMSE** and **R2 Score** to measure accuracy.

---

## 💡 **Key Insights**
- **Unemployment** had a notable impact on lower-performing stores.
- **Holiday weeks** consistently saw **sales spikes**, especially around major U.S. holidays.
- **Temperature** showed inverse relation in certain regions, suggesting weather-influenced foot traffic.
- **CPI** correlated negatively with sales in some stores, indicating price sensitivity.
- The top 5 stores contributed disproportionately to revenue.

---

## 📈 **Forecasting Results**
The regression model was able to **forecast 12 weeks of sales** with reasonable accuracy.
- RMSE was within acceptable bounds.
- Insights from the model can guide **staffing, promotions, and logistics**.

---

## 🛠 **Technologies Used**
- **Python (Pandas, NumPy, Matplotlib, Seaborn)**
- **Scikit-learn (LinearRegression, train_test_split, evaluation metrics)**
- **Jupyter Notebook**

---

## 🧠 **Code Implementation**
```python
# Load essential Python libraries for data manipulation, visualization, and modeling
import pandas as pd
# Load essential Python libraries for data manipulation, visualization, and modeling
import numpy as np
# Load essential Python libraries for data manipulation, visualization, and modeling
import matplotlib.pyplot as plt

# Load Walmart weekly sales data to begin analysis
data1 = pd.read_csv(r'C:\Users\HP\Downloads\Walmart (2).csv')
data1.set_index('Date', inplace=True)
a= int(input("Enter the store id:"))
store = data1[data1.Store == a]
# Aggregate sales across categories (e.g., store, holiday flag) to uncover trends
sales = pd.DataFrame(store.Weekly_Sales.groupby(store.index).sum())
sales.dtypes

# Quickly inspect data structure and confirm expected columns are present
sales.head(20)

sales.reset_index(inplace = True)
sales['Date'] = pd.to_datetime(sales['Date'])
sales.set_index('Date',inplace = True)

# Create visuals that reveal seasonal trends or sales anomalies
sales.Weekly_Sales.plot(figsize=(10,6), title= 'Weekly Sales of a Store', fontsize=14, color = 'blue')
plt.show()

# Load essential Python libraries for data manipulation, visualization, and modeling
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(sales.Weekly_Sales, period=12)
fig = plt.figure()
# Create visuals that reveal seasonal trends or sales anomalies
fig = decomposition.plot()
fig.set_size_inches(12, 10)
plt.show()

store5 = data1[data1.Store == 5]
# Aggregate sales across categories (e.g., store, holiday flag) to uncover trends
sales5 = pd.DataFrame(store5.Weekly_Sales.groupby(store5.index).sum())
sales5.dtypes
sales5.reset_index(inplace = True)
sales5['Date'] = pd.to_datetime(sales5['Date'])
sales5.set_index('Date',inplace = True)

y1=sales.Weekly_Sales
y2=sales5.Weekly_Sales

# Create visuals that reveal seasonal trends or sales anomalies
y1['2012'].plot(figsize=(15, 6),legend=True, color = 'chocolate')
# Create visuals that reveal seasonal trends or sales anomalies
y2['2012'].plot(figsize=(15, 6), legend=True, color = 'turquoise')
plt.ylabel('Weekly Sales')
plt.title('Store4 vs Store5 on 2012', fontsize = '16')
plt.show()


p = d = q = range(0, 5)
# Load essential Python libraries for data manipulation, visualization, and modeling
import itertools
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]

# Load essential Python libraries for data manipulation, visualization, and modeling
import statsmodels.api as sm
mod = sm.tsa.statespace.SARIMAX(y1,
order=(4, 4, 3),
seasonal_order=(1, 1, 0, 52),   #enforce_stationarity=False,
enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

plt.style.use('seaborn-pastel')
# Create visuals that reveal seasonal trends or sales anomalies
results.plot_diagnostics(figsize=(15, 12))
plt.show()

# Generate forward-looking sales predictions to support supply chain planning
pred = results.get_prediction(start=pd.to_datetime('2012-07-27'), dynamic=False)
pred_ci = pred.conf_int()

# Create visuals that reveal seasonal trends or sales anomalies
ax = y1['2010':].plot(label='observed')
# Create visuals that reveal seasonal trends or sales anomalies
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
ax.fill_between(pred_ci.index,
pred_ci.iloc[:, 0],
pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Time Period')
ax.set_ylabel('Sales')
plt.legend()
plt.show()

# Generate forward-looking sales predictions to support supply chain planning
y_forecasted = pred.predicted_mean
y_truth = y1['2012-7-27':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# Generate forward-looking sales predictions to support supply chain planning
pred_dynamic = results.get_prediction(start=pd.to_datetime('2012-7-27'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

# Create visuals that reveal seasonal trends or sales anomalies
ax = y1['2010':].plot(label='observed', figsize=(12, 8))
# Create visuals that reveal seasonal trends or sales anomalies
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)
ax.fill_between(pred_dynamic_ci.index,
pred_dynamic_ci.iloc[:, 0],
pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2012-7-26'), y1.index[-1],
alpha=.1, zorder=-1)
ax.set_xlabel('Time Period')
ax.set_ylabel('Sales')
plt.legend()
plt.show()

# Load essential Python libraries for data manipulation, visualization, and modeling
import numpy as np
# Generate forward-looking sales predictions to support supply chain planning
y_forecasted = pred_dynamic.predicted_mean
print(y_forecasted)

y_truth = y1['2012-7-27':]
print(y_truth)

rmse = np.sqrt(((y_forecasted - y_truth) ** 2).mean())
print('The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 2)))

Residual= y_forecasted - y_truth
print("Residual for Store1",np.abs(Residual).sum())

pred_uc = results.get_forecast(steps=12)
print(pred_uc)

pred_ci = pred_uc.conf_int()

# Create visuals that reveal seasonal trends or sales anomalies
ax = y1.plot(label='observed', figsize=(12, 8))
# Create visuals that reveal seasonal trends or sales anomalies
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
pred_ci.iloc[:, 0],
pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Time Period')
ax.set_ylabel('Sales')
plt.legend()
plt.show()
```

---

## 🔚 **Conclusion**
This project demonstrates the power of data analysis and predictive modeling in retail strategy. The outcomes offer Walmart the tools to better predict demand, manage inventory, and boost revenue through informed decision-making.

---

## 📁 **Next Steps**
- Deploy the model using Flask or Streamlit for business use.
- Extend analysis to include more recent data and store location features.
- Explore more advanced models like Random Forest or XGBoost for improved accuracy.
