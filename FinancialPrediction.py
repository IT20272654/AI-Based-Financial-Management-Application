

!sudo apt install tesseract-ocr
!pip install pytesseract opencv-python pillow

from google.colab import files
import cv2
import pytesseract
import re
from PIL import Image
from datetime import datetime

uploaded = files.upload()
image_path = next(iter(uploaded))

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

text_raw = pytesseract.image_to_string(Image.fromarray(gray))
text_thresh = pytesseract.image_to_string(Image.fromarray(thresh))
text = text_thresh if len(text_thresh.strip()) > len(text_raw.strip()) else text_raw

def extract_final_info(text):
    result = {}

    raw_amounts = re.findall(r'(?:Rs\.?|LKR)?\s*[\.:]?\s*([\d]{1,3}(?:,\d{3})*(?:\.\d{2})?)', text, re.IGNORECASE)
    cleaned_amounts = []
    for amt in raw_amounts:
        amt_clean = amt.replace(',', '')
        try:
            val = float(amt_clean)
            if val > 0:
                cleaned_amounts.append(val)
        except:
            pass
    if cleaned_amounts:
        result['Final Amount'] = f"{max(cleaned_amounts):,.2f}"

    date_match = re.findall(r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', text)
    if date_match:
        result['Receipt Date'] = date_match[0]

    return result

final_info = extract_final_info(text)

today_date = datetime.today().strftime('%d/%m/%Y')

category = "Bill Payment"
sub_category = "Receipt"

print("\n📊 Extracted Data:")
print(f"Date: {today_date}")
if 'Receipt Date' in final_info:
    print(f"Date on Receipt: {final_info['Receipt Date']}")
print(f"Category: {category}")
print(f"Sub Category: {sub_category}")

if 'Final Amount' in final_info:
    print(f"Final Amount: Rs. {final_info['Final Amount']}")
else:
    print("[No amount found!]")

from google.colab import files
import cv2
import pytesseract
import re
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import numpy as np
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

financial_data = pd.DataFrame(columns=['Date', 'Category', 'SubCategory', 'Amount', 'Type'])

def process_receipt():
    print("\n Upload your receipt image:")
    uploaded = files.upload()
    image_path = next(iter(uploaded))

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    text_raw = pytesseract.image_to_string(Image.fromarray(gray))
    text_thresh = pytesseract.image_to_string(Image.fromarray(thresh))
    text = text_thresh if len(text_thresh.strip()) > len(text_raw.strip()) else text_raw

    result = extract_receipt_info(text)

    print("\n Extracted Data:")
    date = result.get('Receipt Date', datetime.today().strftime('%d/%m/%Y'))
    category = result.get('Category', 'Bill Payment')
    sub_category = result.get('SubCategory', 'Receipt')

    print(f"Date: {date}")
    print(f"Category: {category}")
    print(f"Sub Category: {sub_category}")

    if 'Final Amount' in result:
        amount = float(result['Final Amount'].replace(',', ''))
        print(f"Final Amount: Rs. {result['Final Amount']}")
    else:
        print("[No amount found!]")
        amount = 0

    transaction_type = "Expense"
    add_financial_entry(date, category, sub_category, amount, transaction_type)

    return True

def extract_receipt_info(text):
    result = {}
    raw_amounts = re.findall(r'(?:Rs\.?|LKR)?\s*[\.:]?\s*([\d]{1,3}(?:,\d{3})*(?:\.\d{2})?)', text, re.IGNORECASE)
    cleaned_amounts = []
    for amt in raw_amounts:
        amt_clean = amt.replace(',', '')
        try:
            val = float(amt_clean)
            if val > 0:
                cleaned_amounts.append(val)
        except:
            pass
    if cleaned_amounts:
        result['Final Amount'] = f"{max(cleaned_amounts):,.2f}"

    date_match = re.findall(r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', text)
    if date_match:
        result['Receipt Date'] = date_match[0]

    text_lower = text.lower()
    if any(word in text_lower for word in ['food', 'restaurant', 'cafe', 'meal']):
        result['Category'] = 'Food'
        result['SubCategory'] = 'Restaurant'
    elif any(word in text_lower for word in ['grocery', 'market', 'supermarket']):
        result['Category'] = 'Food'
        result['SubCategory'] = 'Groceries'
    elif any(word in text_lower for word in ['transport', 'taxi', 'uber', 'bus', 'train']):
        result['Category'] = 'Transportation'
        result['SubCategory'] = 'Public Transport'

    return result

def add_manual_entry():
    print("\n Enter financial details:")

    date = input("Date [today]: ") or datetime.today().strftime('%d/%m/%Y')
    category = input("Category: ")
    sub_category = input("Sub Category: ")

    amount_str = input("Amount (Rs.): ")
    try:
        amount = float(amount_str.replace(',', ''))
    except ValueError:
        print("Invalid amount! Please enter a valid number.")
        return False

    transaction_type = input("Type (Income/Expense): ")
    if transaction_type.lower() not in ['income', 'expense']:
        print("Invalid type! Please enter either 'Income' or 'Expense'.")
        return False

    add_financial_entry(date, category, sub_category, amount, transaction_type)
    return True

def add_financial_entry(date, category, sub_category, amount, transaction_type):
    global financial_data
    new_entry = pd.DataFrame({
        'Date': [date],
        'Category': [category],
        'SubCategory': [sub_category],
        'Amount': [amount],
        'Type': [transaction_type]
    })
    financial_data = pd.concat([financial_data, new_entry], ignore_index=True)
    print("\n Entry added successfully!")

def visualize_data():
    global financial_data
    if financial_data.empty:
        print("No data to visualize! Add some entries first.")
        return

    financial_data['Date'] = pd.to_datetime(financial_data['Date'], dayfirst=True)
    financial_data['Amount'] = pd.to_numeric(financial_data['Amount'])

    current_month = datetime.now().month
    current_year = datetime.now().year
    current_month_data = financial_data[
        (financial_data['Date'].dt.month == current_month) &
        (financial_data['Date'].dt.year == current_year)
    ]

    plt.figure(figsize=(18, 12))
    plt.suptitle(f'Financial Data Analysis - {datetime.now().strftime("%B %Y")}', fontsize=16)

    plt.subplot(2, 2, 1)
    type_sums = financial_data.groupby('Type')['Amount'].sum()
    if not type_sums.empty:
        plt.pie(type_sums, labels=type_sums.index, autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336'], startangle=90)
        plt.title('Income vs Expense Distribution')
    else:
        plt.text(0.5, 0.5, 'No data available', ha='center', va='center')

    plt.subplot(2, 2, 2)
    if not current_month_data.empty:
        weekly_data = current_month_data.set_index('Date').groupby(
            [pd.Grouper(freq='W'), 'Type'])['Amount'].sum().unstack(fill_value=0)

        if weekly_data.shape[0] >= 2:
            weekly_data.plot(
                kind='line', marker='o', ax=plt.gca(),
                color={'Income': '#4CAF50', 'Expense': '#F44336'}
            )
            plt.title('Weekly Trends (Current Month)')
            plt.ylabel('Amount (Rs.)')
            plt.grid(True)
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d %b'))
        elif weekly_data.shape[0] == 1:
            weekly_data.plot(kind='bar', ax=plt.gca(), color=['#4CAF50', '#F44336'])
            plt.title('Weekly Total (Only One Week Found)')
            plt.ylabel('Amount (Rs.)')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'No weekly data to show', ha='center', va='center')
    else:
        plt.text(0.5, 0.5, 'No data for current month', ha='center', va='center')

    plt.subplot(2, 2, 3)
    expenses = financial_data[financial_data['Type'].str.lower() == 'expense']
    if not expenses.empty:
        category_sums = expenses.groupby('Category')['Amount'].sum()
        plt.pie(category_sums, labels=category_sums.index, autopct='%1.1f%%',
                colors=sns.color_palette('pastel'), startangle=90)
        plt.title('Expense Breakdown by Category')
    else:
        plt.text(0.5, 0.5, 'No expense data available', ha='center', va='center')

    plt.subplot(2, 2, 4)
    top_transactions = financial_data.nlargest(10, 'Amount')
    if not top_transactions.empty:
        sns.barplot(
            x='Description', y='Amount', hue='Type',
            data=top_transactions.assign(Description=top_transactions['Category'] + " - " + top_transactions['SubCategory']),
            palette={'Income': '#4CAF50', 'Expense': '#F44336'}
        )
        plt.title('Top 10 Transactions')
        plt.ylabel('Amount (Rs.)')
        plt.xlabel('Transaction Description')
        plt.xticks(rotation=90)
    else:
        plt.text(0.5, 0.5, 'No transaction data available', ha='center', va='center')

    plt.tight_layout()
    plt.show()

def show_data():
    if financial_data.empty:
        print("No data available! Add some entries first.")
        return

    print("\n Current Financial Data:")
    print(financial_data.to_string(index=False))

    income_sum = financial_data[financial_data['Type'].str.lower() == 'income']['Amount'].sum()
    expense_sum = financial_data[financial_data['Type'].str.lower() == 'expense']['Amount'].sum()
    balance = income_sum - expense_sum

    print("\n Summary:")
    print(f"Total Income: Rs. {income_sum:,.2f}")
    print(f"Total Expenses: Rs. {expense_sum:,.2f}")
    print(f"Balance: Rs. {balance:,.2f}")

def main():
    print("\n Financial Tracking System 🏦")

    while True:
        print("\nOptions:")
        print("1. Add financial data using receipt")
        print("2. Add financial data manually")
        print("3. Show current data")
        print("4. Visualize data")
        print("5. Export data to CSV")
        print("6. Exit (or type 'done')")

        choice = input("Choice (1-6): ").strip().lower()

        if choice in ['6', 'done', 'exit']:
            income_sum = financial_data[financial_data['Type'].str.lower() == 'income']['Amount'].sum()
            expense_sum = financial_data[financial_data['Type'].str.lower() == 'expense']['Amount'].sum()
            balance = income_sum - expense_sum

            print("\n Final Financial Situation:")
            print(f"Total Income: Rs. {income_sum:,.2f}")
            print(f"Total Expenses: Rs. {expense_sum:,.2f}")
            print(f"Balance: Rs. {balance:,.2f}")
            break
        elif choice == '1':
            process_receipt()
        elif choice == '2':
            add_manual_entry()
        elif choice == '3':
            show_data()
        elif choice == '4':
            visualize_data()
        elif choice == '5':
            if not financial_data.empty:
                filename = input("Filename (without .csv): ") or "financial_data"
                financial_data.to_csv(f"{filename}.csv", index=False)
                print(f"Data saved to {filename}.csv")
            else:
                print("No data to save!")
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from datetime import datetime, timedelta

# ================== DATA LOADING & CLEANING ==================
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

# Clean data
columns_to_drop = ['Institution', 'Description', 'Country', 'Recurrent', 'Tax Deduction']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df['Amount_LKR'] = df['Amount'] * 10  # USD to LKR conversion
df.dropna(subset=['Amount_LKR', 'Date'], inplace=True)

# ================== TIME SERIES PREPROCESSING ==================
daily = df.groupby('Date')['Amount_LKR'].sum().resample('D').sum()

# Display data date range
print(f"\n📅 Data ranges from {daily.index.min().date()} to {daily.index.max().date()}")

# Outlier treatment using IQR
Q1 = daily.quantile(0.25)
Q3 = daily.quantile(0.75)
IQR = Q3 - Q1
daily = daily.where(
    (daily >= (Q1 - 1.5 * IQR)) & (daily <= (Q3 + 1.5 * IQR)),
    daily.median()
)

# Smoothing with 7-day rolling average
data_smoothed = daily.rolling(window=7, min_periods=1).mean()

# ================== FEATURE ENGINEERING ==================
def create_features(df):
    """Create time-series features"""
    features = pd.DataFrame(index=df.index)
    features['amount'] = df.values
    features['day_of_week'] = df.index.dayofweek
    features['month'] = df.index.month
    features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)

    # Lag features (1-7 days)
    for lag in range(1, 8):
        features[f'lag_{lag}'] = features['amount'].shift(lag)

    # Moving averages
    features['ma_3'] = features['amount'].rolling(window=3).mean()
    features['ma_7'] = features['amount'].rolling(window=7).mean()

    return features

features = create_features(data_smoothed)
features.dropna(inplace=True)

# ================== MODEL TRAINING ==================
X = features.drop(columns=['amount'])
y = features['amount']

# Time-based split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# XGBoost model with simpler configuration
model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    objective='reg:squarederror',
    random_state=42
)

model.fit(X_train, y_train)

# ================== EVALUATION ==================
y_pred = model.predict(X_test)

print("\n🔹 Model Evaluation:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# ================== USER DATE INPUT ==================
# Get user input for the forecast start date
def get_user_date():
    date_range_info = f"Available data range: {features.index.min().date()} to {features.index.max().date()}"

    while True:
        try:
            date_input = input("\nEnter the date to start forecasting from (YYYY-MM-DD): ")
            input_date = pd.to_datetime(date_input)

            print(f"✅ Forecast will start from: {input_date.date()}")
            return input_date

        except ValueError as e:
            print(f"❌ Error: {e}. Please enter a valid date.")
        except Exception as e:
            print(f"❌ Unexpected error: {e}. Please try again.")

# Get the most recent data available for model features
def get_prediction_features():
    # Get the most recent date in the dataset with complete features
    latest_date = features.index.max()
    return features.loc[latest_date]

# ================== FUTURE PREDICTION ==================
def generate_dates(start_date, days=7):
    """Generate future dates starting from a specific date"""
    return [start_date + pd.Timedelta(days=i+1) for i in range(days)]

def forecast_future(model, latest_features, feature_columns, target_date, future_days=7):
    """
    Forecast for any future date by:
    1. Using the latest available data as a starting point
    2. Recursively predicting until we reach the target date
    3. Then continuing for the requested forecast period
    """
    current_features = latest_features.copy()
    latest_date = current_features.name

    # If target date is after our latest data, we need to bridge the gap
    if target_date > latest_date:
        # Calculate days between latest data and target date
        days_to_bridge = (target_date - latest_date).days

        # Generate intermediate dates (we'll need these for forecasting)
        bridge_dates = [latest_date + pd.Timedelta(days=i+1) for i in range(days_to_bridge)]

        # Forecast until we reach target date (these predictions won't be shown)
        for next_date in bridge_dates:
            # Make prediction using current features
            current_features_ordered = current_features[feature_columns]
            pred = model.predict(current_features_ordered.values.reshape(1, -1))[0]

            # Update features for next day
            for lag in range(7, 1, -1):
                current_features[f'lag_{lag}'] = current_features[f'lag_{lag-1}']
            current_features['lag_1'] = pred

            # Update moving averages
            current_features['ma_3'] = np.mean([current_features['lag_1'],
                                             current_features['lag_2'],
                                             current_features['lag_3']])
            current_features['ma_7'] = np.mean([current_features[f'lag_{i}'] for i in range(1, 8)])

            # Update date-based features
            current_features.name = next_date
            current_features['day_of_week'] = next_date.dayofweek
            current_features['month'] = next_date.month
            current_features['is_weekend'] = 1 if next_date.dayofweek >= 5 else 0

    # Now we've reached or are at target_date, forecast next X days
    future_dates = generate_dates(target_date, days=future_days)
    forecast_values = []

    # Generate forecasts for the requested period
    for next_date in future_dates:
        # Make prediction using current features
        current_features_ordered = current_features[feature_columns]
        pred = model.predict(current_features_ordered.values.reshape(1, -1))[0]
        forecast_values.append(pred)

        # Update features for next day
        for lag in range(7, 1, -1):
            current_features[f'lag_{lag}'] = current_features[f'lag_{lag-1}']
        current_features['lag_1'] = pred

        # Update moving averages
        current_features['ma_3'] = np.mean([current_features['lag_1'],
                                         current_features['lag_2'],
                                         current_features['lag_3']])
        current_features['ma_7'] = np.mean([current_features[f'lag_{i}'] for i in range(1, 8)])

        # Update date-based features
        current_features.name = next_date
        current_features['day_of_week'] = next_date.dayofweek
        current_features['month'] = next_date.month
        current_features['is_weekend'] = 1 if next_date.dayofweek >= 5 else 0

    return forecast_values, future_dates

# Get user input date
forecast_start_date = get_user_date()
print(f"\n🔍 Generating 7-day forecast starting from: {forecast_start_date.date()}")

# Get the latest features available
latest_features = get_prediction_features()
feature_columns = X_train.columns.tolist()

# Generate forecast
future_pred, future_dates = forecast_future(
    model,
    latest_features,
    feature_columns,
    forecast_start_date
)

# ================== VISUALIZATION ==================
plt.figure(figsize=(14, 7))

# Add historical data if the forecast date is within our data range
if forecast_start_date <= features.index.max():
    # Show 30 days of historical data leading up to forecast date
    lookback_days = 30
    past_start = forecast_start_date - pd.Timedelta(days=lookback_days)
    past_dates = pd.date_range(start=past_start, end=forecast_start_date)
    past_data = data_smoothed.reindex(past_dates).fillna(method='ffill')
    plt.plot(past_data.index, past_data.values, label='Historical', color='blue')

# Plot forecast
plt.plot(future_dates, future_pred, label='Forecast', color='red', linestyle='--')
plt.axvline(x=forecast_start_date, color='green', linestyle=':', label='Forecast Start')
plt.title(f'7-Day Spending Forecast (Starting from {forecast_start_date.date()})')
plt.ylabel('Amount (LKR)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ================== PREDICTION SUMMARY ==================
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Amount_LKR': future_pred
})

print("\n📈 Next 7 Days Forecast:")
print(forecast_df.set_index('Date').round(2))

print(f"\n💵 Total Projected Spending: {forecast_df['Predicted_Amount_LKR'].sum():,.2f} LKR")
print(f"📊 Daily Average: {forecast_df['Predicted_Amount_LKR'].mean():,.2f} LKR")
print(f"📆 Highest Spending Day: {forecast_df.loc[forecast_df['Predicted_Amount_LKR'].idxmax(), 'Date'].date()} ({forecast_df['Predicted_Amount_LKR'].max():,.2f} LKR)")
print(f"📆 Lowest Spending Day: {forecast_df.loc[forecast_df['Predicted_Amount_LKR'].idxmin(), 'Date'].date()} ({forecast_df['Predicted_Amount_LKR'].min():,.2f} LKR)")

# Only calculate week-over-week if the forecast date is within our data range
if forecast_start_date <= features.index.max():
    # Calculate financial insights
    previous_week_data = data_smoothed[forecast_start_date - pd.Timedelta(days=7):forecast_start_date]
    previous_week_total = previous_week_data.sum()
    forecast_total = forecast_df['Predicted_Amount_LKR'].sum()
    percent_change = ((forecast_total - previous_week_total) / previous_week_total * 100) if previous_week_total > 0 else 0

    print("\n📊 Financial Insight:")
    print(f"Previous 7-day total: {previous_week_total:,.2f} LKR")
    print(f"Forecast 7-day total: {forecast_total:,.2f} LKR")
    print(f"Week-over-week change: {percent_change:+.2f}%")

# Save the forecast to CSV
forecast_df.to_csv(f'spending_forecast_{forecast_start_date.date()}.csv', index=False)
print(f"\n💾 Forecast saved to spending_forecast_{forecast_start_date.date()}.csv")