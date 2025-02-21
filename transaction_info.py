import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

# Load the Excel file
file_path = "data/users/Bank_Statement.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Convert Txn Date to datetime format
df["Txn Date"] = pd.to_datetime(df["Txn Date"])

# Aggregate transactions on the same date by summing up Credit, Debit, and Balance
df_grouped = df.groupby("Txn Date", as_index=False).agg({
    "Description": lambda x: " | ".join(x.dropna().astype(str)),  # Combine descriptions
    "Debit": "sum",
    "Credit": "sum",
    "Balance": "last"
})

# Define the full date range for transactions
full_date_range = pd.date_range(start=df_grouped["Txn Date"].min(), end=df_grouped["Txn Date"].max())

# Create a new DataFrame with all dates in the range
full_df = pd.DataFrame({"Txn Date": full_date_range})

# Merge with the grouped data to fill in missing dates
merged_df = full_df.merge(df_grouped, on="Txn Date", how="left")

# Fill missing values
merged_df["Description"] = merged_df["Description"].fillna("-")
merged_df["Debit"] = merged_df["Debit"].fillna(0)
merged_df["Credit"] = merged_df["Credit"].fillna(0)

# Forward-fill the Balance column
merged_df["Balance"] = merged_df["Balance"].ffill()

# Calculate 720-day rolling averages
merged_df['Avg Credit 720D'] = merged_df['Credit'].rolling(window=720, min_periods=720).mean()
merged_df['Avg Debit 720D'] = merged_df['Debit'].rolling(window=720, min_periods=720).mean()

# Compute Relative Strength (RS), avoiding division by zero
merged_df['RS'] = merged_df['Avg Credit 720D'] / merged_df['Avg Debit 720D'].replace(0, np.nan)

# Normalize RS to a scale of 0 to 100
merged_df['RS Normalized'] = (merged_df['RS'] - merged_df['RS'].min()) / (merged_df['RS'].max() - merged_df['RS'].min()) * 100

# Calculate 720-day moving average of balance
merged_df['Balance MA 720D'] = merged_df['Balance'].rolling(window=720, min_periods=720).mean()

# Filter the DataFrame to include only the last 1 year of data
end_date = merged_df['Txn Date'].max()
start_date = end_date - pd.DateOffset(years=1)
last_year_df = merged_df[(merged_df['Txn Date'] >= start_date) & (merged_df['Txn Date'] <= end_date)]

# Calculate average RS for the last year
average_rs_year = last_year_df['RS Normalized'].mean()

# Calculate average RS within the range of 30 to 70 for the last year
valid_rs = last_year_df[(last_year_df['RS Normalized'] >= 30) & (last_year_df['RS Normalized'] <= 70)]['RS Normalized']
average_rs = valid_rs.mean()

# Calculate average balance for the last year
average_balance = last_year_df['Balance'].mean()

# Trend Analysis: Up Trend (>70), Down Trend (<30), Sideways Trend (30-70)
up_trend_df = last_year_df[last_year_df['RS Normalized'] > 70]
down_trend_df = last_year_df[last_year_df['RS Normalized'] < 30]
sideways_trend_df = last_year_df[(last_year_df['RS Normalized'] >= 30) & (last_year_df['RS Normalized'] <= 70)]

total_days = len(last_year_df)

# Calculate percentage of time in each trend
up_trend_percentage = (len(up_trend_df) / total_days) * 100
down_trend_percentage = (len(down_trend_df) / total_days) * 100
sideways_trend_percentage = (len(sideways_trend_df) / total_days) * 100

# Calculate average RS for each trend
avg_rs_up_trend = up_trend_df['RS Normalized'].mean()
avg_rs_down_trend = down_trend_df['RS Normalized'].mean()
avg_rs_sideways_trend = sideways_trend_df['RS Normalized'].mean()

print(f"Average RS for the last year: {average_rs_year:.2f}")
print(f"""
The all 3 trends for the past year, in the following format (% of time, Average RS):
  Up-Trend (>70): ({up_trend_percentage:.2f}%, {avg_rs_up_trend:.2f})
  Down-Trend (<30): ({down_trend_percentage:.2f}%, {avg_rs_down_trend:.2f})
  Sideways-Trend (30-70): ({sideways_trend_percentage:.2f}%, {avg_rs_sideways_trend:.2f})
""")

# Group dates where RS is outside 30-70 by month for the last year
out_of_range_dates = last_year_df[(last_year_df['RS Normalized'] > 70) | (last_year_df['RS Normalized'] < 30)]
out_of_range_summary = {}
for month, group in out_of_range_dates.groupby(out_of_range_dates['Txn Date'].dt.month):
    out_of_range_summary[month] = {
        date: rs for date, rs in zip(group['Txn Date'].dt.strftime('%Y-%m-%d'), group['RS Normalized'])
    }

print("Out of range RS summary (last year):")
print(out_of_range_summary)

# Plotting the data
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.2)

def show_rs(event):
    ax.clear()
    ax.plot(merged_df['Txn Date'], merged_df['RS Normalized'], label='Normalized RS', color='purple')

    # Add trend lines
    ax.axhline(y=70, color='green', linestyle='dashed')
    ax.axhline(y=30, color='red', linestyle='dashed')

    # Fill Sideways Trend (base layer)
    ax.fill_between(merged_df['Txn Date'], 30, 70, color='lightgray', alpha=0.2)

    # Fill Up Trend (>70) and Down Trend (<30) on top
    ax.fill_between(merged_df['Txn Date'], 30, 70, where=(merged_df['RS Normalized'] > 70),
                    color='lightgreen', alpha=0.3)
    ax.fill_between(merged_df['Txn Date'], 30, 70, where=(merged_df['RS Normalized'] < 30),
                    color='lightpink', alpha=0.3)

    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized RS (0-100)")
    ax.set_title("Normalized Relative Strength (RS) Over Time (720-Day Window)")

    ax.legend()
    ax.set_xlim([start_date, end_date])

    plt.draw()

def show_balance(event):
    ax.clear()
    ax.plot(merged_df['Txn Date'], merged_df['Balance'], label="Balance", color='blue', alpha=0.6)
    ax.plot(merged_df['Txn Date'], merged_df['Balance MA 720D'], label="720-Day Moving Average",
            color='orange', linestyle="dashed")
    ax.set_xlabel("Date")
    ax.set_ylabel("Balance")
    ax.set_title("Balance and 720-Day Moving Average")
    ax.legend()
    ax.set_xlim([start_date, end_date])

    plt.draw()

# Create buttons for navigation
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bprev = Button(axprev, 'RS View')
bnext = Button(axnext, 'Balance View')
bprev.on_clicked(show_rs)
bnext.on_clicked(show_balance)

# Show the initial plot
show_rs(None)
plt.show()

# Rename 'Txn Date' & save the updated dataframe to a new Excel file
merged_df.rename(columns={"Txn Date": "Date"}, inplace=True)
output_file = "data/users/Daily_Bank_Statement.xlsx"
columns_to_store = ["Date", "Description", "Debit", "Credit", "Balance", "RS Normalized", "Balance MA 720D"]
merged_df[columns_to_store].to_excel(output_file, index=False)

print(f"Daily bank statement saved as {output_file}")
