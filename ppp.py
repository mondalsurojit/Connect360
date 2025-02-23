import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = "data/account.xlsx"
df = pd.read_excel(file_path)

# Define IFSC codes for Mumbai and Bilaspur
ifsc_mumbai = "SBIN0070001"
ifsc_bilaspur = "SBIN0000336"

# Filter data for Mumbai and Bilaspur
df_mumbai = df[df["IFSC_CODE"] == ifsc_mumbai]
df_bilaspur = df[df["IFSC_CODE"] == ifsc_bilaspur]

# Calculate average spending based on specified balance and RS value ranges
def average_spending_in_range(df):
    filtered_df = df[(df["BALANCE_LEVEL"] >= 12.5) & 
                     (df["BALANCE_LEVEL"] <= 25) & 
                     (df["RS_VALUE"] >= 12.5) & 
                     (df["RS_VALUE"] <= 87.5)]
    return filtered_df["AVG_DAILY_EXP"].mean()

# Calculate average spending for Mumbai and Bilaspur
avg_spending_mumbai = average_spending_in_range(df_mumbai)
avg_spending_bilaspur = average_spending_in_range(df_bilaspur)

# Calculate Purchasing Power of Bilaspur (relative to Mumbai)
purchasing_power_bilaspur = avg_spending_bilaspur / avg_spending_mumbai

# Define axis ticks (8 classes)
tick_values = np.linspace(0, 100, 9)
tick_labels = [round(t, 1) for t in tick_values]

# Highlight range for balance and RS values
def highlight_range(ax):
    ax.axhspan(12.5, 25, xmin=12.5/100, xmax=87.5/100, color='yellow', alpha=0.3)

# Draw percentile lines
def draw_percentile_lines(ax, df):
    specified_percentiles = [12.5, 25, 37.5, 50, 62.5, 75, 87.5]
    balance_percentiles = np.percentile(df["BALANCE_LEVEL"], specified_percentiles)
    rs_percentiles = np.percentile(df["RS_VALUE"], specified_percentiles)
    
    for p in balance_percentiles:
        ax.axhline(y=p, color='red', linestyle='--', alpha=0.5)
    for p in rs_percentiles:
        ax.axvline(x=p, color='blue', linestyle='--', alpha=0.5)

# Handle plotting
def plot_branch_data(ax, df, title):
    ax.scatter(df["RS_VALUE"], df["BALANCE_LEVEL"], alpha=0.5, s=10)
    ax.set_title(title)
    ax.set_xlabel("RS Value (0-100)")
    ax.set_ylabel("Balance Value (0-100)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)
    ax.grid(True, linestyle='--', alpha=0.6)
    highlight_range(ax)
    draw_percentile_lines(ax, df)

# Scatter plot setup
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_branch_data(axes[0], df_mumbai, "Mumbai Main Branch (SBIN0070001)")
plot_branch_data(axes[1], df_bilaspur, "Bilaspur Branch (SBIN0000336)")

# Display the results
plt.tight_layout()
plt.show()

# Print results
print(f"Mumbai Avg Spending (Balance 12.5-25, RS 12.5-87.5): ₹{avg_spending_mumbai:.2f}")
print(f"Bilaspur Avg Spending (Balance 12.5-25, RS 12.5-87.5): ₹{avg_spending_bilaspur:.2f}")
print(f"Purchasing Power of Bilaspur: {purchasing_power_bilaspur:.2f}")

# Function to calculate and print percentiles in a detailed format
def print_details(df, branch_name):
    specified_percentiles = [12.5, 25, 37.5, 50, 62.5, 75, 87.5]
    balance_percentiles = np.percentile(df["BALANCE_LEVEL"], specified_percentiles)
    print(f"\nBalance Level Percentiles for {branch_name}:")
    for p, value in zip(specified_percentiles, balance_percentiles):
        print(f"  {p}% percentile: {value:.2f}")

# Function to print the compact format
def print_compact(df, avg_spending, branch_name):
    specified_percentiles = [12.5, 25, 37.5, 50, 62.5, 75, 87.5]
    balance_percentiles = np.percentile(df["BALANCE_LEVEL"], specified_percentiles)
    formatted_output = {
        "ppp": round(float(avg_spending), 2),
        "blp": [round(float(value), 2) for value in balance_percentiles]
    }
    print(f"\n{branch_name}: {formatted_output}")

# Display detailed stats
print_details(df_mumbai, "Mumbai Main Branch (SBIN0070001)")
print_details(df_bilaspur, "Bilaspur Branch (SBIN0000336)")
print_compact(df_bilaspur, avg_spending_bilaspur, "Bilaspur (SBIN0000336)")
