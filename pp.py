import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = "data/account_data.xlsx"
df = pd.read_excel(file_path)

# Define IFSC codes for Mumbai and Bilaspur
ifsc_mumbai = "SBIN0070001"
ifsc_bilaspur = "SBIN0000336"

# Filter data for Mumbai and Bilaspur
df_mumbai = df[df["IFSC_CODE"] == ifsc_mumbai]
df_bilaspur = df[df["IFSC_CODE"] == ifsc_bilaspur]

# Function to calculate average spending based on specified balance and RS value ranges
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

# Scatter plot setup
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Define axis ticks (8 classes)
tick_values = np.linspace(0, 100, 9)
tick_labels = [round(t, 1) for t in tick_values]

# Force both graphs to show 0-100 range
for ax in axes:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

# Highlight range for balance (12.5 to 25) and RS value (12.5 to 87.5)
def highlight_range(ax):
    ax.axhspan(12.5, 25, xmin=12.5/100, xmax=87.5/100, color='yellow', alpha=0.3)

# Function to draw percentile lines at specified percentiles
def draw_percentile_lines(ax, df):
    specified_percentiles = [12.5, 25, 37.5, 50, 62.5, 75, 87.5]
    balance_percentiles = np.percentile(df["BALANCE_LEVEL"], specified_percentiles)
    rs_percentiles = np.percentile(df["RS_VALUE"], specified_percentiles)
    
    for p in balance_percentiles:
        ax.axhline(y=p, color='red', linestyle='--', alpha=0.5)
    for p in rs_percentiles:
        ax.axvline(x=p, color='blue', linestyle='--', alpha=0.5)

# Plot Mumbai data
axes[0].scatter(df_mumbai["RS_VALUE"], df_mumbai["BALANCE_LEVEL"], alpha=0.5, s=10)
axes[0].set_title("Mumbai Main Branch (SBIN0070001)")
axes[0].set_xlabel("RS Value (0-100)")
axes[0].set_ylabel("Balance Value (0-100)")
axes[0].set_xticks(tick_values)
axes[0].set_yticks(tick_values)
axes[0].grid(True, linestyle='--', alpha=0.6)
highlight_range(axes[0])  # Highlight the specified region
draw_percentile_lines(axes[0], df_mumbai)  # Draw percentile lines based on Mumbai data

# Plot Bilaspur data
axes[1].scatter(df_bilaspur["RS_VALUE"], df_bilaspur["BALANCE_LEVEL"], alpha=0.5, s=10)
axes[1].set_title("Bilaspur Branch (SBIN0000336)")
axes[1].set_xlabel("RS Value (0-100)")
axes[1].set_xticks(tick_values)
axes[1].set_yticks(tick_values)
axes[1].grid(True, linestyle='--', alpha=0.6)
highlight_range(axes[1])  # Highlight the specified region
draw_percentile_lines(axes[1], df_bilaspur)  # Draw percentile lines based on Bilaspur data

# Display the results
plt.tight_layout()
plt.show()

# Print results
print(f"Mumbai Avg Spending (Balance 12.5-25, RS 12.5-87.5): ₹{avg_spending_mumbai:.2f}")
print(f"Bilaspur Avg Spending (Balance 12.5-25, RS 12.5-87.5): ₹{avg_spending_bilaspur:.2f}")
print(f"Purchasing Power of Bilaspur: {purchasing_power_bilaspur:.2f}")
