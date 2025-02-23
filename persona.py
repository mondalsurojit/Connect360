import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

# ------------------------------
# Utility Functions
# ------------------------------

def load_and_preprocess_data(file_path):
    """Load, clean, and preprocess bank statement data."""
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    df["Txn Date"] = pd.to_datetime(df["Txn Date"])

    # Aggregate transactions on the same date
    df_grouped = df.groupby("Txn Date", as_index=False).agg({
        "Description": lambda x: " | ".join(x.dropna().astype(str)),
        "Debit": "sum",
        "Credit": "sum",
        "Balance": "last"
    })
    
    # Fill in missing dates
    full_date_range = pd.date_range(start=df_grouped["Txn Date"].min(), end=df_grouped["Txn Date"].max())
    full_df = pd.DataFrame({"Txn Date": full_date_range})
    merged_df = full_df.merge(df_grouped, on="Txn Date", how="left").fillna({
        "Description": "-",
        "Debit": 0,
        "Credit": 0
    })
    merged_df["Balance"] = merged_df["Balance"].ffill()
    return merged_df

def calculate_metrics(df):
    """Calculate RS, moving averages, and trends."""
    # Slow RS (720 days)
    df['Avg Credit 720D'] = df['Credit'].rolling(window=720, min_periods=720).mean()
    df['Avg Debit 720D'] = df['Debit'].rolling(window=720, min_periods=720).mean()
    df['RS Slow'] = df['Avg Credit 720D'] / df['Avg Debit 720D'].replace(0, np.nan)
    df['RS Normalized Slow'] = ((df['RS Slow'] - df['RS Slow'].min()) / (df['RS Slow'].max() - df['RS Slow'].min())) * 100
    df['Balance MA 720D'] = df['Balance'].rolling(window=720, min_periods=720).mean()
    
    # Fast RS (90 days)
    df['Avg Credit 90D'] = df['Credit'].rolling(window=90, min_periods=90).mean()
    df['Avg Debit 90D'] = df['Debit'].rolling(window=90, min_periods=90).mean()
    df['RS Fast'] = df['Avg Credit 90D'] / df['Avg Debit 90D'].replace(0, np.nan)
    df['RS Normalized Fast'] = ((df['RS Fast'] - df['RS Fast'].min()) / (df['RS Fast'].max() - df['RS Fast'].min())) * 100
    return df

def filter_last_year(df):
    """Filter data for the last year."""
    end_date = df['Txn Date'].max()
    start_date = end_date - pd.DateOffset(years=1)
    return df[(df['Txn Date'] >= start_date) & (df['Txn Date'] <= end_date)], start_date, end_date

def calculate_trend_stats(df, rs_column):
    """Calculate trends and related statistics."""
    total_days = len(df)
    trends = {
        'Up-Trend': df[df[rs_column] > 70],
        'Down-Trend': df[df[rs_column] < 30],
        'Sideways-Trend': df[(df[rs_column] >= 30) & (df[rs_column] <= 70)]
    }
    stats = {
        trend: {
            'Percentage': (len(data) / total_days) * 100,
            'Average RS': data[rs_column].mean()
        } for trend, data in trends.items()
    }
    return stats

def calculate_sideways_avg_spending(df, rs_column):
    """Calculate average daily spending during sideways trend."""
    sideways_trend = df[(df[rs_column] >= 30) & (df[rs_column] <= 70)]
    total_spending = sideways_trend['Debit'].sum()
    days_in_sideways = len(sideways_trend)
    return total_spending / days_in_sideways if days_in_sideways else 0


def normalize_balance(balance, min_balance=6000, max_balance=10000000):
    """Normalize balance between 0 to 100."""
    normalized = ((balance - min_balance) / (max_balance - min_balance)) * 100
    return min(max(normalized, 0), 100)  # Ensure it stays within 0-100

def assign_balance_level(normalized_balance):
    """Assign balance level based on normalized balance."""
    if normalized_balance <= 12.5:
        return 1
    elif normalized_balance <= 25:
        return 2
    elif normalized_balance <= 37.5:
        return 3
    elif normalized_balance <= 50:
        return 4
    elif normalized_balance <= 62.5:
        return 5
    elif normalized_balance <= 75:
        return 6
    elif normalized_balance <= 87.5:
        return 7
    else:
        return 8

def assign_social_class(normalized_balance, branch_df, ifsc_code):
    """Assign social class based on branch-specific thresholds."""
    thresholds = branch_df[branch_df['IFSC_CODE'] == ifsc_code].iloc[0]
    if normalized_balance <= thresholds['P12_5']:
        return 1
    elif normalized_balance <= thresholds['P25']:
        return 2
    elif normalized_balance <= thresholds['P37_5']:
        return 3
    elif normalized_balance <= thresholds['P50']:
        return 4
    elif normalized_balance <= thresholds['P62_5']:
        return 5
    elif normalized_balance <= thresholds['P75']:
        return 6
    elif normalized_balance <= thresholds['P87_5']:
        return 7
    else:
        return 8
    
# ------------------------------
# Plotting Functions
# ------------------------------

# def plot_balance(df, ax, start_date, end_date):
#     ax.clear()
#     ax.plot(df['Txn Date'], df['Balance'], label="Balance", color='blue', alpha=0.6)
#     ax.plot(df['Txn Date'], df['Balance MA 720D'], label="720-Day Moving Average", color='orange', linestyle="dashed")
#     ax.set_title("Balance and Moving Average Over Time")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Balance")
#     ax.legend()
#     ax.set_xlim([start_date, end_date])

def plot_rs(df, ax, start_date, end_date, rs_column, title):
    ax.clear()
    ax.plot(df['Txn Date'], df[rs_column], label=title, color='purple')
    ax.axhline(y=70, color='green', linestyle='dashed')
    ax.axhline(y=30, color='red', linestyle='dashed')
    ax.fill_between(df['Txn Date'], 30, 70, color='lightgray', alpha=0.2)
    ax.fill_between(df['Txn Date'], 30, 70, where=(df[rs_column] > 70), color='lightgreen', alpha=0.3)
    ax.fill_between(df['Txn Date'], 30, 70, where=(df[rs_column] < 30), color='lightpink', alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized RS (0-100)")
    ax.legend()
    ax.set_xlim([start_date, end_date])

# ------------------------------
# Main Function
# ------------------------------

def main():
    cif_no = "CIF100001"
    ifsc_code = "SBIN0000336" 
    #This is the IFSC code where the user is currently located. By default, it is set to the IFSC code of the branch where the user has an account or has highest number of transactions. For this example, I have chosen Bilaspur branch
    file_path = f"data/user_instance/{cif_no}/bank_statement.xlsx"
    output_file = f"data/user_instance/{cif_no}/daily_bank_statement.xlsx"

    # Load data
    df = load_and_preprocess_data(file_path)
    branch_df = pd.read_excel("data/branch.xlsx")
    customer_df = pd.read_excel("data/customer.xlsx")  # Load customer data
    persona_df = pd.read_excel("data/discrete_persona.xlsx")  # Load persona data
    df = calculate_metrics(df)
    last_year_df, start_date, end_date = filter_last_year(df)

    # Fetch occupation
    occupation = customer_df[customer_df['CIF_NO'] == cif_no]['OCCUPATION'].values
    occupation = occupation[0] if len(occupation) else "Unknown"

    # Calculate both slow and fast trends
    slow_trend_stats = calculate_trend_stats(last_year_df, 'RS Normalized Slow')
    fast_trend_stats = calculate_trend_stats(last_year_df, 'RS Normalized Fast')
    avg_sideways_spending_fast = calculate_sideways_avg_spending(last_year_df, 'RS Normalized Fast')

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)

    showing_slow = [True]  # Mutable state to switch between slow and fast RS

    def toggle_rs(event):
        if showing_slow[0]:
            plot_rs(df, ax, start_date, end_date, 'RS Normalized Fast', "Fast RS Over Time (90-Day Window)")
        else:
            plot_rs(df, ax, start_date, end_date, 'RS Normalized Slow', "Slow RS Over Time (720-Day Window)")
        showing_slow[0] = not showing_slow[0]
        plt.draw()

    # Button for toggling RS views
    ax_button = plt.axes([0.7, 0.05, 0.2, 0.075])
    toggle_button = Button(ax_button, 'Toggle RS View')
    toggle_button.on_clicked(toggle_rs)

    # Initial plot
    plot_rs(df, ax, start_date, end_date, 'RS Normalized Slow', "Slow RS Over Time (720-Day Window)")
    plt.show()

    # Calculate overall RS averages
    avg_rs_year = df['RS Slow'].mean()

    # Save processed data
    df.rename(columns={"Txn Date": "Date"}, inplace=True)
    columns_to_store = [
        "Date", "Description", "Debit", "Credit", "Balance",
        "RS Normalized Slow", "Balance MA 720D", "RS Normalized Fast"
    ]
    df[columns_to_store].to_excel(output_file, index=False)

    # Display summary statistics
    avg_balance = last_year_df['Balance'].mean()
    normalized_balance = normalize_balance(avg_balance)
    balance_level = assign_balance_level(normalized_balance)
    social_class = assign_social_class(normalized_balance, branch_df, ifsc_code)

    # Fetch Discrete Persona ID
    matching_persona = persona_df[
        (persona_df['BALANCE_LEVEL'] == balance_level) &
        (persona_df['SOCIAL_CLASS'] == social_class) &
        (persona_df['OCCUPATION'] == occupation)
    ]

    discrete_persona_id = matching_persona['DISCRETE_PERSONA_ID'].values
    discrete_persona_id = discrete_persona_id[0] if len(discrete_persona_id) else "Unknown"

    # Print outputs
    print(f"Occupation: {occupation}")
    print(f"Average Balance for the last year: {avg_balance:.2f}")
    print(f"Normalized Balance (0-100): {normalized_balance:.2f}")
    print(f"Balance Level for the last year: {balance_level}")
    print(f"Social Class: {social_class}")
    print(f"Discrete Persona ID: {discrete_persona_id}")
    print(f"Average RS for the last year: {avg_rs_year:.2f}")
    print(f"Average Daily Spending during Fast Sideways Trend: {avg_sideways_spending_fast:.2f}")

    print("\nSlow Trend Stats:")
    for trend, stats in slow_trend_stats.items():
        print(f"{trend} - {stats['Percentage']:.2f}% of time, Avg RS: {stats['Average RS']:.2f}")

    print("\nFast Trend Stats:")
    for trend, stats in fast_trend_stats.items():
        print(f"{trend} - {stats['Percentage']:.2f}% of time, Avg RS: {stats['Average RS']:.2f}")

    print(f"✅ Daily bank statement saved as {output_file}")

    # Generate persona JSON output
    persona_output = {
        "cif_no": cif_no,
        "discrete_persona_id": discrete_persona_id,
        "slow_uptrend_%": slow_trend_stats.get('Up-Trend', {}).get('Percentage', 0.0),
        "slow_uptrend_rs": slow_trend_stats.get('Up-Trend', {}).get('Average RS', 0.0),
        "slow_downtrend_%": slow_trend_stats.get('Down-Trend', {}).get('Percentage', 0.0),
        "slow_downtrend_rs": slow_trend_stats.get('Down-Trend', {}).get('Average RS', 0.0),
        "fast_uptrend_%": fast_trend_stats.get('Up-Trend', {}).get('Percentage', 0.0),
        "fast_uptrend_rs": fast_trend_stats.get('Up-Trend', {}).get('Average RS', 0.0),
        "fast_downtrend_%": fast_trend_stats.get('Down-Trend', {}).get('Percentage', 0.0),
        "fast_downtrend_rs": fast_trend_stats.get('Down-Trend', {}).get('Average RS', 0.0),
    }

    # Print the persona JSON object
    print("\npersona:{")
    for key, value in persona_output.items():
        if isinstance(value, str):
            print(f'    "{key}": "{value}",')
        elif key == "discrete_persona_id":
            print(f'    "{key}": {value},')
        else:
            print(f'    "{key}": {value:.2f},')
    print("}")


if __name__ == "__main__":
    main()


