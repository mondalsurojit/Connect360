import pandas as pd
from rapidfuzz import fuzz

# File paths
customer_file = "data/cust/customer_data.xlsx"
account_file = "data/cust/account_data.xlsx"
nominee_file = "data/cust/nominee_data.xlsx"
customer_account_file = "data/cust/customer_account.xlsx"
family_file = "family.xlsx"

# Load data from separate Excel files with error handling
try:
    customer_df = pd.read_excel(customer_file)
    account_df = pd.read_excel(account_file)
    nominee_df = pd.read_excel(nominee_file)
    customer_account_df = pd.read_excel(customer_account_file)
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit()

# Strip column names to avoid issues with spaces
for df in [customer_df, account_df, nominee_df, customer_account_df]:
    df.columns = df.columns.str.strip().str.upper()

# Define family relationships
FAMILY_RELATIONSHIPS = [
    "Father", "Mother", "Husband", "Wife", "Son", "Daughter",
    "Brother", "Sister", "Grandfather", "Grandmother", "Uncle", "Aunt",
    "Cousin", "Nephew", "Niece", "In-Law", "Guardian"
]

def is_family_relation(relation):
    """Check if the given relation is familial using fuzzy matching (RapidFuzz)."""
    if pd.isna(relation):
        return False
    return any(fuzz.partial_ratio(str(relation).lower(), fam.lower()) > 85 for fam in FAMILY_RELATIONSHIPS)

# Identify joint accounts
joint_accounts = customer_account_df.groupby("ACCOUNT_NO").filter(lambda x: len(x) > 1)

# Print all joint accounts in detailed format
print("\n🔹 JOINT ACCOUNTS (Detailed Format):")
for account_no, group in joint_accounts.groupby("ACCOUNT_NO"):
    print(f"\n➡️ ACCOUNT NO: {account_no}")
    for _, row in group.iterrows():
        print(f"   - CIF_NO: {row['CIF_NO']} ({row['RELATIONSHIP_WITH_PRIMARY']})")

# Create family groups
family_dict = {}
family_id = 1  

for account_no, group in joint_accounts.groupby("ACCOUNT_NO"):
    primary_cif = account_df.loc[account_df["ACCOUNT_NO"] == account_no, "PRIMARY_CIF_NO"].values
    if len(primary_cif) == 0:
        continue
    primary_cif = primary_cif[0]

    members = group["CIF_NO"].tolist()
    for cif in members:
        if cif == primary_cif:
            continue  # Skip primary holder
        relation = customer_account_df.loc[
            (customer_account_df["CIF_NO"] == cif) & (customer_account_df["ACCOUNT_NO"] == account_no),
            "RELATIONSHIP_WITH_PRIMARY"
        ].values
        if len(relation) > 0 and is_family_relation(relation[0]):  
            if cif not in family_dict:
                family_dict[cif] = family_id

    if any(family_dict.get(cif) == family_id for cif in members):
        if primary_cif not in family_dict:
            family_dict[primary_cif] = family_id
        family_id += 1  # Increment for the next family

# Step 1: Identify families using Nominee Table (Mobile/Address Match)
for _, nominee in nominee_df.iterrows():
    nominee_account = nominee["ACCOUNT_NO"]
    nominee_address = nominee["ADDRESS_ID"]
    nominee_mobile = nominee["MOBILE_NO"]

    # Find accounts with the same ADDRESS_ID or MOBILE_NO
    matching_customers = customer_df[
        ((customer_df["ADDRESS_ID"] == nominee_address) | 
         (customer_df["MOBILE_NO"] == nominee_mobile))
    ]

    for _, matched_customer in matching_customers.iterrows():
        matched_cif = matched_customer["CIF_NO"]
        nominee_cif = account_df.loc[account_df["ACCOUNT_NO"] == nominee_account, "CIF_NO"].values[0]

        if matched_cif != nominee_cif and matched_cif not in family_dict:
            print(f"✅ Found Family (Nominee Match): Nominee {nominee_account} & CIF {matched_cif}")
            family_dict[matched_cif] = family_dict.get(nominee_cif, family_id)
            family_id += 1

# Step 2: Find families within accounts based on common Mobile/Address
account_customer_df = account_df.merge(customer_df, on="CIF_NO", how="left")

for _, account in account_customer_df.iterrows():
    account_no = account["ACCOUNT_NO"]
    primary_cif = account["CIF_NO"]
    address_id = account["ADDRESS_ID"]
    mobile_no = account["MOBILE_NO"]

    # Find other accounts with the same ADDRESS_ID or MOBILE_NO
    similar_accounts = account_customer_df[
        ((account_customer_df["ADDRESS_ID"] == address_id) | 
         (account_customer_df["MOBILE_NO"] == mobile_no)) &
        (account_customer_df["ACCOUNT_NO"] != account_no)  
    ]

    for _, similar_account in similar_accounts.iterrows():
        matched_cif = similar_account["CIF_NO"]
        if matched_cif != primary_cif and matched_cif not in family_dict:
            print(f"✅ Found Family (Account Match): Account {account_no} & CIF {matched_cif}")
            family_dict[primary_cif] = family_id
            family_dict[matched_cif] = family_id
            family_id += 1

# Step 3: Remove duplicates (Ensure One CIF_NO Appears Once)
final_family_dict = {}
for cif, fam_id in family_dict.items():
    if cif not in final_family_dict:
        final_family_dict[cif] = fam_id

# Step 4: Save the updated Family table with FAMILY_ID first
family_df = pd.DataFrame(list(final_family_dict.items()), columns=["CIF_NO", "FAMILY_ID"])
family_df = family_df[["FAMILY_ID", "CIF_NO"]]  # Reorder columns

if not family_df.empty:
    family_df.to_excel(family_file, index=False)
    print(f"\n✅ Updated Family table created successfully and saved as {family_file}.")
else:
    print("\n⚠️ No new family relationships found, no file created.")
