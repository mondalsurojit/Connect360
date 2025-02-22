import pandas as pd
from rapidfuzz import fuzz
from collections import defaultdict

# File paths
customer_file = "data/customer.xlsx"
account_file = "data/account.xlsx"
nominee_file = "data/nominee.xlsx"
customer_account_file = "data/relations/customer_account.xlsx"
account_nominee_file = "data/relations/account_nominee.xlsx"
family_file = "data/family.xlsx"

# Load data from separate Excel files with error handling
try:
    customer_df = pd.read_excel(customer_file)
    account_df = pd.read_excel(account_file)
    nominee_df = pd.read_excel(nominee_file)
    customer_account_df = pd.read_excel(customer_account_file)
    account_nominee_df = pd.read_excel(account_nominee_file)
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit()

# Standardize column names (strip spaces & convert to uppercase)
for df in [customer_df, account_df, nominee_df, customer_account_df]:
    df.columns = df.columns.str.strip().str.upper()

# Define family relationships
FAMILY_RELATIONSHIPS = [
    "Father", "Mother", "Husband", "Wife", "Son", "Daughter",
    "Brother", "Sister", "Grandfather", "Grandmother", "Uncle", "Aunt",
    "Cousin", "Nephew", "Niece", "In-Law", "Guardian"
]

def is_family_relation(relation):
    """Check if the given relation is familial using fuzzy matching."""
    if pd.isna(relation):
        return False
    return any(fuzz.partial_ratio(str(relation).lower(), fam.lower()) > 85 for fam in FAMILY_RELATIONSHIPS)

# Step 1: Identify families using joint accounts based on familial relationships

# Identify joint accounts based on multiple customers linked to the same ACCOUNT_NO
joint_accounts = customer_account_df.groupby("ACCOUNT_NO").filter(lambda x: len(x) > 1)

# Print all joint accounts in a detailed format
print("\n🔹 JOINT ACCOUNTS (Detailed Format):")
for account_no, group in joint_accounts.groupby("ACCOUNT_NO"):
    print(f"\n➡️ ACCOUNT NO: {account_no}")
    for _, row in group.iterrows():
        print(f"   - {row['CIF_NO']} ({row['RELATIONSHIP_WITH_PRIMARY']})")

# Create family groups
family_dict = {}
family_id = 1  # Initialize family ID counter

for account_no, group in joint_accounts.groupby("ACCOUNT_NO"):
    # Identify the primary customer based on the relationship marked as 'Primary'
    primary_cif_values = group.loc[group["RELATIONSHIP_WITH_PRIMARY"].str.lower() == "primary", "CIF_NO"].values
    if len(primary_cif_values) == 0:
        continue  # Skip if primary CIF isn't found
    primary_cif = primary_cif_values[0]

    # List of all members linked to the account
    members = group["CIF_NO"].tolist()
    
    # Temporarily store family members for validation
    current_family_members = set()
    
    for cif in members:
        relation = group.loc[group["CIF_NO"] == cif, "RELATIONSHIP_WITH_PRIMARY"].values
        if len(relation) > 0 and is_family_relation(relation[0]):  # If relationship is familial
            current_family_members.add(cif)
    
    # Include the primary holder in the family if at least 1 other familial relation exists
    if len(current_family_members) >= 1:
        current_family_members.add(primary_cif)
        for cif in current_family_members:
            family_dict[cif] = family_id  # Assign family ID
        family_id += 1  # Increment for the next family


# Step 2: Identify families using Account-Nominee Table (Mobile/Address Match)
# Clean mobile numbers by removing '+91' prefix
nominee_df["MOBILE_NO"] = nominee_df["MOBILE_NO"].astype(str).str.replace(r'^\+91', '', regex=True)
customer_df["MOBILE_NO"] = customer_df["MOBILE_NO"].astype(str).str.replace(r'^\+91', '', regex=True)

# Merge nominee_df with account_nominee_df to get ACCOUNT_NO for each nominee
nominee_account_df = nominee_df.merge(account_nominee_df, on="NOMINEE_ID", how="left")

# Iterate over the merged DataFrame to identify family connections
for _, nominee in nominee_account_df.iterrows():
    nominee_account = nominee["ACCOUNT_NO"]
    nominee_address = nominee["ADDRESS_ID"]
    nominee_mobile = nominee["MOBILE_NO"]

    # Fetch primary customer from the customer_account_df
    primary_cif_values = customer_account_df.loc[
        (customer_account_df["ACCOUNT_NO"] == nominee_account) &
        (customer_account_df["RELATIONSHIP_WITH_PRIMARY"].str.lower() == "primary"),
        "CIF_NO"
    ].values

    if len(primary_cif_values) == 0:
        continue  # Skip if ACCOUNT_NO isn't found

    primary_cif = primary_cif_values[0]

    # Find customers with matching ADDRESS_ID or MOBILE_NO (excluding account holder)
    matching_customers = customer_df[
        ((customer_df["ADDRESS_ID"] == nominee_address) |
         (customer_df["MOBILE_NO"] == nominee_mobile)) &
        (customer_df["CIF_NO"] != primary_cif)
    ]

    for _, matched_customer in matching_customers.iterrows():
        matched_cif = matched_customer["CIF_NO"]
        if matched_cif not in family_dict:
            print(f"✅ Found Family (Nominee Match): Account {nominee_account} - Nominee CIF {primary_cif} & Matched CIF {matched_cif}")
            family_dict[matched_cif] = family_dict.get(primary_cif, family_id)
            family_id += 1

# Step 3: Find families within accounts based on common Mobile/Address
account_df.rename(columns=lambda x: x.strip().upper(), inplace=True)
customer_df.rename(columns=lambda x: x.strip().upper(), inplace=True)

# Merge accounts with customers to associate address and mobile numbers
account_customer_df = customer_account_df.merge(customer_df, on="CIF_NO", how="left")

for _, account in account_customer_df.iterrows():
    account_no = account["ACCOUNT_NO"]
    cif_no = account["CIF_NO"]
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
        if matched_cif != cif_no and matched_cif not in family_dict:
            print(f"✅ Found Family (Account Match): Account {account_no} & CIF {matched_cif}")
            family_dict[cif_no] = family_id
            family_dict[matched_cif] = family_id
            family_id += 1

# Step 4: Remove duplicates (Ensure One CIF_NO Appears Once) and clean family data
# Group CIFs by family ID
family_groups = defaultdict(list)
for cif, fam_id in family_dict.items():
    family_groups[fam_id].append(cif)

# Filter out families with only one member and reassign family IDs in ascending order
final_family_dict = {}
new_family_id = 1  # Start family IDs from 1

for fam_id, members in sorted(family_groups.items()):
    if len(members) > 1:  # Only keep families with at least two members
        for cif in members:
            final_family_dict[cif] = new_family_id
        new_family_id += 1  # Increment family ID for the next valid family


# Step 5: Save the updated Family table with FAMILY_ID first
family_df = pd.DataFrame(list(final_family_dict.items()), columns=["CIF_NO", "FAMILY_ID"])
family_df = family_df[["FAMILY_ID", "CIF_NO"]]  # Reorder columns

if not family_df.empty:
    family_df.to_excel(family_file, index=False)
    print(f"\n✅ Updated Family table created successfully and saved as {family_file}.")
else:
    print("\n⚠️ No new family relationships found, no file created.")
