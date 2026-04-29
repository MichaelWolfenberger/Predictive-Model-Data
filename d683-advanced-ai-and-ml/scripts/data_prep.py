import pandas as pd

def load_and_clean_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)

    # 1. Apply the 1950 Filter
    df = df[df['year'] >= 1950]

    # 2. Remove Non-Country Aggregates (Keep only rows with valid iso_codes)
    df = df[df['iso_code'].notna()]

    # 3. Select your key features from Task 1
    features = ['gdp', 'population', 'primary_energy_consumption',
                'energy_per_gdp']
    target = 'co2'

    # Keep only the columns we need, plus the target
    df = df[['country', 'year', target] + features]

    # 4. Handle Missing Values
    # Drop rows where the target (CO2) is missing
    df = df.dropna(subset=[target])

    # Fill missing feature values with the median of that specific column
    for col in features:
        df[col] = df[col].fillna(df[col].median())

    print(f"Data cleaned. Remaining rows: {len(df)}")
    return df, features, target

if __name__ == "__main__":
    # Test the function
    # Make sure your file path points to the data folder
    clean_df, feats, targ = load_and_clean_data('../data/co2_emissions.csv')
    print(clean_df.head())