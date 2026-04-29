import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_prep import load_and_clean_data

def generate_correlation_heatmap(df, features, target):
    # Combine features and target for the matrix
    cols_to_analyze = features + [target]
    corr_matrix = df[cols_to_analyze].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix: CO2 Emissions and Energy Features")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    clean_df, feats, targ = load_and_clean_data('../data/co2_emissions.csv')
    generate_correlation_heatmap(clean_df, feats, targ)