def forward_fill(df, price_col):
    print(
        "Warning: Forward filling all null values. "
        "Check analysis on missing values.")
    df[price_col] = df[price_col].ffill()
    return df
