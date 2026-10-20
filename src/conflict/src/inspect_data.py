import pandas as pd


FILE = "data/raw/xapp_network_data.csv"


def main():

    df = pd.read_csv(FILE)

    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    # Basic information
    print("\n1. Dataset shape")
    print(df.shape)

    # Column names
    print("\n2. Columns")
    for column in df.columns:
        print(f"- {column}")

    # Data types
    print("\n3. Data types")
    print(df.dtypes)

    # Missing values
    print("\n4. Missing values")
    print(df.isnull().sum())

    # Conflict distribution
    print("\n5. Conflict distribution")
    print(df["conflict"].value_counts())

    print("\nConflict percentages:")
    print(
        df["conflict"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    # Direct conflict distribution
    print("\n6. Direct conflict distribution")
    print(df["direct_conflict"].value_counts())

    # Numerical statistics
    print("\n7. Numerical statistics")
    print(df.describe())

    # Sample records
    print("\n8. Sample records")
    print(
        df[
            [
                "xapp_a",
                "xapp_b",
                "cell_id",
                "resource_type",
                "delta_a",
                "delta_b",
                "throughput",
                "latency",
                "packet_loss",
                "prb_utilization",
                "conflict"
            ]
        ].head(10).to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("DATASET INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
