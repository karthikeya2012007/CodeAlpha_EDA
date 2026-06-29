# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("students.csv")

# -------------------------------
# 1. Basic Information
# -------------------------------
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
print(df.info())

# -------------------------------
# 2. Missing Values
# -------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------
# 3. Duplicate Records
# -------------------------------
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# -------------------------------
# 4. Statistical Summary
# -------------------------------
print("\nStatistical Summary:")
print(df.describe())

# -------------------------------
# 5. Correlation
# -------------------------------
print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# -------------------------------
# 6. Histograms
# -------------------------------
df.hist(figsize=(10,8))
plt.show()

# -------------------------------
# 7. Boxplots (Outlier Detection)
# -------------------------------
numeric_columns = df.select_dtypes(include=['int64','float64']).columns

for col in numeric_columns:
    plt.figure(figsize=(5,4))
    plt.boxplot(df[col].dropna())
    plt.title(col)
    plt.show()

# -------------------------------
# 8. Scatter Plot
# -------------------------------
if len(numeric_columns) >= 2:
    plt.figure(figsize=(6,5))
    plt.scatter(df[numeric_columns[0]], df[numeric_columns[1]])
    plt.xlabel(numeric_columns[0])
    plt.ylabel(numeric_columns[1])
    plt.title("Scatter Plot")
    plt.show()

# -------------------------------
# 9. Value Counts (Categorical)
# -------------------------------
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    print(f"\nValue Counts for {col}:")
    print(df[col].value_counts())

# -------------------------------
# 10. Outlier Detection (IQR)
# -------------------------------
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    outliers = df[(df[col] < Q1 - 1.5 * IQR) |
                  (df[col] > Q3 + 1.5 * IQR)]

    print(f"{col}: {len(outliers)} Outliers")

print("\nEDA Completed Successfully!")