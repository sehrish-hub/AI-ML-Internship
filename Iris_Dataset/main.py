# 🌸 Task 1: Exploring and Visualizing Iris Dataset

## 📌 Objective
# The goal of this task is to explore, analyze, and visualize the Iris dataset to understand feature distributions, relationships, and patterns.
# A simple machine learning model is also applied to evaluate data suitability.


# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Iris dataset
df = sns.load_dataset("iris")

# -----------------------------
# Data Inspection
# -----------------------------
print("Shape:", df.shape)
print("Columns:", df.columns)
print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Info:\n")
df.info()

print("\nSummary Statistics:\n")
print(df.describe())

# -----------------------------
# Scatter Plot
# -----------------------------
plt.figure(figsize=(6, 4))
sns.scatterplot(
    x="sepal_length",
    y="sepal_width",
    hue="species",
    data=df
)
plt.title("Sepal Length vs Sepal Width")
plt.savefig("outputs/scatter_plot.png")
plt.show()

### 📌 Scatter Plot Insight
# The scatter plot shows that Setosa species is clearly separated from Versicolor and Virginica. The other two species overlap, indicating similar sepal characteristics.


# -----------------------------
# Histograms
# -----------------------------
df.hist(figsize=(8, 6))
plt.suptitle("Feature Distributions")
plt.savefig("outputs/histograms.png")
plt.show()

### 📌 Histogram Insight
# Most features show smooth distributions. Petal length and petal width vary significantly across species, making them strong features for classification.


# -----------------------------
# Box Plot
# -----------------------------
plt.figure(figsize=(8, 4))
sns.boxplot(data=df.drop(columns="species"))
plt.title("Box Plot for Outlier Detection")
plt.savefig("outputs/box_plot.png")
plt.show()

### 📌 Box Plot Insight
# Minor outliers are present, especially in sepal width, but no extreme anomalies exist. The dataset is clean and usable.











# # 🌸 WSL-Friendly Task 1: Exploring and Visualizing Iris Dataset

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # -----------------------------
# # Create outputs folder if not exists
# # -----------------------------
# os.makedirs("outputs", exist_ok=True)

# # -----------------------------
# # Load Iris dataset
# # -----------------------------
# df = sns.load_dataset("iris")

# # -----------------------------
# # Data Inspection
# # -----------------------------
# print("Shape:", df.shape)
# print("Columns:", df.columns)
# print("\nFirst 5 Rows:\n")
# print(df.head())

# print("\nDataset Info:\n")
# df.info()

# print("\nSummary Statistics:\n")
# print(df.describe())

# # -----------------------------
# # Scatter Plot
# # -----------------------------
# plt.figure(figsize=(6, 4))
# sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=df)
# plt.title("Sepal Length vs Sepal Width")
# plt.savefig("outputs/scatter_plot.png")
# plt.close()  # Close figure to avoid GUI issues

# # -----------------------------
# # Histograms
# # -----------------------------
# df.hist(figsize=(8, 6))
# plt.suptitle("Feature Distributions")
# plt.savefig("outputs/histograms.png")
# plt.close()

# # -----------------------------
# # Box Plot
# # -----------------------------
# plt.figure(figsize=(8, 4))
# sns.boxplot(data=df.drop(columns="species"))
# plt.title("Box Plot for Outlier Detection")
# plt.savefig("outputs/box_plot.png")
# plt.close()

# print("\n✅ Plots saved in 'outputs/' folder. Open them to view visualizations.")

