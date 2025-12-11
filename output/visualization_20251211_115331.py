import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Korean font
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['font.serif'] = 'Malgun Gothic'

# Read CSV file
csv_file_path = "input/sample_querydata/production_orders.csv"
df = pd.read_csv(csv_file_path)

# Create output directory if not exists
os.makedirs('output', exist_ok=True)

# Visualization 1: Histogram of actual_quantity and defect_quantity
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(data=df, x='actual_quantity', kde=False, ax=axs[0])
axs[0].set_title('Histogram of Actual Quantity')
sns.histplot(data=df, x='defect_quantity', kde=False, ax=axs[1])
axs[1].set_title('Histogram of Defect Quantity')
plt.tight_layout()
plt.savefig('output/production_orders_histogram.png')
plt.show()

# Visualization 2: Boxplot of target_quantity by production_date and line_id
sns.set(style="whitegrid")
pivot_table = df.pivot_table(index='production_date', columns='line_id', values='target_quantity')
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(x='production_date', y='target_quantity', hue='line_id', data=pivot_table)
ax.set_title('Boxplot of Target Quantity by Production Date and Line ID')
plt.savefig('output/production_orders_boxplot.png')
plt.show()