import pandas as pd

# Read the CSV file
csv_path = 'csvs/summary.csv'
df = pd.read_csv(csv_path)

# Filter for rows where filter is "0 sigma"
filtered_df = df[df['filter'] == '0 sigma'].copy()

print(f"Found {len(filtered_df)} rows with filter='0 sigma'")

# Define blacklist of models to exclude
blacklist = ['openai-community/openai-gpt']

# Filter for rows where types contains ALL FOUR weight components
# Check if each type string contains all four components: bias, norm, other, embedding
filtered_df['has_all_four'] = filtered_df['types'].str.contains('bias', regex=False) & \
                               filtered_df['types'].str.contains('norm', regex=False) & \
                               filtered_df['types'].str.contains('other', regex=False) & \
                               filtered_df['types'].str.contains('embedding', regex=False)

final_df = filtered_df[filtered_df['has_all_four']].copy()

# Remove blacklisted models
final_df = final_df[~final_df['model'].isin(blacklist)].copy()

print(f"Found {len(final_df)} rows with all four types combined in a single entry")
print(f"\nUnique weight type combinations with all four:")
print(final_df['types'].unique().tolist())
print(f"\nBreakdown by model:")
print(final_df.groupby('model').size().sort_values(ascending=False))

# Create markdown file
md_path = 'output_filter_zero.md'

with open(md_path, 'w') as f:
    f.write("# Rows where Filter = 0 sigma (Rows with ALL FOUR Weight Types Combined)\n\n")
    f.write(f"Total rows: {len(final_df)}\n\n")
    f.write(f"Weight type combinations containing all four (bias, norm, other, embedding):\n")
    for wtype in sorted(final_df['types'].unique()):
        count = len(final_df[final_df['types'] == wtype])
        f.write(f"- {wtype}: {count} rows\n")
    f.write("\n")
    
    f.write("## Data Table\n\n")
    
    # Create markdown table manually
    columns = final_df.columns.tolist()
    f.write("| " + " | ".join(columns) + " |\n")
    f.write("|" + "|".join(["-" * max(10, len(col)) for col in columns]) + "|\n")
    
    for _, row in final_df.iterrows():
        values = [str(row[col]) for col in columns]
        f.write("| " + " | ".join(values) + " |\n")
    
    # Add total count (sum of count column) at the end
    total_count_sum = int(final_df['count'].sum())
    f.write("\n## Summary\n\n")
    f.write(f"**Total Count: {total_count_sum}**\n")

print(f"\nMarkdown file created: {md_path}")
