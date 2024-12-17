import pandas as pd

# Load the CSV file
df = pd.read_csv('./dataframe/1800ChessBrat.csv')

# Specify the formats
formats = ['Bullet', 'Blitz', 'Rapid', 'Classical']

# Function to calculate TER for each format
def calculate_ter_for_format(format_name):
    subset = df[df['Format'] == format_name]
    if not subset.empty:
        total_ai_think_time = subset['Predicted Think Time (Highest)'].sum()
        total_actual_think_time = subset['Think Time'].sum()
        return total_ai_think_time / total_actual_think_time
    return None

# Calculate TER for each specified format and overall
ter_data = {'Format': [], 'TER': []}
for fmt in formats:
    ter_value = calculate_ter_for_format(fmt)
    if ter_value is not None:
        ter_data['Format'].append(fmt)
        ter_data['TER'].append(f"{ter_value:.2f}")

# Calculate overall TER
total_ai_think_time = df['Predicted Think Time (Highest)'].sum()
total_actual_think_time = df['Think Time'].sum()
overall_ter = total_ai_think_time / total_actual_think_time
ter_data['Format'].append('Overall')
ter_data['TER'].append(f"{overall_ter:.2f}")

# Create a DataFrame for the results
ter_df = pd.DataFrame(ter_data)

# Save the results to a new CSV file
output_csv = 'TER1800.csv'
ter_df.to_csv(output_csv, index=False)

print(f'Time Efficiency Ratio CSV generated: {output_csv}')
