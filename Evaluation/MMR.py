import csv
import os

# Function to read the CSV file and extract moves, FENs, and predictions
def read_data_from_csv(file_path):
    data = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return data

# Function to calculate MMR based on matching moves
def calculate_mmr(data):
    matching_moves = 0
    total_moves = len(data)

    for row in data:
        move = row['Move']
        prediction = row['Predictions']
        
        if move == prediction:
            matching_moves += 1

    return (matching_moves / total_moves) * 100

# Function to calculate MMR for each format and overall
def calculate_mmr_by_format(data):
    formats = ["Bullet", "Blitz", "Rapid", "Classical"]
    mmr_results = {}

    for format in formats:
        format_data = [row for row in data if row['Format'] == format]
        if format_data:
            mmr_results[format] = calculate_mmr(format_data)
        else:
            mmr_results[format] = 0

    mmr_results['Overall'] = calculate_mmr(data)
    return mmr_results

# Function to write MMR results to CSV file
def write_mmr_to_csv(file_name, mmr_results):
    output_path = f"./mmr/{file_name}_MMR_Results.csv"
    
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Format", "MMR"])
        
        for format, mmr in mmr_results.items():
            writer.writerow([format, f"{mmr:.2f}"])
    
    print(f"CSV report generated: {output_path}")

# List of input CSV files
files = [
    './dataframe/1800Leela.csv',
    './dataframe/1800Maia.csv',
    './dataframe/1800Stockfish.csv',
    './dataframe/1800ChessBrat.csv',
    './dataframe/1700Leela.csv',
    './dataframe/1700Maia.csv',
    './dataframe/1700Stockfish.csv',
    './dataframe/1700ChessBrat.csv',
    './dataframe/1600Leela.csv',
    './dataframe/1600Maia.csv',
    './dataframe/1600Stockfish.csv',
    './dataframe/1600ChessBrat.csv',
    './dataframe/1500Leela.csv',
    './dataframe/1500Maia.csv',
    './dataframe/1500Stockfish.csv',
    './dataframe/1500ChessBrat.csv'
]

# Loop through each file, calculate MMR, and generate CSV report
for file_path in files:
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Starting the Move Matching Rate (MMR) calculation for {file_name}...")

    # Read data from CSV
    data = read_data_from_csv(file_path)

    # Calculate MMR for each format and overall
    mmr_results = calculate_mmr_by_format(data)

    # Generate CSV report
    write_mmr_to_csv(file_name, mmr_results)

print("MMR calculation and CSV report generation completed for all files.")
