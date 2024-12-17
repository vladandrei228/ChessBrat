import chess
import chess.engine
import pandas as pd
import os
import time
from stockfish import Stockfish
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
engine_path = "./stockfish/stockfishexe"  # Update this path to your Stockfish executable
num_workers = 4  # Adjust the number of parallel workers based on your CPU cores
depth = 18  # Set the depth for the Stockfish analysis
blunder_threshold = 200  # Centipawn loss threshold to consider a move a blunder

# Path to your CSV file
csv_file_path = './testctp.csv'

# Load the dataset into a DataFrame
df = pd.read_csv(csv_file_path)

def initialize_engine():
    return Stockfish(engine_path)

def get_centipawn_loss(engine_path, fen, move):
    stockfish = initialize_engine()
    stockfish.depth = depth  # Set the depth for the engine analysis
    board = chess.Board(fen)
    stockfish.set_fen_position(fen)
    evaluation_before = stockfish.get_evaluation()

    board.push_san(move)
    stockfish.set_fen_position(board.fen())
    evaluation_after = stockfish.get_evaluation()

    if evaluation_before['type'] == 'cp' and evaluation_after['type'] == 'cp':
        evaluation_diff = abs(evaluation_after['value'] - evaluation_before['value'])
        return evaluation_diff
    return 0  # In case of non centipawn evaluations, return 0

# Start the timer
start_time = time.time()

# Analyze moves to get centipawn loss and count blunders with multithreading and progress bar
centipawn_losses = []

def analyze_row(row):
    fen = row['FEN Before the move']
    move = row['Move']
    centipawn_loss = get_centipawn_loss(engine_path, fen, move)
    is_blunder = centipawn_loss >= blunder_threshold
    return {
        'Format': row['Format'],
        'FEN': fen,
        'Move': move,
        'Centipawn Loss': centipawn_loss,
        'Is Blunder': is_blunder
    }

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(analyze_row, row) for _, row in df.iterrows()]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing moves", unit="move"):
        centipawn_losses.append(future.result())

# Convert the results to a DataFrame
centipawn_losses_df = pd.DataFrame(centipawn_losses)

# Calculate the average centipawn loss and blunder rate for each format and overall
formats = ["Bullet", "Blitz", "Rapid", "Classical"]
results = []

for format in formats:
    format_data = centipawn_losses_df[centipawn_losses_df['Format'] == format]
    if not format_data.empty:
        avg_loss = format_data['Centipawn Loss'].mean()
        num_blunders = format_data['Is Blunder'].sum()
        blunder_rate = (num_blunders / len(format_data)) * 100
    else:
        avg_loss = 0
        blunder_rate = 0
    results.append({
        'Format': format,
        'Average Centipawn Loss': f"{avg_loss:.2f}",
        'Blunder Rate': f"{blunder_rate:.2f}%"
    })

# Calculate overall average centipawn loss and blunder rate
overall_avg_loss = centipawn_losses_df['Centipawn Loss'].mean()
overall_num_blunders = centipawn_losses_df['Is Blunder'].sum()
overall_blunder_rate = (overall_num_blunders / len(centipawn_losses_df)) * 100
results.append({
    'Format': 'Overall',
    'Average Centipawn Loss': f"{overall_avg_loss:.2f}",
    'Blunder Rate': f"{overall_blunder_rate:.2f}%"
})

# Convert the results to a DataFrame and save to a file
results_df = pd.DataFrame(results)
output_path = "actual_blunder_rates.csv"
results_df.to_csv(output_path, index=False)
print(f"Blunder rates saved to {output_path}")

# Stop the timer and calculate elapsed time
elapsed_time = time.time() - start_time

# Display the average centipawn loss, blunder rate, and elapsed time for the entire dataset
print(f"The overall average centipawn loss for the given dataset is {overall_avg_loss:.2f}")
print(f"The overall blunder rate for the given dataset is {overall_blunder_rate:.2f}%")
print(f"Time taken: {elapsed_time:.2f} seconds")
