import pandas as pd
import chess
import chess.engine
from tqdm import tqdm

# Configuration
engine_path = "./stockfish/stockfishexe"  # Update this path to your Stockfish executable
depth = 18  # Set the depth for the Stockfish analysis

# Path to your combined CSV file
csv_file_path = './test.csv'

# Load the dataset into a DataFrame
df = pd.read_csv(csv_file_path)

# Initialize the chess engine
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

def lan_to_uci(board, lan_move):
    try:
        move = board.parse_san(lan_move)
        return move.uci()
    except Exception as e:
        print(f"Error converting LAN to UCI for move {lan_move}: {e}")
        return None

def evaluate_move(engine, fen, move, depth):
    board = chess.Board(fen)
    optimal_move_info = engine.analyse(board, chess.engine.Limit(depth=depth))  # Evaluate the best move in the position
    optimal_eval = optimal_move_info['score'].relative.score()

    if move:
        move_obj = chess.Move.from_uci(move)
        if move_obj in board.legal_moves:
            board.push(move_obj)
            move_eval_info = engine.analyse(board, chess.engine.Limit(depth=depth))
            move_eval = move_eval_info['score'].relative.score()
            board.pop()
        else:
            move_eval = None
    else:
        move_eval = None

    return optimal_eval, move_eval

# Create a list to store the evaluation results
evaluations = []

# Analyze each row in the DataFrame with progress bar
for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating moves", unit="move"):
    fen = row['FEN Before the move']
    move = row['Move']
    pchessbrat = row['PChessBrat']
    pmaia = row['PMaia']
    pstockfish = row['PStockfish']
    pleela = row['PLeela']

    board = chess.Board(fen)

    move_uci = lan_to_uci(board, move)
    pchessbrat_uci = lan_to_uci(board, pchessbrat)
    pmaia_uci = lan_to_uci(board, pmaia)
    pstockfish_uci = lan_to_uci(board, pstockfish)
    pleela_uci = lan_to_uci(board, pleela)

    try:
        optimal_eval, evmove = evaluate_move(engine, fen, move_uci, depth)
        _, evchessbrat = evaluate_move(engine, fen, pchessbrat_uci, depth)
        _, evmaia = evaluate_move(engine, fen, pmaia_uci, depth)
        _, evstockfish = evaluate_move(engine, fen, pstockfish_uci, depth)
        _, evleela = evaluate_move(engine, fen, pleela_uci, depth)
    except Exception as e:
        print(f"Error processing row with FEN {fen} and move {move}: {e}")
        continue

    evaluations.append({
        'Optimal': optimal_eval,
        'EvMove': evmove,
        'EvChessBrat': evchessbrat,
        'EvMaia': evmaia,
        'EvStockfish': evstockfish,
        'EvLeela': evleela
    })

# Convert the results to a DataFrame
evaluations_df = pd.DataFrame(evaluations)

# Save the results to a new CSV file
output_path = "./evaluation_results.csv"
evaluations_df.to_csv(output_path, index=False)
print(f"Evaluation results saved to {output_path}")

# Close the engine
engine.quit()
