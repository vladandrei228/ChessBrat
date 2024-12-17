import re
import os

def read_file_in_chunks(file_path):
    with open(file_path, 'r') as file:
        buffer = ""
        for line in file:
            if line.startswith("[Event"):
                if buffer:
                    yield buffer
                buffer = line
            else:
                buffer += line
        if buffer:
            yield buffer

def categorize_and_save_games(file_path, batch_size=10000):
    categorized_games = {
        '1400': [],
        '1500': [],
        '1600': [],
        '1700': [],
        '1800': []
    }

    total_games = 0

    for game in read_file_in_chunks(file_path):
        total_games += 1
        white_elo = re.search(r'\[WhiteElo \"(\d+)\"\]', game)
        black_elo = re.search(r'\[BlackElo \"(\d+)\"\]', game)

        if white_elo:
            white_elo = int(white_elo.group(1))
        if black_elo:
            black_elo = int(black_elo.group(1))

        if white_elo or black_elo:
            if (white_elo and 1350 <= white_elo <= 1450) or (black_elo and 1350 <= black_elo <= 1450):
                categorized_games['1400'].append(game)
            if (white_elo and 1450 <= white_elo <= 1550) or (black_elo and 1450 <= black_elo <= 1550):
                categorized_games['1500'].append(game)
            if (white_elo and 1550 <= white_elo <= 1650) or (black_elo and 1550 <= black_elo <= 1650):
                categorized_games['1600'].append(game)
            if (white_elo and 1650 <= white_elo <= 1750) or (black_elo and 1650 <= black_elo <= 1750):
                categorized_games['1700'].append(game)
            if (white_elo and 1750 <= white_elo <= 1850) or (black_elo and 1750 <= black_elo <= 1850):
                categorized_games['1800'].append(game)

        if total_games % batch_size == 0:
            save_games(categorized_games)
            categorized_games = {key: [] for key in categorized_games}

    if any(categorized_games.values()):
        save_games(categorized_games)

    print(f"Total games processed: {total_games}")

def save_games(categorized_games):
    os.makedirs('Data', exist_ok=True)
    for elo_range, games in categorized_games.items():
        file_name = f'DataFrames/chess_games_{elo_range}.pgn'
        with open(file_name, 'a') as file:
            for game in games:
                file.write(game + '\n\n')

# Example usage
file_path = 'lichess_db_standard_rated_2024-08.pgn'
categorize_and_save_games(file_path)