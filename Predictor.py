import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Define the CNN model architecture
def create_model(output_units):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(output_units, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create models for piece, alpha (file), and number (rank)
piece_model = create_model(output_units=6)  # 6 types of pieces
alpha_model = create_model(output_units=8)  # 8 columns (a-h)
number_model = create_model(output_units=8)  # 8 rows (1-8)

# Example training data (you need to replace this with your actual dataset)
X_train = np.random.rand(1000, 8, 8, 1)  # Example data
y_train_piece = np.random.randint(0, 6, 1000)
y_train_alpha = np.random.randint(0, 8, 1000)
y_train_number = np.random.randint(0, 8, 1000)

# Train the models
piece_model.fit(X_train, y_train_piece, epochs=10)
alpha_model.fit(X_train, y_train_alpha, epochs=10)
number_model.fit(X_train, y_train_number, epochs=10)

# Function to convert FEN to array
def fen_to_array(fen):
    import chess
    board = chess.Board(fen)
    board_array = np.zeros((8, 8), dtype=int)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        board_array[row, col] = piece.piece_type
    return board_array

# Function to predict the best move in Algebraic notation
def predict_best_move(fen):
    piece_symbols = ['', 'N', 'B', 'R', 'Q', 'K']  # Pawn is empty because it's not represented in Algebraic notation
    board_array = fen_to_array(fen)
    board_array = board_array / 6.0  # Example normalization
    board_array = board_array.reshape((1, 8, 8, 1))  # Example reshape for CNN input

    piece_prediction = piece_model.predict(board_array)
    alpha_prediction = alpha_model.predict(board_array)
    number_prediction = number_model.predict(board_array)

    piece = piece_symbols[np.argmax(piece_prediction)]
    alpha = chr(np.argmax(alpha_prediction) + ord('a'))
    number = np.argmax(number_prediction) + 1

    return f"{piece}{alpha}{number}"

# Example usage
fen_board = 'r1bqkbnr/ppp2ppp/2n5/3Bp3/4P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 0 1'
best_move = predict_best_move(fen_board)
print("Best next move:", best_move)
