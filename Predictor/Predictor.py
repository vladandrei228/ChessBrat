import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from MoveConverter import decode_move, encode_move
from ChannelConverter import convert_fen_to_channels, create_additional_channels
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Layer
import random
import chess

# Define the custom LegalMoveMask layer
class LegalMoveMask(Layer):
    def __init__(self, **kwargs):
        super(LegalMoveMask, self).__init__(**kwargs)

    def call(self, inputs):
        predictions, legal_moves_mask = inputs
        return predictions * legal_moves_mask
    
class DropConnect(Layer):
    def __init__(self, rate, **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            keep_prob = 1.0 - self.rate
            random_tensor = keep_prob
            random_tensor += tf.random.uniform(tf.shape(inputs), dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = tf.divide(inputs, keep_prob) * binary_tensor
            return output
        return inputs

# Function to generate the legal moves mask
def generate_legal_moves_mask(fen, nb_classes=4273):
    board = chess.Board(fen)
    legal_moves_mask = np.zeros(nb_classes, dtype=np.float32)
    for move in board.legal_moves:
        encoded_move = encode_move(move.uci())
        legal_moves_mask[encoded_move] = 1
    return legal_moves_mask

# Register the custom components
get_custom_objects().update({
    'LegalMoveMask': LegalMoveMask,
    'DropConnect': DropConnect
})

# Try loading the model with a try-except block
try:
    model = load_model('chessbrat_1800_model.keras', custom_objects={'LegalMoveMask': LegalMoveMask, 'DropConnect' : DropConnect})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error during model loading: {e}")

# Load preprocessor and label encoders
preprocessor = joblib.load('preprocessor.joblib')
label_encoders = joblib.load('label_encoders.joblib')

# Define input values for prediction
input_values = {
    'Format': 'Blitz',
    'Time Control': '180+2',
    'White Elo': 1824,
    'Black Elo': 1768,
    'White Clock Time': 117,
    'Black Clock Time': 133,
    'Phase': 'Middlegame',
    'FEN Before the move': "r2qkb1r/ppp2pp1/3p1n1p/4n1N1/4PPb1/1BN5/PPQ3PP/R1B1K2R b KQkq - 2 10"
}

# Convert input data into DataFrame
input_df = pd.DataFrame([input_values])

# Apply label encoding to categorical columns
for col in ['Format', 'Time Control', 'Phase']:
    input_df[col] = label_encoders[col].transform(input_df[col])

# Apply transformations using preprocessor
transformed_input = preprocessor.transform(input_df)

# List columns that were transformed in order
all_columns = ['White Elo', 'Black Elo', 'White Clock Time', 'Black Clock Time', 'Format', 'Time Control', 'Phase']

# Map the transformations accordingly
column_transformation_map = {col: transformed_input[0, idx] for idx, col in enumerate(all_columns)}

# Create tensor for prediction
fen_channels = convert_fen_to_channels(input_df.loc[0, 'FEN Before the move'])

# Debugging step for checking issues in channels:
print(f"FEN Channels shape: {fen_channels.shape}")
additional_channels = create_additional_channels(
    column_transformation_map['Format'],
    column_transformation_map['Time Control'],
    column_transformation_map['White Elo'],
    column_transformation_map['Black Elo'],
    column_transformation_map['White Clock Time'],
    column_transformation_map['Black Clock Time'],
    column_transformation_map['Phase']
)
print(f"Additional Channels shape: {additional_channels.shape}")

# Correcting any discrepancies in channel shapes if needed
try:
    input_tensor = np.concatenate((fen_channels, additional_channels), axis=-1)
except Exception as e:
    print(f"Error concatenating input tensors: {e}")

# Generate and reshape legal moves mask
legal_moves_mask = generate_legal_moves_mask(input_values['FEN Before the move'])
input_tensor = np.expand_dims(input_tensor, axis=0)
legal_moves_mask = np.expand_dims(legal_moves_mask, axis=0)

# Predict if the model loaded correctly
if 'model' in locals():
    predictions = model.predict([input_tensor, legal_moves_mask])

    # Decode moves and print results
    move_probabilities = predictions[0]
    decoded_moves = [(decode_move(idx), prob) for idx, prob in enumerate(move_probabilities[0])]
    decoded_moves.sort(key=lambda x: x[1], reverse=True)

    print("Top 10 move predictions with probabilities:")
    for move, prob in decoded_moves[:10]:
        print(f"Move: {move}, Probability: {prob:.4f}")

    # Use probabilities to select a move
    moves, probabilities = zip(*decoded_moves)
    chosen_move = random.choices(moves, weights=probabilities, k=1)[0]

    print(f"Chosen move by the bot: {chosen_move}")

    # Decode and sort think time probabilities
    think_time_probabilities = predictions[1][0]
    think_time_distribution = sorted(enumerate(think_time_probabilities), key=lambda x: x[1], reverse=True)

    print("Top 5 think time predictions with probabilities:")
    for time, prob in think_time_distribution[:5]:
        print(f"Think time: {time} seconds, Probability: {prob:.4f}")

    # Select a think time based on probabilities
    available_times = list(range(len(think_time_probabilities)))
    chosen_think_time = random.choices(available_times, weights=think_time_probabilities, k=1)[0]

    print(f"Chosen think time by the bot: {chosen_think_time} seconds")