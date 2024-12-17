import numpy as np
import pandas as pd
import joblib
from keras.callbacks import EarlyStopping
from CNN.Attention import create_cnn_attention_model, compile_model
from MoveConverter import encode_move
from ChannelConverter import convert_fen_to_channels, create_additional_channels
import chess
import tensorflow as tf
from sklearn.model_selection import KFold

print("Model 1500")
# Adjust display settings to show all columns
pd.set_option('display.max_columns', None)

# Path to your dataset
file_path = 'DataFrames/game-data_1500.csv'
categorical_cols = ['Format', 'Time Control', 'Phase']
numerical_cols = ['White Elo', 'Black Elo', 'White Clock Time', 'Black Clock Time']

# Read the DataFrame
df = pd.read_csv(file_path)

# Load pre-existing label encoders and preprocessor
label_encoders = joblib.load('label_encoders.joblib')
preprocessor = joblib.load('preprocessor.joblib')

# Apply label encoding to categorical columns
for col in categorical_cols:
    df[col] = label_encoders[col].transform(df[col])

# Transform the specified columns using the loaded preprocessor
transformed_data = preprocessor.transform(df)

# Combine transformed and untouched columns
all_columns = numerical_cols + categorical_cols

# Create a new DataFrame with the transformed data
df_transformed = pd.DataFrame(transformed_data, columns=all_columns)

# Print the first value in each column of the transformed DataFrame
print("First values in each column of the transformed DataFrame:")
for col in df_transformed.columns:
    print(f"{col}: {df_transformed[col].iloc[0]}")

df['Think Time'] = df['Think Time'].astype(int)
df.loc[df['Think Time'] < 0, 'Think Time'] = 0
max_think_time = df['Think Time'].max()

print(f"FEN Before the move: {df.loc[0, 'FEN Before the move']}")
print(f"Move: {df.loc[0, 'Move']}")
print(f"Think Time: {df.loc[0, 'Think Time']}")

# Separate function to handle inputs
def create_combined_input_tensor(fen, format, time_control, white_elo, black_elo, white_clock_time, black_clock_time, phase):
    # Convert FEN to board state channels (should be 8x8x17)
    board_state_channels = convert_fen_to_channels(fen)
    if board_state_channels.shape != (8, 8, 17):
        raise ValueError(f"Invalid shape for board_state_channels: {board_state_channels.shape}")

    # Convert additional features (should be 7,)
    additional_channels = create_additional_channels(format, time_control, white_elo, black_elo, white_clock_time, black_clock_time, phase)
    if additional_channels.shape != (7,):
        raise ValueError(f"Invalid shape for additional_channels: {additional_channels.shape}")

    # Expand additional channels to match the board state shape (8x8x7)
    additional_channels_expanded = np.expand_dims(np.expand_dims(additional_channels, axis=0), axis=0)
    additional_channels_expanded = np.tile(additional_channels_expanded, (8, 8, 1))

    # Combine into a single tensor
    combined_tensor = np.concatenate((board_state_channels, additional_channels_expanded), axis=-1)

    return combined_tensor

def generate_legal_moves_mask(fen, nb_classes):
    board = chess.Board(fen)
    legal_moves_mask = np.zeros(nb_classes, dtype=np.float32)
    for move in board.legal_moves:
        encoded_move = encode_move(move.uci())
        if encoded_move < nb_classes:
            legal_moves_mask[encoded_move] = 1
    return legal_moves_mask

def tensor_generator(df_transformed, original_df, nb_classes=4273):
    def gen():
        for index, row in df_transformed.iterrows():
            fen = original_df.loc[index, 'FEN Before the move']
            format = row['Format']
            time_control = row['Time Control']
            white_elo = row['White Elo']
            black_elo = row['Black Elo']
            white_clock_time = row['White Clock Time']
            black_clock_time = row['Black Clock Time']
            phase = row['Phase']
            move = encode_move(original_df.loc[index, 'Move'])
            think_time = original_df.loc[index, 'Think Time']
            combined_tensor = create_combined_input_tensor(
                fen, format, time_control, white_elo, black_elo, white_clock_time, black_clock_time, phase)

            legal_mask = generate_legal_moves_mask(fen, nb_classes=nb_classes)

            yield {
                "combined_input": tf.convert_to_tensor(combined_tensor, dtype=tf.float32),
                "legal_mask_input": tf.convert_to_tensor(legal_mask, dtype=tf.float32)
            }, {
                "legal_masking": move,
                "time_output_categorical": think_time
            }

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                "combined_input": tf.TensorSpec(shape=(8, 8, 24), dtype=tf.float32),
                "legal_mask_input": tf.TensorSpec(shape=(nb_classes,), dtype=tf.float32),
            },
            {
                "legal_masking": tf.TensorSpec(shape=(), dtype=tf.int32),
                "time_output_categorical": tf.TensorSpec(shape=(), dtype=tf.int32),
            }
        )
    )
    
# Use the Dataset properly in training
train_dataset = tensor_generator(df_transformed, df)

# Configure your batch size
batch_size = 32

# Prepare datasets
kf = KFold(n_splits=5)
for train_index, val_index in kf.split(df_transformed):
    train_data = df_transformed.iloc[train_index]
    val_data = df_transformed.iloc[val_index]
    
    train_dataset = tensor_generator(df_transformed, df).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tensor_generator(val_data, df).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Model training integration with strategy scope
model = create_cnn_attention_model((8, 8, 24), nb_classes=4273, max_think_time=max_think_time + 1)
compile_model(model)

class CustomEarlyStopping(EarlyStopping):
    def _implements_train_batch_hooks(self):
        return False

    def _implements_test_batch_hooks(self):
        return False

    def _implements_predict_batch_hooks(self):
        return False

# Use the custom callback
model.fit(
    train_dataset,
    steps_per_epoch=len(train_data) // batch_size,
    epochs=10,
    validation_data=val_dataset,
    validation_steps=len(val_data) // batch_size,
    callbacks=[CustomEarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)]
)

# Save the model after training
model.save('chessbrat_1500_model.keras')
print("MODEL IS SAVED")