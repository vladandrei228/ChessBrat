import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (GlobalAveragePooling2D, Conv2D,MaxPooling2D, Input, 
                                     Reshape, Lambda, Flatten, Dense, BatchNormalization, Concatenate, 
                                     MultiHeadAttention, Add, LayerNormalization)
from CNN.Masks import LegalMoveMask, DropConnect

# Load pre-trained weights from .pb file
def load_pb_model(pb_file_path):
    with tf.io.gfile.GFile(pb_file_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
    return tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())

session = load_pb_model("C:\\Users\\bnc\\Downloads\\ChessBrat\\CNN\\weights\\maia_weights\\maia-1500.pb")

# Example usage in a model
def create_cnn_attention_model(input_shape=(8, 8, 24), nb_classes=4273, max_think_time=60):
    # Input layer
    combined_input = Input(shape=input_shape, name='combined_input')

    # Split the combined input
    board_state_input = Lambda(lambda x: x[:, :, :, :17])(combined_input)
    additional_features_input = Lambda(lambda x: x[:, :, :, 17:])(combined_input)
    additional_features_input = GlobalAveragePooling2D()(additional_features_input)

    # Process board state
    x_board = Conv2D(64, (3, 3), activation='relu', padding='same')(board_state_input)
    x_board = BatchNormalization()(x_board)
    x_board = MaxPooling2D((2, 2))(x_board)

    x_board = Conv2D(128, (3, 3), activation='relu', padding='same')(x_board)
    x_board = BatchNormalization()(x_board)
    x_board = MaxPooling2D((2, 2))(x_board)

    x_board = Conv2D(256, (3, 3), activation='relu', padding='same')(x_board)
    x_board = BatchNormalization()(x_board)
    x_board = GlobalAveragePooling2D()(x_board)

    # Merge board and additional features
    x = Concatenate()([x_board, additional_features_input])
    x = Reshape((1, -1))(x) 

    # Attention mechanism
    attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)
    x = Flatten()(x)

    # Move output
    move_output = Dense(512, activation='relu')(x)
    move_output = DropConnect(0.3)(move_output)
    move_output = Dense(nb_classes, activation='softmax', name='move_output')(move_output)

    time_output_categorical = Dense(512, activation='relu')(x)
    time_output_categorical = DropConnect(0.5)(time_output_categorical)
    time_output_categorical = Dense(max_think_time + 1, activation='softmax', name='time_output_categorical')(time_output_categorical)
    
    # Legal move mask
    legal_mask_input = Input(shape=(nb_classes,), name='legal_mask_input')
    masked_move_output = LegalMoveMask(name='legal_masking')([move_output, legal_mask_input])

    model = Model(inputs=[combined_input, legal_mask_input],
                  outputs=[masked_move_output, time_output_categorical])

    return model

def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss={
            'legal_masking': 'sparse_categorical_crossentropy',
            'time_output_categorical': 'sparse_categorical_crossentropy',
        },
        metrics={
            'legal_masking': 'accuracy',
            'time_output_categorical': 'accuracy',
        }
    )
    return model

""" from keras.utils import plot_model

attention = create_cnn_attention_model()
attention.summary()
plot_model(attention, to_file="Attention.png", show_shapes=True) """



