import chess
import numpy as np

np.set_printoptions(threshold=np.inf)

# Function to convert FEN to multi-channel 8x8x17 representation
def convert_fen_to_channels(fen):
    board = chess.Board(fen)
    channels = np.zeros((8, 8, 17), dtype=np.float32)  
    piece_map = board.piece_map()
    
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        channel_index = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        channels[row, col, channel_index] = 1
    
    # Add additional game state information
    channels[0, 0, 12] = board.turn
    channels[0, 0, 13] = (
        (board.has_kingside_castling_rights(chess.WHITE) << 0) |
        (board.has_queenside_castling_rights(chess.WHITE) << 1) |
        (board.has_kingside_castling_rights(chess.BLACK) << 2) |
        (board.has_queenside_castling_rights(chess.BLACK) << 3)
    )
    channels[0, 0, 14] = board.ep_square if board.ep_square is not None else -1
    channels[0, 0, 15] = board.halfmove_clock
    channels[0, 0, 16] = board.fullmove_number
    
    return channels
# Function to convert multi-channel 8x8x17 representation back to FEN
def convert_channels_to_fen(channels):
    board = chess.Board()
    board.clear_board()
    
    for row in range(8):
        for col in range(8):
            for channel in range(12):
                if channels[row, col, channel] == 1:
                    piece_type = channel % 6 + 1
                    color = chess.BLACK if channel >= 6 else chess.WHITE
                    piece = chess.Piece(piece_type, color)
                    square = chess.square(col, row)
                    board.set_piece_at(square, piece)
    
    # Retrieve additional game state information
    board.turn = bool(channels[0, 0, 12])
    
    # Correctly set castling rights
    castling_rights = int(channels[0, 0, 13])
    if castling_rights & 1:
        board.castling_rights |= chess.BB_H1
    if castling_rights & 2:
        board.castling_rights |= chess.BB_A1
    if castling_rights & 4:
        board.castling_rights |= chess.BB_H8
    if castling_rights & 8:
        board.castling_rights |= chess.BB_A8
    
    board.ep_square = int(channels[0, 0, 14]) if channels[0, 0, 14] != -1 else None
    board.halfmove_clock = int(channels[0, 0, 15])
    board.fullmove_number = int(channels[0, 0, 16])
    
    return board.fen()

def create_additional_channels(format, time_control, white_elo, black_elo, white_clock_time, black_clock_time, phase):
    additional_channels = np.zeros(7, dtype=np.float32)
    
    # Format channel
    additional_channels[0] = format
    
    # Time Control channel
    additional_channels[1] = time_control
    
    # White ELO channel
    additional_channels[2] = white_elo 
    
    # Black ELO channel
    additional_channels[3] = black_elo
    
    # White Clock channel
    additional_channels[4] = white_clock_time 
    
    # Black Clock channel
    additional_channels[5] = black_clock_time 
    
    # Phase channel
    additional_channels[6] = phase
    
    return additional_channels