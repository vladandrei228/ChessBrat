import chess

def generate_pgn(game_data):
    header = """[Event "Classical"]
[Site "Lichess"]
[Date "2024.12.09"]
[White "ChessBrat"]
[Black "Random 1800 Player"]
[Result "0-1"]
[WhiteElo "1831"]
[BlackElo "1821"]
[TimeControl "1800+0"]\n"""

    moves_with_clocks = []
    board = chess.Board()  # Initialize the board with the standard starting position

    for i, (move_uci, white_time, black_time) in enumerate(game_data):
        try:
            move = chess.Move.from_uci(move_uci)

            if move not in board.legal_moves:  # Ensure the move is legal
                print(f"Illegal move: {move_uci} on board:\n{board}")
                continue

            move_san = board.san(move)  # Convert move to SAN
            board.push(move)  # Apply the move to the board

            move_number = i // 2 + 1
            if i % 2 == 0:  # White's move
                moves_with_clocks.append(f"{move_number}. {move_san} {{[%clk {white_time}]}}")
            else:  # Black's move
                moves_with_clocks.append(f"{move_san} {{[%clk {black_time}]}}")

        except ValueError as e:
            print(f"Invalid move detected, skipping: {move_uci} - Error: {e}")

    moves_string = ' '.join(moves_with_clocks)

    # Assemble the full PGN text
    pgn = header + moves_string + " 0-1"
    return pgn

# Updated list of game events
game_events = [
    ("d2d4", 1800, 1800),
    ("g8f6", 1800, 1800),
    ("c1g5", 1800, 1800),
    ("g7g6", 1797, 1798),
    ("d1d2", 1797, 1798),
    ("h7h6", 1797, 1794),
    ("g5e3", 1789, 1794),
    ("f6g4", 1789, 1789),
    ("e3f4", 1786, 1789),
    ("f8g7", 1786, 1785),
    ("h2h3", 1771, 1785),
    ("g4f6", 1771, 1779),
    ("b1c3", 1769, 1779),
    ("d7d5", 1769, 1775),
    ("e1c1", 1766, 1775),
    ("c8f5", 1766, 1773),
    ("g2g4", 1716, 1773),
    ("f5d7", 1716, 1762),
    ("f4e5", 1708, 1762),
    ("c7c6", 1708, 1759),
    ("e2e4", 1704, 1759),
    ("d5e4", 1704, 1754),
    ("c3e4", 1676, 1754),
    ("f6e4", 1676, 1752),
    ("e5g7", 1675, 1752),
    ("e4d2", 1675, 1750),
    ("g7h8", 1674, 1750),
    ("f7f6", 1674, 1749),
    ("d1d2", 1672, 1749),
    ("e8f7", 1672, 1746),
    ("f1c4", 1664, 1746),
    ("d7e6", 1664, 1741),
    ("c4e6", 1659, 1741),
    ("f7e6", 1659, 1736),
    ("h8g7", 1659, 1736),
    ("e6f7", 1659, 1735),
    ("g7h6", 1655, 1735),
    ("g6g5", 1655, 1734),
    ("h3h4", 1652, 1734),
    ("d8d5", 1652, 1730),
    ("f2f3", 1643, 1730),
    ("b8d7", 1643, 1724),
    ("h4g5", 1633, 1724),
    ("f6g5", 1633, 1722),
    ("d2d3", 1631, 1722),
    ("d7f6", 1631, 1714),
    ("b2b3", 1606, 1714),
    ("a7a5", 1606, 1709),
    ("h1h5", 1604, 1709),
    ("a5a4", 1604, 1707),
    ("h5g5", 1601, 1707),
    ("d5d6", 1601, 1706),
    ("g5g7", 1596, 1706),
    ("f7e6", 1596, 1700),
    ("g7g6", 1594, 1700),
    ("a4b3", 1594, 1696),
    ("d3e3", 1574, 1696),
    ("e6d7", 1574, 1659),
    ("g6g7", 1569, 1659),
    ("b3a2", 1569, 1647),
    ("e3e7", 1563, 1647),
    ("d6e7", 1563, 1645),
    ("g7e7", 1562, 1645),
    ("d7e7", 1562, 1644),
    ("h6g5", 1561, 1644),
    ("a2a1q", 1561, 1641),
    ("c1d2", 1560, 1641),
    ("a1a5", 1560, 1640),
    ("c2c3", 1557, 1640),
    ("a5g5", 1557, 1637),
    ("d2d3", 1556, 1637),
    ("c6c5", 1556, 1633),
    ("g1e2", 1553, 1633),
    ("c5c4", 1553, 1631),
    ("d3c4", 1551, 1631),
    ("a8a5", 1551, 1630),
    ("c4d3", 1547, 1630),
    ("g5b5", 1547, 1621),
    ("d3e3", 1542, 1621),
    ("a5a2", 1542, 1615),
    ("e2f4", 1540, 1615),
    ("f6d5", 1540, 1612),
    ("f4d5", 1537, 1612),
    ("b5d5", 1537, 1586),
    ("f3f4", 1535, 1586),
    ("d5c4", 1535, 1582),
    ("g4g5", 1529, 1582),
    ("c4e2", 1529, 1580)
]

pgn_output = generate_pgn(game_events)
print(pgn_output)