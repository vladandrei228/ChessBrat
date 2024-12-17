file_to_number = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
number_to_file = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h'}

index_mapping = {
    ('a', 'b'): 0, ('b', 'a'): 1, ('b', 'c'): 2, ('c', 'b'): 3,
    ('c', 'd'): 4, ('d', 'c'): 5, ('d', 'e'): 6, ('e', 'f'): 7,
    ('f', 'g'): 8, ('g', 'f'): 9, ('h', 'g'): 10, ('g', 'h'): 11
}
reverse_index_mapping = {
    0 : ('a', 'b'), 1: ('b', 'a'), 2: ('b', 'c'), 3: ('c', 'b'),
    4: ('c', 'd'), 5: ('d', 'c'), 6: ('d', 'e'), 7: ('e', 'f'),
    8: ('f', 'g'), 9: ('g', 'f'), 10: ('h', 'g'), 11: ('g', 'h')
}

straight_promotion = {
  4097: "a2a1q", 4098: "a2a1r", 4099: "a2a1b", 4100: "a2a1n",
  4101: "b2b1q", 4102: "b2b1r", 4103: "b2b1b", 4104: "b2b1n",
  4105: "c2c1q", 4106: "c2c1r", 4107: "c2c1b", 4108: "c2c1n",
  4109: "d2d1q", 4110: "d2d1r", 4111: "d2d1b", 4112: "d2d1n",
  4113: "e2e1q", 4114: "e2e1r", 4115: "e2e1b", 4116: "e2e1n",
  4117: "f2f1q", 4118: "f2f1r", 4119: "f2f1b", 4120: "f2f1n",
  4121: "g2g1q", 4122: "g2g1r", 4123: "g2g1b", 4124: "g2g1n",
  4125: "h2h1q", 4126: "h2h1r", 4127: "h2h1b", 4128: "h2h1n",
  
  4129: "a7a8q", 4130: "a7a8r", 4131: "a7a8b", 4132: "a7a8n",
  4133: "b7b8q", 4134: "b7b8r", 4135: "b7b8b", 4136: "b7b8n",
  4137: "c7c8q", 4138: "c7c8r", 4139: "c7c8b", 4140: "c7c8n",
  4141: "d7d8q", 4142: "d7d8r", 4143: "d7d8b", 4144: "d7d8n",
  4145: "e7e8q", 4146: "e7e8r", 4147: "e7e8b", 4148: "e7e8n",
  4149: "f7f8q", 4150: "f7f8r", 4151: "f7f8b", 4152: "f7f8n",
  4153: "g7g8q", 4154: "g7g8r", 4155: "g7g8b", 4156: "g7g8n",
  4157: "h7h8q", 4158: "h7h8r", 4159: "h7h8b", 4160: "h7h8n"
}
diagonal_promotion = {
  4161: "a7b8q", 4162: "a7b8r", 4163: "a7b8b", 4164: "a7b8n",
  4165: "b7a8q", 4166: "b7a8r", 4167: "b7a8b", 4168: "b7a8n",
  4169: "b7c8q", 4170: "b7c8r", 4171: "b7c8b", 4172: "b7c8n",
  4173: "c7b8q", 4174: "c7b8r", 4175: "c7b8b", 4176: "c7b8n",
  4177: "c7d8q", 4178: "c7d8r", 4179: "c7d8b", 4180: "c7d8n",
  4181: "d7c8q", 4182: "d7c8r", 4183: "d7c8b", 4184: "d7c8n",
  4185: "d7e8q", 4186: "d7e8r", 4187: "d7e8b", 4188: "d7e8n",
  4189: "e7d8q", 4190: "e7d8r", 4191: "e7d8b", 4192: "e7d8n",
  4193: "e7f8q", 4194: "e7f8r", 4195: "e7f8b", 4196: "e7f8n",
  4197: "f7e8q", 4198: "f7e8r", 4199: "f7e8b", 4200: "f7e8n",
  4201: "f7g8q", 4202: "f7g8r", 4203: "f7g8b", 4204: "f7g8n",
  4205: "g7f8q", 4206: "g7f8r", 4207: "g7f8b", 4208: "g7f8n",
  4209: "g7h8q", 4210: "g7h8r", 4211: "g7h8b", 4212: "g7h8n",
  4213: "h7g8q", 4214: "h7g8r", 4215: "h7g8b", 4216: "h7g8n",
  
  4217: "a2b1q", 4218: "a2b1r", 4219: "a2b1b", 4220: "a2b1n",
  4221: "b2a1q", 4222: "b2a1r", 4223: "b2a1b", 4224: "b2a1n",
  4225: "b2c1q", 4226: "b2c1r", 4227: "b2c1b", 4228: "b2c1n",
  4229: "c2b1q", 4230: "c2b1r", 4231: "c2b1b", 4232: "c2b1n",
  4233: "c2d1q", 4234: "c2d1r", 4235: "c2d1b", 4236: "c2d1n",
  4237: "d2c1q", 4238: "d2c1r", 4239: "d2c1b", 4240: "d2c1n",
  4241: "d2e1q", 4242: "d2e1r", 4243: "d2e1b", 4244: "d2e1n",
  4245: "e2d1q", 4246: "e2d1r", 4247: "e2d1b", 4248: "e2d1n",
  4249: "e2f1q", 4250: "e2f1r", 4251: "e2f1b", 4252: "e2f1n",
  4253: "f2e1q", 4254: "f2e1r", 4255: "f2e1b", 4256: "f2e1n",
  4257: "f2g1q", 4258: "f2g1r", 4259: "f2g1b", 4260: "f2g1n",
  4261: "g2f1q", 4262: "g2f1r", 4263: "g2f1b", 4264: "g2f1n",
  4265: "g2h1q", 4266: "g2h1r", 4267: "g2h1b", 4268: "g2h1n",
  4269: "h2g1q", 4270: "h2g1r", 4271: "h2g1b", 4272: "h2g1n"
}




# Function to encode a move
def encode_move(move: str) -> int:
    if len(move) < 4:
        raise ValueError("Move must be at least 4 characters long.")

    file1, rank1, file2, rank2 = move[0], int(move[1]), move[2], int(move[3])
    promotion_type = move[4] if len(move) == 5 else None

    # Validate files and ranks
    if file1 not in file_to_number or file2 not in file_to_number:
        raise ValueError(f"Invalid file: {file1} or {file2}")
    if not (1 <= rank1 <= 8) or not (1 <= rank2 <= 8):
        raise ValueError(f"Invalid rank: {rank1} or {rank2}")

    # Base encoded value calculation
    encoded = (file_to_number[file1] - 1) * 512 + (rank1 - 1) * 64 + \
              (file_to_number[file2] - 1) * 8 + (rank2 - 1) + 1
              
    # Handle promotions using the arrays
    if promotion_type:
        if (rank1 == 7 and rank2 == 8) and file1 == file2:  # Straight promotion
            for key, value in straight_promotion.items():
                if value == f"{file1}{rank1}{file2}{rank2}{promotion_type}":
                    return key
        elif (rank1 == 2 and rank2 == 1) and file1 == file2:  # Straight promotion
            for key, value in straight_promotion.items():
                if value == f"{file1}{rank1}{file2}{rank2}{promotion_type}":
                    return key
        elif (rank1 == 7 and rank2 == 8) and file1 != file2:  # Diagonal promotion
            for key, value in diagonal_promotion.items():
                if value == f"{file1}{rank1}{file2}{rank2}{promotion_type}":
                    return key
        elif (rank1 == 2 and rank2 == 1) and file1 != file2:  # Diagonal promotion
            for key, value in diagonal_promotion.items():
                if value == f"{file1}{rank1}{file2}{rank2}{promotion_type}":
                    return key
                
    return encoded

def decode_move(encoded: int) -> str:
    # Handle promotions
    if encoded >= 4096:
        if encoded >= 4160:  # Diagonal promotions
            for key, value in diagonal_promotion.items():
                if key == encoded:
                    return value
        else:  # Straight promotions
            for key, value in straight_promotion.items():
                if key == encoded:
                    return value

    # Regular moves
    encoded -= 1
    rank2 = (encoded % 8) + 1
    encoded //= 8
    file2 = number_to_file[encoded % 8 + 1]
    encoded //= 8
    rank1 = (encoded % 8) + 1
    encoded //= 8
    file1 = number_to_file[encoded % 8 + 1]

    return f"{file1}{rank1}{file2}{rank2}"

print(decode_move(2470))
