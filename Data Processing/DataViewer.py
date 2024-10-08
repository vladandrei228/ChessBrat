def read_in_chunks(file_path, chunk_size=1024):
    with open(file_path, 'r') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

# Count the number of chunks
#chunk_count = 0
#for chunk in read_in_chunks('lichess_db_standard_rated_2024-09.pgn'):
#    chunk_count += 1

def read_one_chunk(file_path, chunk_size=1024):
    with open(file_path, 'r') as file:
        chunk = file.read(chunk_size)
        return chunk

chunk_size = 10 * 1024 * 1024  # 10 MB
# Read and print one chunk
chunk = read_one_chunk('lichess_db_standard_rated_2024-09.pgn', chunk_size)
print(chunk)
