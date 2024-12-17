import zstandard as zstd

with open('lichess_db_standard_rated_2024-08.pgn.zst', 'rb') as compressed_file:
    dctx = zstd.ZstdDecompressor()

    with open('lichess_db_standard_rated_2024-08.pgn', 'wb') as decompressed_file:
        dctx.copy_stream(compressed_file, decompressed_file)

print("Decompression complete!")
