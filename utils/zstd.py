from zstandard import ZstdDecompressor, ZstdCompressor
from io import TextIOWrapper

def read_zstd(filename):
    """
    Open plain text or .zst-compressed text files transparently.
    """
    if str(filename).endswith(".zst"):
        fh = open(filename, "rb")
        decompressor = ZstdDecompressor(max_window_size=2**31)
        stream = decompressor.stream_reader(fh)
        return TextIOWrapper(stream, encoding="utf-8"), fh
    else:
        return open(filename, "r", encoding="utf-8"), None

def write_zstd(filename):
    """
    Open plain text or .zst-compressed output.
    """
    if not str(filename).endswith(".zst"):
        filename += ".zst"
    compressor = ZstdCompressor(level=3)
    fh = open(filename, "wb")
    stream = compressor.stream_writer(fh)
    return TextIOWrapper(stream, encoding="utf-8"), (stream, fh)