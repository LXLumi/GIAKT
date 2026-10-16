"""File integrity checks shared by preprocessing and checkpoint evaluation."""
import hashlib
from pathlib import Path


def sha256(path):
    """Hash a file in bounded-size blocks without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
