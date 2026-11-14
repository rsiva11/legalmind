import hashlib
import re
import uuid
from pathlib import Path


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def safe_stored_name(original: str) -> str:
    """Never trust a user-supplied filename. Generate our own."""
    ext = get_extension(original)
    return f"{uuid.uuid4().hex}{ext}"


def clean_display_name(original: str) -> str:
    """Keep a harmless version of the original name for display only."""
    name = Path(original).name
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name)[:200]


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()