import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.config import get_settings

MANIFEST_NAME = "manifest.yaml"
_SOURCE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ManifestError(Exception):
    """The manifest is unusable. `problems` never contain input values or document text."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


class KnowledgeSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = Field(min_length=3, max_length=200)
    file: str
    sha256: str
    jurisdiction: str = Field(min_length=2, max_length=60)
    publisher: str = Field(min_length=2, max_length=200)
    source_url: str
    retrieved_on: date
    version_note: str | None = Field(None, max_length=200)
    terms_note: str = Field(min_length=10, max_length=500)
    terms_verified: bool = False

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not _SOURCE_ID.fullmatch(v):
            raise ValueError("id must be lowercase letters, digits and hyphens (1-64 characters)")
        return v

    @field_validator("file")
    @classmethod
    def _check_file(cls, v: str) -> str:
        if not _FILE_NAME.fullmatch(v):
            raise ValueError("file must be a plain file name in the knowledge folder")
        return v

    @field_validator("sha256")
    @classmethod
    def _check_sha(cls, v: str) -> str:
        v = v.strip().lower()
        if not _SHA256.fullmatch(v):
            raise ValueError("sha256 must be 64 hexadecimal characters")
        return v

    @field_validator("source_url")
    @classmethod
    def _check_url(cls, v: str) -> str:
        try:
            parts = urlsplit(v.strip())
            host = parts.hostname
        except ValueError:
            raise ValueError("source_url is not a valid URL") from None
        if parts.scheme != "https" or not host or parts.username or parts.password:
            raise ValueError("source_url must be an https URL without credentials")
        return v.strip()


class KnowledgeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    sources: list[KnowledgeSource] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self):
        if len({s.id for s in self.sources}) != len(self.sources):
            raise ValueError("duplicate source ids")
        if len({s.file.lower() for s in self.sources}) != len(self.sources):
            raise ValueError("duplicate files")
        return self


class LoadedManifest(BaseModel):
    directory: str
    sources: list[KnowledgeSource]
    unregistered_files: list[str] = []

    @property
    def source_ids(self) -> list[str]:
        return [s.id for s in self.sources]

    def get(self, source_id: str) -> KnowledgeSource | None:
        return next((s for s in self.sources if s.id == source_id), None)

    def path_of(self, source: KnowledgeSource) -> Path:
        return Path(self.directory) / source.file


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _problems_from(exc: ValidationError) -> list[str]:
    out = []
    for e in exc.errors():
        where = ".".join(str(part) for part in e["loc"]) or "manifest"
        out.append(f"{where}: {e['msg']}")  # the message never contains the offending value
    return out


def load_manifest(
    directory: Path | str,
    *,
    allowed_jurisdictions: set[str] | None = None,
    require_terms_verified: bool = True,
) -> LoadedManifest:
    settings = get_settings()
    base = Path(directory)
    if not base.is_dir():
        raise ManifestError([f"folder not found: {base}"])
    path = base / MANIFEST_NAME
    if not path.is_file():
        raise ManifestError(["manifest.yaml not found"])
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError):
        raise ManifestError(["manifest.yaml is not valid YAML"]) from None
    if not isinstance(raw, dict):
        raise ManifestError(["manifest.yaml must be a mapping with 'version' and 'sources'"])
    try:
        manifest = KnowledgeManifest.model_validate(raw)
    except ValidationError as exc:
        raise ManifestError(_problems_from(exc)) from None

    jurisdictions = {j.lower() for j in (allowed_jurisdictions or settings.knowledge_allowed_jurisdictions)}
    extensions = {e.lower() for e in settings.allowed_extensions}
    problems: list[str] = []
    for s in manifest.sources:
        if s.jurisdiction.lower() not in jurisdictions:
            problems.append(f"{s.id}: jurisdiction is not allowed")
        if require_terms_verified and not s.terms_verified:
            problems.append(f"{s.id}: terms_verified is false; read the publisher's terms of use first")
        file_path = base / s.file
        if Path(s.file).suffix.lower() not in extensions:
            problems.append(f"{s.id}: file type is not allowed")
        elif file_path.is_symlink() or not file_path.is_file():
            problems.append(f"{s.id}: file not found")
        elif file_sha256(file_path) != s.sha256:
            problems.append(f"{s.id}: file does not match the recorded sha256")
    if problems:
        raise ManifestError(problems)

    registered = {s.file.lower() for s in manifest.sources}
    unregistered = sorted(
        p.name
        for p in base.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() not in {".yaml", ".yml"}
        and p.name.lower() not in registered
    )
    return LoadedManifest(directory=str(base), sources=manifest.sources, unregistered_files=unregistered)