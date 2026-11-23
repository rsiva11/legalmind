"""Ingest the legal knowledge base. Run from the backend folder:

    python ../scripts/ingest_knowledge.py --print-hashes   # sha256 of each file, for the manifest
    python ../scripts/ingest_knowledge.py --check          # validate the manifest only
    python ../scripts/ingest_knowledge.py                  # embed and store in Atlas
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.database.mongodb import close_mongo_connection, connect_to_mongo  # noqa: E402
from app.knowledge.knowledge_ingestion import ingest_knowledge  # noqa: E402
from app.knowledge.knowledge_manifest import ManifestError, file_sha256, load_manifest  # noqa: E402
from app.knowledge.knowledge_store import MongoKnowledgeBackend  # noqa: E402


async def _ingest(manifest) -> int:
    await connect_to_mongo()
    try:
        backend = MongoKnowledgeBackend()
        await backend.ensure_index()
        report = await ingest_knowledge(manifest, backend)
    finally:
        await close_mongo_connection()
    for r in report.sources:
        print(f"  {r.source_id}: {r.status} chunks={r.chunks} withheld={r.withheld} {r.error or ''}".rstrip())
    return 0 if report.ok else 1


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=None, help="knowledge folder (default: KNOWLEDGE_DIR)")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-hashes", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    directory = Path(args.dir or settings.knowledge_dir)

    if args.print_hashes:
        allowed = {e.lower() for e in settings.allowed_extensions}
        for p in sorted(directory.iterdir()):
            if p.is_file() and p.suffix.lower() in allowed:
                print(f"{file_sha256(p)}  {p.name}")
        return 0

    try:
        manifest = load_manifest(directory)
    except ManifestError as exc:
        print("Manifest problems:")
        for problem in exc.problems:
            print(f"  - {problem}")
        return 1
    print(f"Manifest OK: {len(manifest.sources)} source(s)")
    for name in manifest.unregistered_files:
        print(f"  ignored (not in manifest): {name}")
    if args.check:
        return 0
    return asyncio.run(_ingest(manifest))


if __name__ == "__main__":
    raise SystemExit(main())