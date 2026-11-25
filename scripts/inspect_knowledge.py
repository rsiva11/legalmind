"""Preview how the knowledge documents will be chunked. No database, no embeddings.

Run from the backend folder:
    python ../scripts/inspect_knowledge.py
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings  # noqa: E402
from app.ingestion.ingestion_pipeline import ingest_document  # noqa: E402
from app.knowledge.knowledge_manifest import ManifestError, load_manifest  # noqa: E402


def main() -> int:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(get_settings().knowledge_dir)
    try:
        manifest = load_manifest(directory)
    except ManifestError as exc:
        print("Manifest problems:")
        for problem in exc.problems:
            print(f"  - {problem}")
        return 1

    for source in manifest.sources:
        print(f"\n== {source.id} ==")
        try:
            result = ingest_document(manifest.path_of(source))
        except Exception as exc:
            print(f"  ingestion failed: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        sizes = sorted(c.char_count for c in result.chunks)
        short = sum(1 for s in sizes if s < 150)
        print(f"  pages={result.page_count} chunks={result.chunk_count} withheld={result.withheld_count}")
        print(f"  size min/median/max: {sizes[0]} / {statistics.median(sizes):.0f} / {sizes[-1]}   under 150 chars: {short}")
        for name, i in (("first", 0), ("middle", len(result.chunks) // 2), ("last", len(result.chunks) - 1)):
            c = result.chunks[i]
            print(f"  {name} (page {c.page_number}): {c.text[:120]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())