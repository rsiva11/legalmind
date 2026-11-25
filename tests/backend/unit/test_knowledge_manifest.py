import pytest
import yaml

from knowledge_fixture import JURISDICTION, write_fixture, write_manifest
from app.knowledge.knowledge_manifest import ManifestError, file_sha256, load_manifest

IDS = ["synthetic-garden-rules", "synthetic-notice-policy"]


def load(directory, **kw):
    return load_manifest(directory, allowed_jurisdictions={JURISDICTION}, **kw)


def problems(directory, **kw):
    with pytest.raises(ManifestError) as exc:
        load(directory, **kw)
    return exc.value.problems


def rewrite(directory, change):
    path = directory / "manifest.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_valid_manifest_loads(tmp_path):
    write_fixture(tmp_path)
    m = load(tmp_path)
    assert m.source_ids == IDS
    assert m.unregistered_files == ["unregistered-note.txt"]
    notice = m.get("synthetic-notice-policy")
    assert notice.title == "Synthetic Notice Policy (test fixture, not law)" and notice.jurisdiction == JURISDICTION
    assert m.path_of(notice).name == "synthetic-notice-policy.txt" and m.path_of(notice).is_file()
    assert m.get("unregistered-note") is None and m.get("../etc/passwd") is None


def test_default_jurisdiction_is_india_only(tmp_path):
    write_fixture(tmp_path)
    with pytest.raises(ManifestError) as exc:
        load_manifest(tmp_path)
    assert exc.value.problems == [f"{i}: jurisdiction is not allowed" for i in IDS]

    india = tmp_path / "india"
    write_manifest(india, {"a-act.txt": "Some text.\n"}, jurisdiction="India")
    assert load_manifest(india).source_ids == ["a-act"]


def test_terms_must_be_verified(tmp_path):
    write_fixture(tmp_path, terms_verified=False)
    found = problems(tmp_path)
    assert len(found) == 2 and all("terms_verified is false" in p for p in found)
    assert load(tmp_path, require_terms_verified=False).source_ids == IDS


def test_missing_manifest(tmp_path):
    assert problems(tmp_path) == ["manifest.yaml not found"]


def test_missing_folder(tmp_path):
    found = problems(tmp_path / "nope")
    assert len(found) == 1 and found[0].startswith("folder not found")


def test_invalid_yaml(tmp_path):
    (tmp_path / "manifest.yaml").write_text("sources: [unclosed", encoding="utf-8")
    assert problems(tmp_path) == ["manifest.yaml is not valid YAML"]


def test_manifest_must_be_a_mapping(tmp_path):
    (tmp_path / "manifest.yaml").write_text("- a\n- b\n", encoding="utf-8")
    assert problems(tmp_path) == ["manifest.yaml must be a mapping with 'version' and 'sources'"]


def test_duplicate_ids_and_files(tmp_path):
    write_fixture(tmp_path)
    rewrite(tmp_path, lambda d: d["sources"][1].update(id=d["sources"][0]["id"]))
    assert any("duplicate source ids" in p for p in problems(tmp_path))
    write_fixture(tmp_path)
    rewrite(tmp_path, lambda d: d["sources"][1].update(file=d["sources"][0]["file"]))
    assert any("duplicate files" in p for p in problems(tmp_path))


def test_hash_mismatch_is_detected(tmp_path):
    write_fixture(tmp_path)
    (tmp_path / "synthetic-garden-rules.txt").write_bytes(b"changed after registration")
    assert problems(tmp_path) == ["synthetic-garden-rules: file does not match the recorded sha256"]


def test_missing_file_is_detected(tmp_path):
    write_fixture(tmp_path)
    (tmp_path / "synthetic-notice-policy.txt").unlink()
    assert problems(tmp_path) == ["synthetic-notice-policy: file not found"]


def test_disallowed_extension(tmp_path):
    write_fixture(tmp_path)
    rewrite(tmp_path, lambda d: d["sources"][0].update(file="tool.exe"))
    assert problems(tmp_path) == ["synthetic-garden-rules: file type is not allowed"]


def test_error_messages_do_not_echo_input(tmp_path):
    write_fixture(tmp_path)
    rewrite(tmp_path, lambda d: d["sources"][0].update(sha256="SECRET-VALUE-123"))
    found = problems(tmp_path)
    assert found and all("SECRET-VALUE-123" not in p for p in found)


def test_file_sha256(tmp_path):
    empty, abc = tmp_path / "empty.txt", tmp_path / "abc.txt"
    empty.write_bytes(b"")
    abc.write_bytes(b"abc")
    assert file_sha256(empty) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert file_sha256(abc) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_unregistered_files_are_listed_and_hidden_files_ignored(tmp_path):
    write_fixture(tmp_path)
    for name in (".gitkeep", "manifest.example.yaml", "notes.md"):
        (tmp_path / name).write_bytes(b"x")
    assert load(tmp_path).unregistered_files == ["notes.md", "unregistered-note.txt"]


BAD_ENTRIES = [
    ("sources.0.id", lambda d: d["sources"][0].update(id="Bad_ID")),
    ("sources.0.file", lambda d: d["sources"][0].update(file="../x.txt")),
    ("sources.0.sha256", lambda d: d["sources"][0].update(sha256="abc")),
    ("sources.0.source_url", lambda d: d["sources"][0].update(source_url="http://example.gov/x")),
    ("sources.0.source_url", lambda d: d["sources"][0].update(source_url="https://user:pw@example.gov/x")),
    ("sources.0.terms_note", lambda d: d["sources"][0].pop("terms_note")),
    ("sources.0.terms_note", lambda d: d["sources"][0].update(terms_note="short")),
    ("sources.0.extra_field", lambda d: d["sources"][0].update(extra_field="x")),
    ("version", lambda d: d.update(version=2)),
    ("sources", lambda d: d.update(sources=[])),
]


@pytest.mark.parametrize("where, change", BAD_ENTRIES)
def test_invalid_entries_are_rejected_with_their_location(tmp_path, where, change):
    write_fixture(tmp_path)
    rewrite(tmp_path, change)
    assert any(p.startswith(where + ":") for p in problems(tmp_path))