import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".cache" / "sda_pyarrow"))
pa = pytest.importorskip("pyarrow")
pytest.importorskip("huggingface_hub")


def _module():
    directory = ROOT / "reports" / "sda_fineweb_source_probe_20261109"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("sda_fineweb_fetch", directory / "fetch_candidates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_fetch_resume_checks_identity_and_bytes(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "TARGET", tmp_path)
    class Reader:
        requests = 2
        transferred = 10
    class Parquet:
        metadata = type("Metadata", (), {"row_group": lambda self, i: type("Group", (), {"num_rows": 1})()})()
        def read_row_group(self, index, columns):
            assert index == 3 and columns == module.IDENTITY["columns"]
            return pa.table({column: [column] for column in columns})
    monkeypatch.setattr(module, "RangeReader", Reader)
    monkeypatch.setattr(module.pq, "ParquetFile", lambda reader: Parquet())
    record = module.get_group(3)
    assert record["rows"] == 1 and record["identity"]["source_sha256_status"] == "remote_metadata_only"
    monkeypatch.setattr(module, "RangeReader", lambda: pytest.fail("resume must not fetch again"))
    assert module.get_group(3) == record
    with (tmp_path / "group-0003.parquet").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.get_group(3)


def test_candidate_fetch_refuses_orphaned_existing_group(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "TARGET", tmp_path)
    (tmp_path / "group-0000.parquet").write_bytes(b"existing")
    with pytest.raises(ValueError, match="incomplete existing group"):
        module.get_group(0)
