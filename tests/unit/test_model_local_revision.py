from obfuscation_attacks.schemes.ab_deployment import local_model_revision, verify_model_identity


def test_local_revision_tracks_model_and_tokenizer_bytes(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "REFERENCE_REVISION").write_text("old-remote", encoding="utf-8")
    first = verify_model_identity(tmp_path)
    revision = local_model_revision(first)
    assert revision.startswith("local-sha256:")
    assert "old-remote" not in revision
    (tmp_path / "added_tokens.json").write_text('{"new": 1}', encoding="utf-8")
    second = verify_model_identity(tmp_path)
    assert local_model_revision(second) != revision
    assert "added_tokens.json" in second["files"]
