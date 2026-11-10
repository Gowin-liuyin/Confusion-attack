"""SDA input and decode helpers over frozen, public upload records.

This module never reads scoring truth or client-side state. The public caller
supplies actual uploaded IDs, text, or vectors and a public candidate table or
text tokenizer when needed. Formal eligibility is decided elsewhere.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
import json
import re
import numpy as np

from ..schemes.adapters import Upload, UploadSurface
from .sda_stream import PublicVectorSymbols, decode_sda


def _observation_sha256(observations: dict) -> str:
    """Bind the processed input, frontend settings and original upload digest."""
    payload = {key: value for key, value in observations.items()
               if key != "processed_observations_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _public_ids(value, *, label: str) -> list[int]:
    ids = np.asarray(value)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.issubdtype(ids.dtype, np.bool_):
        raise ValueError(f"{label} must be a one-dimensional integer sequence")
    if np.any(ids < 0) or (np.issubdtype(ids.dtype, np.unsignedinteger)
                           and ids.size and int(ids.max()) > np.iinfo(np.int64).max):
        raise ValueError(f"{label} contains IDs outside the non-negative int64 range")
    return [int(value) for value in ids]


def _stable_full_table_knn(queries: np.ndarray, table: np.ndarray, *, k: int,
                           metric: str, query_chunk: int) -> np.ndarray:
    """Full-table distances in float64, with bounded memory and stable ties."""
    result = np.empty((len(queries), k), dtype=np.int64)
    table_chunk = 2048
    for start in range(0, len(queries), query_chunk):
        for offset, query in enumerate(queries[start:start + query_chunk]):
            query64 = query.astype(np.float64)
            query_norm = max(float(np.linalg.norm(query64)), 1e-12) if metric == "cosine" else None
            best_scores = np.empty(0, dtype=np.float64)
            best_ids = np.empty(0, dtype=np.int64)
            for table_start in range(0, len(table), table_chunk):
                block = table[table_start:table_start + table_chunk].astype(np.float64)
                if metric == "l2":
                    diff = block - query64
                    scores = np.einsum("ij,ij->i", diff, diff)
                else:
                    norms = np.maximum(np.linalg.norm(block, axis=1), 1e-12)
                    scores = -((block / norms[:, None]) @ (query64 / query_norm))
                ids = np.arange(table_start, table_start + len(block), dtype=np.int64)
                scores = np.concatenate((best_scores, scores))
                ids = np.concatenate((best_ids, ids))
                order = np.lexsort((ids, scores))[:k]
                best_scores, best_ids = scores[order], ids[order]
            if not np.isfinite(best_scores).all():
                raise ValueError("candidate distance computation is numerically unsafe")
            result[start + offset] = best_ids
    return result


def prepare_sda_observations(
    uploads: Sequence[Upload], *, vector_method: str = "rounded",
    distance_threshold: float | None = None, candidate_table: np.ndarray | None = None,
    candidate_k: int = 1, metric: str = "l2", query_chunk: int = 64,
    text_tokenizer: Callable[[str], Sequence[int]] | None = None,
) -> dict:
    """Convert existing public uploads to cipher symbols, without label access.

    ``nearest_table`` is full-table KNN with real top-k candidates; only its
    first candidate forms the symbol stream, so later SDA decode remains a
    single-path engineering composition. Text tokenization acts on uploaded
    text; it does not establish alignment to an undisclosed original prompt.
    """
    if vector_method not in {"rounded", "nearest_l2", "nearest_table"}:
        raise ValueError("unsupported SDA vector method")
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k <= 0:
        raise ValueError("candidate_k must be a positive integer")
    if isinstance(query_chunk, bool) or not isinstance(query_chunk, int) or query_chunk <= 0:
        raise ValueError("query_chunk must be a positive integer")
    if metric not in {"l2", "cosine"}:
        raise ValueError("metric must be l2 or cosine")
    uploads = list(uploads)
    if any(isinstance(u.request_id, (bool, np.bool_)) or
           not isinstance(u.request_id, (int, np.integer)) for u in uploads):
        raise ValueError("upload request IDs must be integers")
    if len({int(u.request_id) for u in uploads}) != len(uploads):
        raise ValueError("upload request IDs must be unique")
    if any(not isinstance(u.surface, UploadSurface) for u in uploads):
        raise ValueError("upload surface must be an UploadSurface value")
    if any(u.group_id is not None and not isinstance(u.group_id, str) for u in uploads):
        raise ValueError("public group IDs must be strings or None")
    surfaces = {u.surface for u in uploads}
    if len(surfaces) > 1:
        raise ValueError("one SDA run cannot mix upload surfaces")
    vector_symbols = (PublicVectorSymbols(vector_method, distance_threshold)
                      if surfaces == {UploadSurface.VECTORS} and vector_method != "nearest_table" else None)
    table = None
    if vector_method == "nearest_table" and surfaces == {UploadSurface.VECTORS}:
        if candidate_table is None:
            raise ValueError("nearest_table requires a public candidate table")
        table = np.asarray(candidate_table, dtype=np.float32)
        if table.ndim != 2 or not table.shape[0] or not table.shape[1] or not np.isfinite(table).all():
            raise ValueError("candidate table must be a finite non-empty matrix")
        if candidate_k > table.shape[0]:
            raise ValueError("candidate_k exceeds public candidate table size")
    rows = []
    digest = hashlib.sha256()
    for upload in uploads:
        if (isinstance(upload.seq_len, (bool, np.bool_)) or not isinstance(upload.seq_len, (int, np.integer)) or
                upload.seq_len <= 0 or isinstance(upload.full_len, (bool, np.bool_)) or
                not isinstance(upload.full_len, (int, np.integer)) or upload.full_len < 0 or
                (upload.full_len and upload.full_len < upload.seq_len) or not isinstance(upload.truncated, bool)):
            raise ValueError("upload length and truncation metadata are invalid")
        header = json.dumps({"surface": upload.surface.value, "request_id": int(upload.request_id),
                             "group_id": upload.group_id, "seq_len": int(upload.seq_len),
                             "full_len": int(upload.full_len), "truncated": upload.truncated},
                            sort_keys=True, separators=(",", ":")).encode("utf-8")
        if upload.surface == UploadSurface.TEXT:
            payload = str(upload.payload).encode("utf-8")
        else:
            array = np.ascontiguousarray(upload.payload)
            payload = str(array.dtype).encode() + json.dumps(array.shape).encode() + array.tobytes()
        for part in (header, payload):
            digest.update(len(part).to_bytes(8, "little"))
            digest.update(part)
        if upload.truncated or (upload.full_len and upload.full_len > upload.seq_len):
            raise ValueError("truncated upload cannot be treated as a complete SDA observation")
        candidates = None
        if upload.surface == UploadSurface.IDS:
            symbols = _public_ids(upload.payload, label="uploaded IDs")
            if len(symbols) != upload.seq_len:
                raise ValueError("uploaded ID length disagrees with public metadata")
        elif upload.surface == UploadSurface.TEXT:
            if text_tokenizer is None or not isinstance(upload.payload, str):
                raise ValueError("text uploads require public text and an explicit tokenizer")
            symbols = _public_ids(text_tokenizer(upload.payload), label="tokenized uploaded text")
        elif upload.surface == UploadSurface.VECTORS:
            vectors = np.asarray(upload.payload, dtype=np.float32)
            if vectors.ndim != 2 or len(vectors) != upload.seq_len or not vectors.shape[1] or not np.isfinite(vectors).all():
                raise ValueError("uploaded vectors must be finite and match the public sequence length")
            if vector_method == "nearest_table":
                if vectors.shape[1] != table.shape[1]:
                    raise ValueError("candidate table and upload dimensions do not match")
                candidates = _stable_full_table_knn(vectors, table, k=candidate_k, metric=metric,
                                                    query_chunk=query_chunk).tolist()
                symbols = [int(row[0]) for row in candidates]
            else:
                symbols = vector_symbols.encode(vectors, group_id=upload.group_id,
                                                request_id=upload.request_id)
        else:
            raise ValueError(f"unsupported upload surface {upload.surface!r}")
        if not symbols:
            raise ValueError("empty SDA observations cannot be decoded")
        rows.append({"request_id": int(upload.request_id), "group_id": upload.group_id,
                     "surface": upload.surface.value, "symbols": symbols,
                     "candidates": candidates, "source_length": int(upload.seq_len)})
    surface = next(iter(surfaces)).value if surfaces else None
    result = {"surface": surface, "rows": rows, "public_uploads_sha256": digest.hexdigest(),
            "frontend": {"vector_method": vector_method if surface == "vectors" else None,
                         "distance_threshold": distance_threshold if vector_method == "nearest_l2" and surface == "vectors" else None,
                         "candidate_k": candidate_k if vector_method == "nearest_table" and surface == "vectors" else None,
                         "candidate_table_sha256": (hashlib.sha256(json.dumps(table.shape).encode() + table.tobytes()).hexdigest()
                                                    if table is not None else None),
                         "candidate_table_shape": list(table.shape) if table is not None else None,
                         "metric": metric if vector_method == "nearest_table" and surface == "vectors" else None,
                         "query_chunk": query_chunk if table is not None else None,
                         "text_alignment": "unverified" if surface == "text" else None,
                         "status": "engineering_variant_unvalidated"}}
    result["processed_observations_sha256"] = _observation_sha256(result)
    return result


def decode_lm_key_observations(observations: dict, plaintext_alphabet: Sequence[int],
                               next_log_probs: Callable, *, beam_width: int, seed: int,
                               homophonic: bool = False, group_joint: bool = False) -> dict:
    """Search requests independently or share a key within each public group."""
    from ..attacks.sda.lm_key_search import lm_key_beam_search
    if observations.get("processed_observations_sha256") != _observation_sha256(observations):
        raise ValueError("processed SDA observations do not match their fingerprint")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(group_joint, bool):
        raise ValueError("group_joint must be a boolean")
    if group_joint:
        grouped = {}
        for row in observations["rows"]:
            label = ("group", row["group_id"]) if row["group_id"] is not None else ("request", row["request_id"])
            grouped.setdefault(label, []).append(row)
        groups, by_request = [], {}
        for label, members in grouped.items():
            material = json.dumps([seed, *label], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            group_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2**63
            lengths = [len(row["symbols"]) for row in members]
            cipher = [symbol for row in members for symbol in row["symbols"]]
            decoded = lm_key_beam_search(cipher, plaintext_alphabet, next_log_probs,
                                         beam_width=beam_width, seed=group_seed, homophonic=homophonic,
                                         sequence_lengths=lengths)
            start = 0
            for row, length in zip(members, lengths):
                by_request[row["request_id"]] = {"request_id": row["request_id"], "group_id": row["group_id"],
                                                  "tokens": decoded["tokens"][start:start + length]}
                start += length
            groups.append({"group": list(label), "request_ids": [row["request_id"] for row in members],
                           "decode": decoded})
        return {"mode": "lm_key_search_global_sample_probe", "rows": [by_request[row["request_id"]] for row in observations["rows"]],
                "groups": groups, "seed": seed, "group_joint": True, "frontend": observations["frontend"],
                "public_uploads_sha256": observations["public_uploads_sha256"],
                "processed_observations_sha256": observations["processed_observations_sha256"]}
    rows = []
    for row in observations["rows"]:
        material = f"{seed}:{row['request_id']}".encode("ascii")
        request_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2**63
        decoded = lm_key_beam_search(row["symbols"], plaintext_alphabet, next_log_probs,
                                     beam_width=beam_width, seed=request_seed, homophonic=homophonic)
        rows.append({"request_id": row["request_id"], "group_id": row["group_id"],
                     "tokens": decoded["tokens"], "decode": decoded})
    return {"mode": "lm_key_search_global_sample_probe", "rows": rows, "seed": seed,
            "group_joint": False, "frontend": observations["frontend"],
            "public_uploads_sha256": observations["public_uploads_sha256"],
            "processed_observations_sha256": observations["processed_observations_sha256"]}


def decode_sda_observations(bundle: dict, observations: dict, *, beam_width: int = 10,
                            group_joint: bool = False, visible_plaintext: bool = False,
                            use_cache: bool = False, decode_mode: str = "raw") -> dict:
    """Decode only public observations; no truth or scoring interface is accepted."""
    if observations.get("processed_observations_sha256") != _observation_sha256(observations):
        raise ValueError("processed SDA observations do not match their fingerprint")
    if decode_mode not in {"raw", "constrained"}:
        raise ValueError("decode mode must be raw or constrained")
    if visible_plaintext and decode_mode != "raw":
        raise ValueError("decode_mode is not applicable to visible plaintext")
    rows = observations["rows"]
    if visible_plaintext:
        if observations["surface"] != "ids":
            raise ValueError("direct plaintext requires a verified plaintext ID surface")
        return {"mode": "visible_plaintext", "rows": [
            {"request_id": row["request_id"], "tokens": list(row["symbols"])} for row in rows],
            "frontend": observations["frontend"],
            "public_uploads_sha256": observations["public_uploads_sha256"],
            "processed_observations_sha256": observations["processed_observations_sha256"]}
    decoded = [{"request_id": row["request_id"],
                "decode": decode_sda(bundle, row["symbols"], beam_width=beam_width, use_cache=use_cache,
                                     mode=decode_mode)}
               for row in rows]
    groups = []
    if group_joint:
        from .sda_group import decode_sda_group
        if bundle.get("length_mode", "fixed") != "fixed":
            raise ValueError("group joint decoding requires fixed length")
        grouped = {}
        for index, row in enumerate(rows):
            label = ("group", row["group_id"]) if row["group_id"] is not None else ("request", row["request_id"])
            grouped.setdefault(label, []).append(index)
        for label, indices in grouped.items():
            result = decode_sda_group(bundle, [rows[index]["symbols"] for index in indices],
                                      beam_width=beam_width, use_cache=use_cache)
            group = {"group": list(label), "request_ids": [rows[i]["request_id"] for i in indices],
                     "decode": result}
            if result["status"] == "ok":
                pos, parts = 0, []
                for index in indices:
                    length = len(rows[index]["symbols"])
                    parts.append({"request_id": rows[index]["request_id"],
                                  "tokens": result["tokens"][pos:pos + length]})
                    pos += length
                group["rows"] = parts
            groups.append(group)
    return {"mode": "neural", "decode_mode": decode_mode, "rows": decoded, "group_joint": groups,
            "frontend": observations["frontend"],
            "public_uploads_sha256": observations["public_uploads_sha256"],
            "processed_observations_sha256": observations["processed_observations_sha256"]}


def decode_candidate_observations(observations: dict, score_prefixes: Callable, *, beam_width: int,
                                  scorer_sha256: str, candidate_id_space: str,
                                  batch_size: int = 128) -> dict:
    """Score all retained public candidates; never accepts private scoring truth."""
    from ..attacks.sda.candidate_search import candidate_prefix_beam_search
    if observations.get("processed_observations_sha256") != _observation_sha256(observations):
        raise ValueError("processed SDA observations do not match their fingerprint")
    if observations.get("surface") != "vectors" or observations.get("frontend", {}).get("vector_method") != "nearest_table":
        raise ValueError("candidate search requires public nearest-table vector candidates")
    if candidate_id_space != "public_plaintext_token_ids":
        raise ValueError("candidate table rows must explicitly be declared public plaintext token IDs")
    if not isinstance(scorer_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", scorer_sha256):
        raise ValueError("scorer_sha256 must be a SHA-256 hex digest")
    rows = observations["rows"]
    ids = [row["request_id"] for row in rows]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("candidate request IDs must be unique integers")
    output = []
    for row in rows:
        candidates = row["candidates"]
        if candidates is None or len(candidates) != len(row["symbols"]):
            raise ValueError("candidate positions must match public symbols")
        if any(not choices or choices[0] != symbol for choices, symbol in zip(candidates, row["symbols"])):
            raise ValueError("first candidate must match the public symbol")
        result = candidate_prefix_beam_search(candidates, score_prefixes, beam_width=beam_width, batch_size=batch_size)
        output.append({"request_id": row["request_id"], "decode": result})
    return {"mode": "neural", "variant": "candidate_prefix_scoring_probe", "rows": output,
            "scorer_sha256": scorer_sha256, "scorer_provenance": "caller_supplied_unverified",
            "candidate_id_space": candidate_id_space, "candidate_table_semantics": "caller_supplied_unverified",
            "frontend": observations["frontend"], "public_uploads_sha256": observations["public_uploads_sha256"],
            "processed_observations_sha256": observations["processed_observations_sha256"],
            "formal_status": "not_assessed"}


def decoded_sda_tokens(decoded: dict) -> dict[int, list[int]]:
    """Extract public predictions by request ID, without accepting scoring truth."""
    if decoded.get("mode") not in {"neural", "visible_plaintext", "lm_key_search_global_sample_probe"}:
        raise ValueError("unsupported public SDA decoding result")
    result = {}
    for row in decoded["rows"]:
        request_id = row["request_id"]
        if isinstance(request_id, (bool, np.bool_)) or not isinstance(request_id, (int, np.integer)):
            raise ValueError("SDA output request ID must be an integer")
        request_id = int(request_id)
        if request_id in result:
            raise ValueError("duplicate SDA output request ID")
        tokens = row["decode"]["tokens"] if decoded["mode"] == "neural" else row["tokens"]
        result[request_id] = _public_ids(tokens, label="decoded token IDs") if len(tokens) else []
    return result


def decoded_sda_group_tokens(decoded: dict) -> dict[int, list[int]]:
    """Extract successful joint predictions; infeasible groups remain missing."""
    if decoded.get("mode") != "neural" or not isinstance(decoded.get("group_joint"), list) or not decoded["group_joint"]:
        raise ValueError("joint neural predictions are not present")
    request_ids = set(decoded_sda_tokens(decoded))
    visited, predictions = set(), {}
    for group in decoded["group_joint"]:
        members = group.get("request_ids")
        if (not isinstance(members, list) or not members or
                any(isinstance(x, bool) or not isinstance(x, int) for x in members) or
                len(set(members)) != len(members) or visited.intersection(members)):
            raise ValueError("joint prediction groups must have distinct request IDs")
        visited.update(members)
        result = group.get("decode", {})
        if result.get("status") == "infeasible":
            if "rows" in group:
                raise ValueError("infeasible joint group cannot contain predictions")
            continue
        if result.get("status") != "ok" or not isinstance(group.get("rows"), list):
            raise ValueError("invalid joint group result")
        if [row.get("request_id") for row in group["rows"]] != members:
            raise ValueError("joint prediction rows do not match group members")
        combined = []
        for row in group["rows"]:
            tokens = _public_ids(row["tokens"], label="joint decoded token IDs")
            if not tokens:
                raise ValueError("successful joint predictions cannot be empty")
            predictions[row["request_id"]] = tokens
            combined.extend(tokens)
        if combined != result.get("tokens"):
            raise ValueError("joint prediction rows do not match group decoding")
    if visited != request_ids:
        raise ValueError("joint prediction groups must cover all request IDs")
    return predictions


def decoded_sda_text(decoded: dict, decode_tokens: Callable[[Sequence[int]], str]) -> dict[int, str]:
    """Convert attacker outputs to text without receiving reference strings."""
    result = {}
    for request_id, tokens in decoded_sda_tokens(decoded).items():
        text = decode_tokens(tokens) if tokens else ""
        if not isinstance(text, str):
            raise TypeError("public decoder must return a string")
        result[request_id] = text
    return result
