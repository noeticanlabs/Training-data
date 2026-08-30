#!/usr/bin/env python3
"""Import a heterogeneous project archive into the governed Research Database.

This importer preserves source documents and structured research records as
their actual record classes. It does not fabricate conversations, infer truth
from historical status labels, or create training candidates automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time
from collections import Counter, defaultdict


SCHEMA_VERSION = "0.02"
PIPELINE_VERSION = "project_archive_import_v0.01"

RECORD_TYPE_MAP = {
    "thread": "note",
    "definition": "definition",
    "axiom": "claim",
    "lemma": "claim",
    "theorem": "claim",
    "equation_section": "note",
    "physics": "note",
    "grid": "note",
    "cohbit": "note",
    "model": "note",
    "experiment": "experiment",
    "verification": "result",
    "crosslink": "note",
}


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid JSON at %s line %d: %s"
                    % (path, line_no, exc)) from exc
    return records


def parse_checksum_manifest(archive_root):
    checksum_path = archive_root / "SHA256SUMS.txt"
    entries = []
    with checksum_path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line:
                continue
            try:
                digest, relpath = line.split("  ", 1)
            except ValueError as exc:
                raise ValueError(
                    "invalid checksum line %d in %s"
                    % (line_no, checksum_path)) from exc
            if len(digest) != 64 or any(
                    char not in "0123456789abcdef" for char in digest):
                raise ValueError(
                    "invalid SHA-256 at checksum line %d" % line_no)
            pure = pathlib.PurePosixPath(relpath)
            if pure.is_absolute() or ".." in pure.parts or "\\" in relpath:
                raise ValueError("unsafe checksum path: %s" % relpath)
            entries.append((digest, pure))
    if not entries:
        raise ValueError("checksum manifest is empty")
    return entries


def verify_checksum_manifest(archive_root, entries):
    resolved_root = archive_root.resolve()
    failures = []
    for expected, pure in entries:
        path = archive_root.joinpath(*pure.parts)
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            failures.append("missing: %s" % pure.as_posix())
            continue
        if resolved_root != resolved and resolved_root not in resolved.parents:
            failures.append("escaped archive root: %s" % pure.as_posix())
            continue
        if path.is_symlink() or not path.is_file():
            failures.append("not a regular file: %s" % pure.as_posix())
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append("hash mismatch: %s" % pure.as_posix())
    if failures:
        raise ValueError(
            "archive checksum verification failed: %s"
            % "; ".join(failures[:20]))


def training_policy():
    return {
        "default_admission": "pending_review",
        "do_not_remember": False,
        "contains_reasoning": False,
        "pii_status": "not_scanned",
        "license_status": "unknown",
        "notes": (
            "Imported as research evidence. Historical labels are preserved "
            "only as source metadata. Training use requires explicit review."),
    }


def source_object(source_type, source_uri, artifact_ids=None, **extra):
    value = {
        "source_type": source_type,
        "conversation_id": None,
        "message_ids": [],
        "artifact_ids": artifact_ids or [],
        "source_uri": source_uri,
    }
    value.update(extra)
    return value


def content_object(text, content_type):
    return {
        "text": text,
        "role": None,
        "content_type": content_type,
        "reasoning": None,
        "images": 0,
    }


def provenance_object(created_at, source_hash, parents, original_hash=None):
    value = {
        "created_at": created_at,
        "updated_at": None,
        "source_hash": source_hash,
        "source_hash_basis": "content.text:utf8",
        "parent_record_ids": list(dict.fromkeys(parents)),
        "model_slugs": [],
        "pipeline_version": PIPELINE_VERSION,
    }
    if original_hash:
        value["original_content_sha256"] = original_hash
    return value


def research_record(record_id, record_type, title, tags, source, content,
                    provenance, relations=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "record_type": record_type,
        "title": title,
        "project_ids": ["project-archive"],
        "tags": sorted(set(tags)),
        "source": source,
        "content": content,
        "provenance": provenance,
        "epistemic_status": "unknown",
        "relations": relations or {},
        "sensitivity": "internal",
        "training_policy": training_policy(),
    }


def title_for(record):
    for key in (
            "name", "title", "heading", "term", "object", "thread_id",
            "theorem_id", "axiom_id", "lemma_id", "record_id", "id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return str(record.get("record_type") or "Project archive record")


def validate_source_inventory(archive_root, records, text_corpus,
                              project_manifest):
    source_records = [r for r in records if r.get("record_type") == "source"]
    for record in source_records:
        relpath = record.get("path")
        digest = record.get("sha256")
        if not relpath or not digest:
            raise ValueError("source record lacks path or SHA-256")
        path = archive_root / relpath
        if not path.is_file():
            raise ValueError("source record path is missing: %s" % relpath)
        if sha256_file(path) != digest:
            raise ValueError("source record hash mismatch: %s" % relpath)
        if path.stat().st_size != record.get("bytes"):
            raise ValueError("source record size mismatch: %s" % relpath)
        expected_id = "SRC-%s" % digest[:16]
        if record.get("source_id") != expected_id:
            raise ValueError("source identity mismatch: %s" % relpath)

    text_hashes = set()
    for record in text_corpus:
        digest = record.get("sha256")
        relpath = record.get("path")
        if digest in text_hashes:
            raise ValueError("duplicate text corpus hash: %s" % digest)
        text_hashes.add(digest)
        path = archive_root / relpath
        text = path.read_text(encoding="utf-8")
        if text != record.get("content"):
            raise ValueError("text corpus mismatch: %s" % relpath)
        if sha256_file(path) != digest:
            raise ValueError("text corpus hash mismatch: %s" % relpath)

    counts = project_manifest.get("counts") or {}
    expected_source_count = counts.get("source_file_entries")
    expected_unique_count = counts.get("unique_source_hashes")
    expected_text_count = counts.get("unique_text_records")
    if expected_source_count is not None and \
            expected_source_count != len(source_records):
        raise ValueError("project manifest source count mismatch")
    if expected_unique_count is not None and expected_unique_count != len({
            r["sha256"] for r in source_records}):
        raise ValueError("project manifest unique source count mismatch")
    if expected_text_count is not None and \
            expected_text_count != len(text_corpus):
        raise ValueError("project manifest text count mismatch")


def build_records(archive_root, archive_identity, records, text_corpus,
                  project_manifest):
    generated_at = project_manifest.get("generated_at")
    root_text = canonical_json(project_manifest)
    root_id = "research:archive:%s" % archive_identity
    yield research_record(
        root_id,
        "document",
        "Project archive manifest",
        ["archive-root", "source-status:accessible-reconstruction"],
        source_object(
            "uploaded_file", "knowledge_json/00_project_manifest.json",
            ["archive:sha256:%s" % archive_identity]),
        content_object(root_text, "application/json"),
        provenance_object(generated_at, sha256_text(root_text), []),
    )

    source_records = [r for r in records if r.get("record_type") == "source"]
    by_hash = defaultdict(list)
    for record in source_records:
        by_hash[record["sha256"]].append(record)
    text_by_hash = {record["sha256"]: record for record in text_corpus}
    source_id_to_artifact = {}

    for digest in sorted(by_hash):
        occurrences = sorted(by_hash[digest], key=lambda r: r["path"])
        aliases = [r["path"] for r in occurrences]
        canonical = occurrences[0]
        artifact_record_id = "research:artifact:sha256:%s" % digest
        for occurrence in occurrences:
            source_id = occurrence["source_id"]
            existing = source_id_to_artifact.get(source_id)
            if existing and existing != artifact_record_id:
                raise ValueError("ambiguous source ID: %s" % source_id)
            source_id_to_artifact[source_id] = artifact_record_id

        text_record = text_by_hash.get(digest)
        if text_record:
            text = text_record["content"]
            content_type = "text/plain"
        else:
            text = canonical_json({
                "binary_artifact": True,
                "sha256": digest,
                "byte_size": canonical.get("bytes"),
                "paths": aliases,
            })
            content_type = "application/vnd.source-artifact-metadata+json"

        extensions = sorted({str(r.get("extension") or "") for r in occurrences})
        provenance_classes = sorted({
            str(r.get("provenance_class") or "unknown") for r in occurrences})
        tags = ["archive-source"]
        tags.extend("source-extension:%s" % ext for ext in extensions)
        tags.extend("provenance:%s" % value for value in provenance_classes)
        yield research_record(
            artifact_record_id,
            "artifact",
            pathlib.PurePosixPath(canonical["path"]).name,
            tags,
            source_object(
                "uploaded_file",
                canonical["path"],
                ["artifact:sha256:%s" % digest],
                alias_paths=aliases,
                content_sha256=digest,
                byte_size=canonical.get("bytes"),
            ),
            content_object(text, content_type),
            provenance_object(
                generated_at, sha256_text(text), [root_id], digest),
            {"derived_from": [root_id]},
        )

    seen_ids = {root_id} | set(source_id_to_artifact.values())
    for record in records:
        source_record_type = record.get("record_type")
        if source_record_type == "source":
            continue
        if source_record_type not in RECORD_TYPE_MAP:
            raise ValueError(
                "unsupported project archive record type: %r"
                % source_record_type)
        text = canonical_json(record)
        record_id = "research:archive-record:%s:%s" % (
            source_record_type, sha256_text(text))
        if record_id in seen_ids:
            raise ValueError("duplicate imported record identity: %s" % record_id)
        seen_ids.add(record_id)

        source_id = record.get("source_id")
        artifact_record_id = source_id_to_artifact.get(source_id)
        parents = [root_id]
        artifact_ids = []
        relations = {"derived_from": [root_id]}
        if artifact_record_id:
            parents.append(artifact_record_id)
            artifact_ids.append(
                "artifact:sha256:%s" % artifact_record_id.rsplit(":", 1)[-1])
            relations["derived_from"].append(artifact_record_id)
        source_uri = record.get("source_path") or (
            "knowledge_json/99_all_records.jsonl#%s"
            % sha256_text(text)[:24])
        tags = [
            "archive-record",
            "source-record-type:%s" % source_record_type,
        ]
        status = record.get("status") or record.get("record_status")
        if status is not None:
            tags.append("source-status:%s" % status)
        if source_record_type == "thread":
            tags.append("transcript-status:reconstructed-summary")

        yield research_record(
            record_id,
            RECORD_TYPE_MAP[source_record_type],
            title_for(record),
            tags,
            source_object(
                "other" if source_record_type == "thread" else "document",
                source_uri,
                artifact_ids,
                native_record_type=source_record_type,
                native_source_id=source_id,
            ),
            content_object(text, "application/json"),
            provenance_object(generated_at, sha256_text(text), parents),
            relations,
        )


def write_jsonl(path, records):
    count = 0
    type_counts = Counter()
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(
                record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
            type_counts[record["record_type"]] += 1
    return count, type_counts


def atomic_json(path, value):
    temp = path + ".tmp"
    try:
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.remove(temp)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Import a governed heterogeneous project archive.")
    ap.add_argument("--archive-root", required=True)
    ap.add_argument("--archive-zip")
    ap.add_argument("--research-dir", default="data/research_db")
    ap.add_argument("--training-dir", default="data/training_db")
    args = ap.parse_args(argv)

    archive_root = pathlib.Path(args.archive_root)
    if not archive_root.is_dir():
        raise ValueError("archive root is not a directory: %s" % archive_root)
    checksum_entries = parse_checksum_manifest(archive_root)
    verify_checksum_manifest(archive_root, checksum_entries)

    project_manifest_path = archive_root / "knowledge_json/00_project_manifest.json"
    records_path = archive_root / "knowledge_json/99_all_records.jsonl"
    text_corpus_path = archive_root / "knowledge_json/18_text_corpus_unique.json"
    project_manifest = load_json(project_manifest_path)
    source_records = load_jsonl(records_path)
    text_corpus = load_json(text_corpus_path)
    if not isinstance(text_corpus, list):
        raise ValueError("unique text corpus must be a JSON array")
    validate_source_inventory(
        archive_root, source_records, text_corpus, project_manifest)

    if args.archive_zip:
        archive_identity = sha256_file(args.archive_zip)
    else:
        archive_identity = sha256_file(archive_root / "SHA256SUMS.txt")

    os.makedirs(args.research_dir, exist_ok=True)
    os.makedirs(args.training_dir, exist_ok=True)
    research_path = os.path.join(args.research_dir, "records.jsonl")
    training_path = os.path.join(args.training_dir, "candidates.jsonl")
    research_temp = research_path + ".tmp"
    training_temp = training_path + ".tmp"

    try:
        research_count, type_counts = write_jsonl(
            research_temp,
            build_records(
                archive_root, archive_identity, source_records, text_corpus,
                project_manifest))
        with open(training_temp, "w", encoding="utf-8"):
            pass
        verify_checksum_manifest(archive_root, checksum_entries)
        if args.archive_zip and sha256_file(args.archive_zip) != archive_identity:
            raise RuntimeError("source archive ZIP changed during import")
        os.replace(research_temp, research_path)
        os.replace(training_temp, training_path)
    finally:
        for temp in (research_temp, training_temp):
            if os.path.exists(temp):
                os.remove(temp)

    generated_at = now_iso()
    common = {
        "generated_at": generated_at,
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "source_mode": "project_archive",
        "source_archive_name": archive_root.name,
        "source_archive_sha256": archive_identity,
        "source_records": str(records_path.relative_to(archive_root)),
        "source_records_sha256": sha256_file(records_path),
        "source_record_count": len(source_records),
        "automatic_admission": False,
    }
    research_manifest = dict(common)
    research_manifest.update({
        "database": "research",
        "record_count": research_count,
        "record_type_counts": dict(sorted(type_counts.items())),
        "records_sha256": sha256_file(research_path),
        "epistemic_default": "unknown",
    })
    training_manifest = dict(common)
    training_manifest.update({
        "database": "training",
        "candidate_count": 0,
        "candidates_sha256": sha256_file(training_path),
        "admission_counts": {},
        "task_counts": {},
        "reasoning_in_targets": False,
        "split_policy": "not assigned; research-only archive import",
    })
    atomic_json(os.path.join(args.research_dir, "manifest.json"),
                research_manifest)
    atomic_json(os.path.join(args.training_dir, "manifest.json"),
                training_manifest)

    print("Archive records: %d" % len(source_records))
    print("Research DB: %d records -> %s" % (research_count, research_path))
    print("Research types: %s" % dict(sorted(type_counts.items())))
    print("Training DB: 0 candidates -> %s" % training_path)
    print("Automatic admission: disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
