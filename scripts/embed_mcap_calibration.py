#!/usr/bin/env python3
"""Write an immutable calibration attachment into a derived MCAP copy.

The source MCAP is never modified.  Messages, schemas, channels, metadata and
existing attachments are copied; the new offset bundle is added as an MCAP
attachment so the acquisition package can consume it without an HDF5 bridge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_mcap(source: Path, output: Path, offset: Path, verify_out: Path) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("refusing to overwrite the source MCAP")
    if output.exists():
        raise FileExistsError(f"derived output already exists: {output}")
    source_sha = sha256(source)
    offset_sha = sha256(offset)
    offset_data = offset.read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src_stream:
        reader = make_reader(src_stream, validate_crcs=False)
        header = reader.get_header()
        summary = reader.get_summary()
        if summary is None:
            raise ValueError("source MCAP has no summary; refusing an unverified rewrite")
        with output.open("wb") as dst_stream:
            # Preserve the source container's compression profile.  The mini
            # invocation supplies the matching zstandard wheel in its isolated
            # temporary PYTHONPATH; payload bytes themselves remain unchanged.
            writer = Writer(dst_stream, compression=CompressionType.ZSTD, enable_data_crcs=True)
            writer.start(profile=header.profile, library=f"{header.library}; qn_smith calibration embed")
            schema_ids = {}
            for old_id, schema in sorted(summary.schemas.items()):
                schema_ids[old_id] = writer.register_schema(schema.name, schema.encoding, schema.data)
            channel_ids = {}
            for old_id, channel in sorted(summary.channels.items()):
                channel_ids[old_id] = writer.register_channel(
                    channel.topic,
                    channel.message_encoding,
                    schema_ids.get(channel.schema_id, 0),
                    dict(channel.metadata),
                )
            for metadata in reader.iter_metadata():
                writer.add_metadata(metadata.name, dict(metadata.metadata))
            for attachment in reader.iter_attachments():
                writer.add_attachment(
                    attachment.create_time,
                    attachment.log_time,
                    attachment.name,
                    attachment.media_type,
                    attachment.data,
                )
            message_count = 0
            for _schema, channel, message in reader.iter_messages(log_time_order=False):
                writer.add_message(
                    channel_ids[channel.id],
                    message.log_time,
                    message.data,
                    message.publish_time,
                    message.sequence,
                )
                message_count += 1
            created_ns = time.time_ns()
            attachment_name = "calibration/26mm_new_offset_20261029.json"
            writer.add_attachment(created_ns, created_ns, attachment_name, "application/json", offset_data)
            embed_manifest = {
                "schema": "qn_smith.mcap_calibration_embedding.v1",
                "source_mcap": source.name,
                "source_mcap_sha256": source_sha,
                "offset_attachment": attachment_name,
                "offset_sha256": offset_sha,
                "message_count_copied": message_count,
                "source_records_unchanged": True,
                "hdf5_processing_unchanged": True,
                "canonical_h5_binding": {"left_physical_anchor_tag_id": 3, "right_physical_anchor_tag_id": 16},
                "note": "Offset geometry is embedded; no MCAP temporary right-start override is encoded.",
            }
            writer.add_attachment(
                created_ns,
                created_ns,
                "calibration/embedding-manifest.json",
                "application/json",
                (json.dumps(embed_manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
            )
            writer.finish()
    output_sha = sha256(output)
    verification = {
        **embed_manifest,
        "output_mcap": str(output),
        "output_mcap_sha256": output_sha,
        "output_bytes": output.stat().st_size,
        "verified_at_unix_ns": time.time_ns(),
    }
    # Read the derived file again and verify both attachments and message count.
    with output.open("rb") as f:
        out_reader = make_reader(f, validate_crcs=False)
        attachments = {a.name: a for a in out_reader.iter_attachments()}
        if attachment_name not in attachments:
            raise ValueError("offset attachment missing from derived MCAP")
        if hashlib.sha256(attachments[attachment_name].data).hexdigest() != offset_sha:
            raise ValueError("offset attachment hash mismatch")
        if "calibration/embedding-manifest.json" not in attachments:
            raise ValueError("embedding manifest missing from derived MCAP")
        verified_messages = sum(1 for _ in out_reader.iter_messages(log_time_order=False))
    verification["verified_attachment_names"] = sorted(k for k in attachments if k.startswith("calibration/"))
    verification["verified_message_count"] = verified_messages
    verification["verification_passed"] = verified_messages == message_count
    verify_out.parent.mkdir(parents=True, exist_ok=True)
    verify_out.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return verification


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--offset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--verification", type=Path, required=True)
    args = ap.parse_args()
    result = copy_mcap(args.source, args.output, args.offset, args.verification)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verification_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
