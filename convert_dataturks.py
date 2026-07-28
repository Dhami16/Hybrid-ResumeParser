"""
convert_dataturks.py — convert the DataTurks "Resume Entities for NER" dataset
(external_data/dataturks_resume_ner/{traindata,testdata}.json) into (text,
entities) examples usable by our NAME/UNIVERSITY spaCy pipeline.

Format notes verified by direct inspection (see DATA_SOURCES.md):
  - JSON-lines, one resume per line: {"content": str, "annotation": [...]}
  - Each annotation: {"label": [str, ...], "points": [{"start", "end", "text"}]}
  - "end" is INCLUSIVE — the correct slice is content[start:end+1], not
    content[start:end]. Verified empirically against the recorded "text" field.
  - Only "Name" -> NAME and "College Name" -> UNIVERSITY are extracted; every
    other label (Skills, Degree, Companies worked at, ...) is ignored, since
    those fields are already handled by regex/heuristic tiers.

Usage:
    python convert_dataturks.py
"""

import json
import pathlib

import spacy

ROOT = pathlib.Path(__file__).parent
DATA_DIR = ROOT / "external_data" / "dataturks_resume_ner"
LABEL_MAP = {"Name": "NAME", "College Name": "UNIVERSITY"}


def _iter_raw_records():
    for fname in ("traindata.json", "testdata.json"):
        path = DATA_DIR / fname
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _locate(content, start, end, text):
    """
    Return a corrected (start, end_inclusive) that actually matches `text`.

    Verified by direct inspection: 41/554 Name+College Name annotations in
    this dataset have start/end offsets that don't match their own recorded
    "text" field at all (character-offset drift in the source export, not an
    off-by-one) -- e.g. offsets pointing at "gg\n\nGNIT institute of Technol"
    when "text" says "GNIT institute of Technology ". Trust "text" over the
    offsets: verify first, relocate via exact search (local window first,
    since the drift is usually a small shift) if they disagree, and signal
    failure (None) if the text can't be found at all so the caller can drop
    it rather than train on a corrupted span.
    """
    if content[start:end + 1] == text:
        return start, end
    window_start = max(0, start - 200)
    window_end = min(len(content), end + 200)
    idx = content.find(text, window_start, window_end)
    if idx == -1:
        idx = content.find(text)  # fall back to a global search
    if idx == -1:
        return None
    return idx, idx + len(text) - 1


def _extract_entities(record, stats):
    """Returns [(start, end_exclusive, label), ...] for Name/College Name only."""
    content = record["content"]
    entities = []
    for ann in record.get("annotation") or []:
        labels = [LABEL_MAP[l] for l in ann["label"] if l in LABEL_MAP]
        if not labels:
            continue
        label = labels[0]
        for point in ann["points"]:
            start, end, text = point["start"], point["end"], point["text"]  # end is inclusive
            located = _locate(content, start, end, text)
            if located is None:
                stats["offset_unrecoverable"] += 1
                continue
            start, end = located
            if (start, end) != (point["start"], point["end"]):
                stats["offset_relocated"] += 1

            # Trim leading/trailing whitespace so entities don't include stray
            # newlines picked up by the original annotation tool.
            while start <= end and content[start].isspace():
                start += 1
            while end >= start and content[end].isspace():
                end -= 1
            if start > end:
                continue  # whitespace-only span, nothing left
            entities.append((start, end + 1, label))  # convert to exclusive end
    return entities


def convert(nlp):
    """
    Returns:
        examples: list of (text, [(start, end, label), ...]) for resumes with
                   at least one usable NAME/UNIVERSITY span after char_span
                   alignment.
        report: dict of counts for the conversion summary.
    """
    examples = []
    total_records = 0
    dropped_no_entities = 0
    dropped_alignment_failed = 0
    spans_skipped = 0
    spans_total = 0
    with_name = 0
    with_university = 0
    with_both = 0
    stats = {"offset_relocated": 0, "offset_unrecoverable": 0}

    for record in _iter_raw_records():
        total_records += 1
        text = record["content"]
        raw_entities = _extract_entities(record, stats)
        if not raw_entities:
            dropped_no_entities += 1
            continue

        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in raw_entities:
            spans_total += 1
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None or len(span) == 0:
                spans_skipped += 1
                continue
            spans.append(span)

        if not spans:
            dropped_alignment_failed += 1
            continue

        # Drop overlapping spans (keep first-seen) so doc.ents assignment can't fail.
        spans.sort(key=lambda s: s.start_char)
        kept = []
        last_end = -1
        for span in spans:
            if span.start_char >= last_end:
                kept.append(span)
                last_end = span.end_char

        labels_present = {s.label_ for s in kept}
        if "NAME" in labels_present:
            with_name += 1
        if "UNIVERSITY" in labels_present:
            with_university += 1
        if "NAME" in labels_present and "UNIVERSITY" in labels_present:
            with_both += 1

        entities = [(s.start_char, s.end_char, s.label_) for s in kept]
        examples.append((text, entities))

    report = {
        "total_raw_records": total_records,
        "offset_relocated_via_text_match": stats["offset_relocated"],
        "offset_unrecoverable_dropped": stats["offset_unrecoverable"],
        "dropped_no_name_or_university_annotation": dropped_no_entities,
        "dropped_all_spans_failed_alignment": dropped_alignment_failed,
        "spans_total": spans_total,
        "spans_skipped_alignment": spans_skipped,
        "usable_resumes": len(examples),
        "usable_with_name": with_name,
        "usable_with_university": with_university,
        "usable_with_both": with_both,
    }
    return examples, report


def main():
    nlp = spacy.blank("en")
    examples, report = convert(nlp)

    print("=" * 60)
    print("  DataTurks conversion report")
    print("=" * 60)
    for key, value in report.items():
        print(f"  {key:<42}{value}")

    out_path = ROOT / "external_data" / "dataturks_converted.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(examples, fh)
    print(f"\nSaved {len(examples)} usable examples to {out_path}")


if __name__ == "__main__":
    main()
