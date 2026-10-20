"""PHI redaction that runs before any text reaches the LLM, the logs or the database.

This is a rule-based baseline covering the identifiers that appear in referral notes (names after a
label, dates, phone numbers, emails, record and member numbers, street addresses, ages over 89).
A production system would add a clinical NER model (for example Microsoft Presidio with a medical
recognizer) and keep these rules as a high-precision first pass.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

_NAME = r"[A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+){1,2}"


@dataclass(frozen=True)
class Detector:
    label: str
    pattern: re.Pattern[str]
    group: int = 0  # which capture group is the identifier (0 = whole match)


DETECTORS: list[Detector] = [
    Detector("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    Detector("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    Detector("PHONE", re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}\b")),
    Detector(
        "MRN", re.compile(r"\b(?:MRN|medical record (?:number|no\.?))\s*[:#]?\s*([A-Z0-9-]{5,})\b", re.I), 1
    ),
    Detector(
        "MEMBER_ID",
        re.compile(r"\b(?:member|subscriber|policy)\s+(?:id|number|no\.?)\s*[:#]?\s*([A-Z0-9-]{6,})\b", re.I),
        1,
    ),
    Detector("DATE", re.compile(r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b")),
    Detector(
        "ADDRESS",
        re.compile(
            r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct)\b\.?"
        ),
    ),
    Detector("NAME", re.compile(rf"\b(?:Patient|Pt\.?|Name|Member)\s*[:-]?\s+({_NAME})"), 1),
    Detector("NAME", re.compile(r"\b(?:Mr|Mrs|Ms|Dr)\.?\s+([A-Z][a-z'-]+)"), 1),
    Detector("AGE_OVER_89", re.compile(r"\b(?:9\d|1[0-1]\d)[- ]year[- ]old\b", re.I)),
]

# Words that follow "Patient" without being a name ("Patient Reports ..." in title case notes).
_PLACEHOLDER = re.compile(r"\[([A-Z_0-9]+)\]")
_NOT_NAMES = {"Reports", "Denies", "States", "Presents", "Is", "Has", "Was", "Will", "Education", "Seen"}


@dataclass
class RedactionResult:
    text: str
    counts: Counter[str] = field(default_factory=Counter)


def _spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for det in DETECTORS:
        for m in det.pattern.finditer(text):
            value = m.group(det.group)
            if det.label == "NAME" and value.split()[0] in _NOT_NAMES:
                continue
            spans.append((m.start(det.group), m.end(det.group), det.label))
    return spans


def redact(text: str) -> RedactionResult:
    """Replace identifiers with [LABEL] placeholders. A detected name is removed everywhere in the text."""
    spans = _spans(text)

    # A name found once ("Patient: Daniel Okafor") is redacted wherever it appears later ("Okafor reports").
    name_tokens = {
        tok for s, e, label in spans if label == "NAME" for tok in text[s:e].split() if len(tok) > 2
    }
    for token in name_tokens:
        spans += [(m.start(), m.end(), "NAME") for m in re.finditer(rf"\b{re.escape(token)}\b", text)]

    # Merge overlaps, keeping the earliest and longest span.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    merged: list[tuple[int, int, str]] = []
    for span in spans:
        if merged and span[0] < merged[-1][1]:
            continue
        merged.append(span)

    out, cursor = [], 0
    for start, end, label in merged:
        out.extend([text[cursor:start], f"[{label}]"])
        cursor = end
    out.append(text[cursor:])
    # Adjacent name tokens ("[NAME] [NAME]") collapse into one placeholder.
    redacted = re.sub(r"\[NAME\](?:\s+\[NAME\])+", "[NAME]", "".join(out))
    counts = Counter(m.group(1) for m in _PLACEHOLDER.finditer(redacted))
    counts.subtract(Counter(m.group(1) for m in _PLACEHOLDER.finditer(text)))
    return RedactionResult(redacted, +counts)


def find_phi(text: str) -> list[str]:
    """Labels of identifiers still present in `text` (used by the output guard)."""
    return sorted({label for _, _, label in _spans(text)})
