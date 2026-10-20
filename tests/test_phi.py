import pytest

from prior_auth.phi import find_phi, redact


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Patient: Daniel Okafor, seen today.", "NAME"),
        ("Patient Kevin Tran, DOB 02/02/1983.", "DATE"),
        ("MRN 00482913 on file.", "MRN"),
        ("Call (415) 555-0142 for results.", "PHONE"),
        ("Call 212-555-0187 for results.", "PHONE"),
        ("Contact lucas.martin@example.com.", "EMAIL"),
        ("SSN 123-45-6789.", "SSN"),
        ("member ID ZX448812093 verified.", "MEMBER_ID"),
        ("Lives at 123 Market Street.", "ADDRESS"),
        ("A 93-year-old with hip pain.", "AGE_OVER_89"),
    ],
)
def test_each_identifier_type_is_redacted(text: str, label: str) -> None:
    result = redact(text)
    assert f"[{label}]" in result.text
    assert result.counts[label] >= 1


def test_detected_name_is_removed_everywhere() -> None:
    result = redact("Patient: Daniel Okafor. Later, Okafor reported less pain.")
    assert "Okafor" not in result.text
    assert "Daniel" not in result.text
    assert result.text.startswith("Patient: [NAME].")


def test_clinical_content_is_kept() -> None:
    text = (
        "52-year-old with 9 weeks of pain. Completed 6 weeks of PT, naproxen 500 mg twice daily. "
        "Patient reports 7/10 pain. FEV1 38 percent predicted. Epworth 14."
    )
    assert redact(text).text == text
    assert find_phi(text) == []


def test_find_phi_flags_leftover_identifiers() -> None:
    assert find_phi("Reach the member at 212-555-0187.") == ["PHONE"]
    assert find_phi("Patient: [NAME], DOB [DATE].") == []
