from app.guardrails.pii_guardrail import find_pii, luhn_valid, mask_pii, verhoeff_valid


def kinds(text):
    return [m.kind for m in find_pii(text)]


def aadhaar(base="23456789012"):
    """A 12-digit number with a valid Verhoeff check digit."""
    return base + str(next(d for d in range(10) if verhoeff_valid(base + str(d))))


def test_checksums():
    assert verhoeff_valid("2363") and not verhoeff_valid("2364")
    assert luhn_valid("4111111111111111") and not luhn_valid("4111111111111112")


def test_aadhaar_plain_and_formatted():
    d = aadhaar()
    assert kinds(d) == ["aadhaar"]
    assert kinds(f"id {d[:4]} {d[4:8]} {d[8:]} end") == ["aadhaar"]
    assert kinds(f"id {d[:4]}-{d[4:8]}-{d[8:]}.") == ["aadhaar"]


def test_aadhaar_needs_valid_checksum_and_prefix():
    d = aadhaar()
    assert kinds(d[:-1] + str((int(d[-1]) + 1) % 10)) == []
    assert kinds("1" + d[1:]) == []


def test_card_is_luhn_checked():
    assert kinds("4111111111111111") == ["card"]
    assert kinds("pay 4111 1111 1111 1111 now") == ["card"]
    assert kinds("4111111111111112") == []
    assert kinds("4111 1111 1111 1112") == []


def test_card_is_not_also_reported_as_aadhaar_or_phone():
    assert kinds("4111 1111 1111 1111") == ["card"]


def test_pan():
    assert kinds("PAN ABCPE1234F") == ["pan"]
    assert kinds("pan abcpe1234f") == ["pan"]
    assert kinds("ABCZE1234F") == []  # Z is not a valid holder-type letter


def test_email():
    assert kinds("write to jane.doe@example.com now") == ["email"]


def test_indian_mobile_numbers():
    for number in ("9876543210", "+91 98765 43210", "09876543210", "98765-43210"):
        assert kinds(f"call {number}") == ["phone"], number
    assert kinds("1234567890") == [] and kinds("5876543210") == []


def test_ordinary_contract_numbers_are_untouched():
    assert find_pii("Pay 45 days after 2026-03-05, amount 5,000,000, ref INV-2026-0042, clause 12.3") == []


def test_mask_counts_and_text():
    text, counts = mask_pii("Email a@b.com and a@b.com, PAN ABCPE1234F")
    assert text == "Email [REDACTED:EMAIL] and [REDACTED:EMAIL], PAN [REDACTED:PAN]"
    assert counts == {"email": 2, "pan": 1}
    assert mask_pii("nothing here") == ("nothing here", {})