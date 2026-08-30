#!/usr/bin/env python3
from datetime import timezone

from export_cblm_lang_002c import (
    DEFAULT_CUTOFF,
    _parse_time,
    _remaining_entity_hits,
    redact_text,
    render_conversation,
)


def main() -> None:
    cutoff = _parse_time(DEFAULT_CUTOFF)
    assert cutoff is not None and cutoff.tzinfo is not None
    assert cutoff.astimezone(timezone.utc).isoformat() == DEFAULT_CUTOFF

    redacted, counts = redact_text(
        "Write to test@example.com from Reykjavik or call 312-555-0199."
    )
    assert "test@example.com" not in redacted
    assert "Reykjavik" not in redacted
    assert "312-555-0199" not in redacted
    assert counts["email"] == 1
    assert counts["phone"] == 1
    assert counts["heldout_entity"] == 1
    assert _remaining_entity_hits(redacted) == []

    rendered, count = render_conversation(
        {
            "messages": [
                {"role": "user", "text": "hello", "reasoning": "must not appear"},
                {"role": "assistant", "text": "hi", "reasoning": "hidden scratch"},
            ]
        }
    )
    assert rendered == "User: hello\n\nAssistant: hi"
    assert "must not appear" not in rendered
    assert "hidden scratch" not in rendered
    assert count == 2

    print("CBLM-LANG-002C export selftest PASS")


if __name__ == "__main__":
    main()
