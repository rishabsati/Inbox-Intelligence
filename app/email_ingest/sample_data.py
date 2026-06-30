"""
Loads the fake/sample inbox used when DATA_SOURCE=sample. This is what
keeps the demo working even with no Gmail OAuth setup at all -- useful
for first-run testing and for a public demo where you don't want to
expose a real inbox.
"""
import json
import os

_SAMPLE_DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_emails.json")
)


def get_sample_emails():
    with open(_SAMPLE_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
