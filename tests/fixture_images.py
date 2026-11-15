"""Test-only image fixtures.

``PARKOUR_FIXTURE`` is a frozen copy of the old sample photo, kept only for
parity and golden tests whose expected numbers depend on its exact pixels.
It is not shipped in the wheel and must never appear in user-facing material;
the sample image is ``libreyolo.SAMPLE_IMAGE`` (the Guggenheim Bilbao photo).
"""

from pathlib import Path

PARKOUR_FIXTURE = str(Path(__file__).resolve().parent / "data" / "parkour.jpg")
