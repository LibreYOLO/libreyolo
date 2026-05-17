"""Manual check for save=True path logging."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from libreyolo import LibreYOLO, SAMPLE_IMAGE
    from libreyolo.utils.logging import setup_logging

    setup_logging()

    model_path = sys.argv[1] if len(sys.argv) > 1 else "LibreYOLOXs.pt"
    output_path = "runs/manual_save_path_test/parkour.jpg"

    model = LibreYOLO(model_path, device="cpu")
    result = model.predict(
        SAMPLE_IMAGE,
        save=True,
        output_path=output_path,
        conf=0.25,
    )

    print("returned saved_path:", result.saved_path)


if __name__ == "__main__":
    main()
