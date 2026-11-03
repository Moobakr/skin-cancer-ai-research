"""Small import smoke test.

Run from ai_model:
python -m src.smoke_test
"""

from .models import build_custom_cnn


def main():
    model = build_custom_cnn(image_size=64, num_classes=3)
    print(model.name)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
