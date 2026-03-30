#!/usr/bin/env python3
import importlib
from importlib import metadata
import sys


REQUIRED_MODULES = [
    "deep_translator",
    "easyocr",
    "fastapi",
    "networkx",
    "numpy",
    "opencv_python_headless",
    "pandas",
    "pymorphy3",
    "spacy",
    "streamlit",
    "torch",
    "torchvision",
    "uvicorn",
]
PACKAGE_NAME_MAP = {
    "deep_translator": "deep-translator",
    "opencv_python_headless": "opencv-python-headless",
}


def import_module(module_name):
    if module_name == "opencv_python_headless":
        return importlib.import_module("cv2")
    return importlib.import_module(module_name)


def package_version(module_name, module):
    package_name = PACKAGE_NAME_MAP.get(module_name, module_name)
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return getattr(module, "__version__", "no-version")


def main():
    failures = []
    print("Python:", sys.executable)
    for module_name in REQUIRED_MODULES:
        try:
            module = import_module(module_name)
            version = package_version(module_name, module)
            print(f"[OK] {module_name}: {version}")
        except Exception as exc:  # pragma: no cover - CLI guard
            failures.append((module_name, str(exc)))
            print(f"[FAIL] {module_name}: {exc}")

    if failures:
        print("\nEnvironment check failed.")
        for module_name, error in failures:
            print(f"- {module_name}: {error}")
        raise SystemExit(1)

    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()
