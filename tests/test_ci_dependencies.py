from pathlib import Path


def test_requirements_include_fastapi_for_backend_test_collection():
    requirements = (Path(__file__).parents[1] / "requirements.txt").read_text(
        encoding="utf-8"
    )
    package_names = {
        line.split("#", 1)[0].strip().split("=", 1)[0].split(">", 1)[0].lower()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "fastapi" in package_names
    assert "httpx2" in package_names


def test_requirements_pin_transformers_to_the_supported_api_family():
    requirements = (Path(__file__).parents[1] / "requirements.txt").read_text(
        encoding="utf-8"
    )

    assert "transformers>=4.36.0,<5.0.0" in requirements
