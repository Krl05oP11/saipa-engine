"""Bloque F: an empty SAIPA_LICENSE_KEY must not skip validation unless SAIPA_DEV_MODE is on."""
import pytest

import license_validator


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # .env / .license_cache are cwd-relative
    for var in ("SAIPA_LICENSE_KEY", "SAIPA_DEV_MODE"):
        monkeypatch.delenv(var, raising=False)


def test_empty_key_refused_without_dev_mode(monkeypatch):
    monkeypatch.setenv("SAIPA_LICENSE_KEY", "")
    with pytest.raises(RuntimeError, match="license/request"):
        license_validator.validate_license()


def test_empty_key_allowed_in_dev_mode(monkeypatch):
    monkeypatch.setenv("SAIPA_DEV_MODE", "true")
    assert license_validator.validate_license() is True
