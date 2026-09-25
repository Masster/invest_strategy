"""TLS для T-Invest: проверка сертификата никогда не отключается."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from moexlab.broker.tinvest import ca_bundle_path  # noqa: E402


def test_ca_bundle_never_disables_verification(tmp_path, monkeypatch):
    monkeypatch.delenv("TINVEST_CA_BUNDLE", raising=False)
    monkeypatch.setattr("moexlab.broker.tinvest.DEFAULT_CA", str(tmp_path / "missing.crt"))
    assert ca_bundle_path() is True
    f = tmp_path / "b.crt"
    f.write_text("x")
    monkeypatch.setenv("TINVEST_CA_BUNDLE", str(f))
    assert ca_bundle_path() == str(f)
    assert ca_bundle_path(str(tmp_path / "nope")) == str(f)


def test_pem_normalization_handles_crlf_without_trailing_newline():
    import os
    import ssl

    import pytest
    from setup_tinvest_ca import normalize
    cafile = ssl.get_default_verify_paths().cafile or "/etc/ssl/certs/ca-certificates.crt"
    if not os.path.exists(cafile):
        pytest.skip("no system CA file to take a sample certificate from")
    der_pem = open(cafile).read()
    first = der_pem[der_pem.index("-----BEGIN CERTIFICATE-----"):der_pem.index("-----END CERTIFICATE-----") + 25]
    raw = first.replace("\n", "\r\n").encode()
    out, fp = normalize(raw)
    assert out.endswith("-----END CERTIFICATE-----\n") and "\r" not in out and len(fp) == 64
