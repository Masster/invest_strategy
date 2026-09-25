"""Сборка CA-bundle для запросов к T-Invest API (TLS-проверка НЕ отключается).

Сертификат *.tbank.ru выдан «Russian Trusted Root CA» (Минцифры), которого нет в стандартных наборах.
Скрипт скачивает корневой и промежуточный сертификаты с официального хоста Госуслуг (gu-st.ru),
сверяет SHA-256 отпечатки с закреплёнными значениями, нормализует PEM (исходные файлы в CRLF и без
завершающего перевода строки — простая склейка ломает PEM) и пишет bundle:
    <базовый bundle окружения (/root/.ccr/ca-bundle.crt или certifi)> + root + sub
по умолчанию в ~/.tinvest/ca-bundle.crt. Используется только клиентом T-Invest
(moexlab.broker.tinvest: параметр ca_bundle или переменная TINVEST_CA_BUNDLE).

Запуск: python3 scripts/setup_tinvest_ca.py [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import ssl
import sys
from pathlib import Path

import requests

SOURCES = {
    "russian_trusted_root_ca": "https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt",
    "russian_trusted_sub_ca": "https://gu-st.ru/content/lending/russian_trusted_sub_ca_pem.crt",
}
# SHA-256 DER-отпечатки (проверены при первой загрузке 2026-09-25, research/sources.md §1.9)
PINNED = {
    "russian_trusted_root_ca": "D26D2D0231B7C39F92CC738512BA54103519E4405D68B5BD703E9788CA8ECF31",
    "russian_trusted_sub_ca": "BBBDE2103E790B999EC62BD03CF625A5A2E7C316E10AFE6A490EEDEAD8B3FD9B",
}
DEFAULT_OUT = Path.home() / ".tinvest" / "ca-bundle.crt"


def base_bundle() -> Path:
    for p in (os.environ.get("MOEXLAB_BASE_CA"), "/root/.ccr/ca-bundle.crt"):
        if p and Path(p).exists():
            return Path(p)
    import certifi
    return Path(certifi.where())


def normalize(pem_bytes: bytes) -> tuple[str, str]:
    """Возвращает (нормализованный PEM, SHA-256 отпечаток DER)."""
    text = pem_bytes.decode("ascii").replace("\r\n", "\n").strip() + "\n"
    der = ssl.PEM_cert_to_DER_cert(text)
    return ssl.DER_cert_to_PEM_cert(der), hashlib.sha256(der).hexdigest().upper()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    parts = [base_bundle().read_text(encoding="utf-8").rstrip("\n") + "\n"]
    for name, url in SOURCES.items():
        r = requests.get(url, timeout=30)          # загрузка с обычной проверкой TLS
        r.raise_for_status()
        pem, fp = normalize(r.content)
        if fp != PINNED[name]:
            print(f"FINGERPRINT MISMATCH for {name}: {fp}", file=sys.stderr)
            return 2
        parts.append(f"# {name} (Минцифры РФ), sha256={fp}\n{pem}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    print(f"written {out}; set TINVEST_CA_BUNDLE={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
