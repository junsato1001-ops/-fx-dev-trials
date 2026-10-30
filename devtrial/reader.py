"""Allowlist reader for the ECB eurofxref-hist CSV (Freeze-prep v0.1 §3.2).

Boundary: raw bytes of non-allowlisted columns may pass through this scanner while it
locates the column delimiters. No field value other than Date, JPY and USD is ever
materialized, decoded, stored, returned or logged.

Implementation rules (enforced by tests/test_reader.py and tests/test_static.py):
- The `csv` module, `str.split(",")`/`bytes.split(b",")` and pandas are NOT used.
- The header row is decoded only to find the positions of the three allowlisted names.
  Only those three integer positions are kept; no other column name is retained.
- Each data row is scanned as bytes; only the byte ranges of the three allowlisted
  fields are sliced and decoded. Other fields are skipped by position.
- A double quote anywhere in the input stops the reader (fail-closed).
"""
from __future__ import annotations

import datetime as dt
import io
import zipfile

ALLOWLIST = ("Date", "JPY", "USD")
_COMMA = 0x2C
_QUOTE = 0x22


class ReaderError(RuntimeError):
    """Raised with a message that never contains field content."""


def _header_positions(header: bytes) -> tuple[int, int, int, int]:
    if _QUOTE in header:
        raise ReaderError("quote character in header")
    text = header.decode("utf-8-sig").rstrip("\r\n")
    # Scan names one by one; keep only the positions of the allowlisted names.
    found = {name: [] for name in ALLOWLIST}
    pos = 0
    start = 0
    n = len(text)
    while True:
        end = text.find(",", start)
        if end < 0:
            end = n
        name = text[start:end].strip()
        if name in found:
            found[name].append(pos)
        del name
        pos += 1
        if end >= n:
            break
        start = end + 1
    out = []
    for key in ALLOWLIST:
        if len(found[key]) != 1:
            raise ReaderError(f"allowlisted column {key} missing or duplicated")
        out.append(found[key][0])
    n_fields = pos
    return out[0], out[1], out[2], n_fields


def _num(b: bytes):
    s = b.decode("ascii").strip()
    if s == "" or s == "N/A":
        return None
    return float(s)


def read_allowlisted(csv_bytes: bytes):
    """Return a list of (date, jpy, usd) sorted by date. jpy/usd are float or None."""
    if not csv_bytes:
        raise ReaderError("empty input")
    nl = csv_bytes.find(b"\n")
    if nl < 0:
        raise ReaderError("no header line")
    p_date, p_jpy, p_usd, n_fields = _header_positions(csv_bytes[:nl])
    wanted = {p_date: 0, p_jpy: 1, p_usd: 2}
    last_wanted = max(wanted)

    rows = []
    seen = set()
    i = nl + 1
    L = len(csv_bytes)
    while i < L:
        j = csv_bytes.find(b"\n", i)
        if j < 0:
            j = L
        line_end = j
        if line_end > i and csv_bytes[line_end - 1] == 0x0D:
            line_end -= 1
        if line_end > i:
            if csv_bytes.find(b'"', i, line_end) >= 0:
                raise ReaderError("quote character in data row")
            slots = [None, None, None]
            field = 0
            fstart = i
            k = i
            # Walk delimiters; slice only the wanted fields.
            while True:
                nxt = csv_bytes.find(b",", k, line_end)
                fend = line_end if nxt < 0 else nxt
                slot = wanted.get(field)
                if slot is not None:
                    slots[slot] = csv_bytes[fstart:fend]
                field += 1
                if nxt < 0:
                    break
                k = nxt + 1
                fstart = k
                if field > last_wanted:
                    # Count the remaining delimiters without slicing any field.
                    field += 1 + csv_bytes.count(b",", k, line_end)
                    break
            if field != n_fields:
                raise ReaderError("field count differs from header")
            d = dt.date.fromisoformat(slots[0].decode("ascii").strip())
            if d in seen:
                raise ReaderError("duplicate date")
            seen.add(d)
            rows.append((d, _num(slots[1]), _num(slots[2])))
        i = j + 1
    rows.sort(key=lambda r: r[0])
    return rows


def read_snapshot_zip(path: str, member: str = "eurofxref-hist.csv"):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ReaderError("ZIP CRC failure")
        if z.namelist() != [member]:
            raise ReaderError("unexpected ZIP members")
        data = z.read(member)
    return read_allowlisted(data)
