import datetime as dt
import unittest

from devtrial import reader

MARK = "987654.3219"  # appears only in non-allowlisted columns


def csv_bytes(rows, header="Date,XAA,JPY,XBB,USD,XCC,", trailing=","):
    lines = [header]
    for r in rows:
        lines.append(",".join(r) + trailing)
    return ("\n".join(lines) + "\n").encode()


class ReaderTests(unittest.TestCase):
    def test_slices_only_allowlisted(self):
        b = csv_bytes([["2020-01-03", MARK, "120.5", MARK, "1.1", MARK],
                       ["2020-01-02", MARK, "N/A", MARK, "1.0", MARK]])
        rows = reader.read_allowlisted(b)
        self.assertEqual(rows, [(dt.date(2020, 1, 2), None, 1.0), (dt.date(2020, 1, 3), 120.5, 1.1)])
        self.assertNotIn(float(MARK), [v for r in rows for v in r[1:]])

    def test_header_positions_any_order(self):
        b = csv_bytes([["1.1", MARK, "2020-01-02", "120.5"]], header="USD,XAA,Date,JPY", trailing="")
        self.assertEqual(reader.read_allowlisted(b), [(dt.date(2020, 1, 2), 120.5, 1.1)])

    def test_crlf(self):
        b = csv_bytes([["2020-01-02", MARK, "120.5", MARK, "1.1", MARK]]).replace(b"\n", b"\r\n")
        self.assertEqual(reader.read_allowlisted(b)[0][1], 120.5)

    def test_quote_fails_closed(self):
        b = csv_bytes([["2020-01-02", '"' + MARK + '"', "120.5", MARK, "1.1", MARK]])
        with self.assertRaises(reader.ReaderError) as cm:
            reader.read_allowlisted(b)
        self.assertNotIn(MARK, str(cm.exception))

    def test_field_count_mismatch(self):
        for bad in (["2020-01-02", MARK, "120.5", MARK, "1.1"],
                    ["2020-01-02", MARK, "120.5", MARK, "1.1", MARK, MARK]):
            with self.assertRaises(reader.ReaderError) as cm:
                reader.read_allowlisted(csv_bytes([bad]))
            self.assertNotIn(MARK, str(cm.exception))

    def test_duplicate_date(self):
        r = ["2020-01-02", MARK, "120.5", MARK, "1.1", MARK]
        with self.assertRaises(reader.ReaderError):
            reader.read_allowlisted(csv_bytes([r, r]))

    def test_missing_or_duplicate_header(self):
        with self.assertRaises(reader.ReaderError):
            reader.read_allowlisted(csv_bytes([], header="Date,XAA,JPY,"))
        with self.assertRaises(reader.ReaderError):
            reader.read_allowlisted(csv_bytes([], header="Date,USD,JPY,USD,"))


if __name__ == "__main__":
    unittest.main()
