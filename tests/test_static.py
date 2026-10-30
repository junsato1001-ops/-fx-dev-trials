"""Static guards: reader implementation rules, no sealed-pair codes outside vendor/, tests never read data/."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Built from reversed strings so this file itself does not contain them.
SEALED = ["".join(reversed(s)) for s in ("PBG", "DUA", "DAC", "DGS", "WRK")]
FORBIDDEN_CALLS = ["read_" + "snapshot_zip", "load_" + "inputs("]


class StaticTests(unittest.TestCase):
    def test_reader_rules(self):
        src = (ROOT / "devtrial" / "reader.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) else [])
                for n in names:
                    self.assertNotIn((n or "").split(".")[0], {"csv", "pandas", "polars"})
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, {"split", "rsplit", "splitlines", "DictReader", "read_csv"})

    def test_no_sealed_codes_outside_vendor(self):
        files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts
                 and "vendor" not in p.parts and p.suffix in {".py", ".md", ".yml", ".yaml", ".json", ".txt"}]
        self.assertTrue(files)
        for p in files:
            text = p.read_text(encoding="utf-8")
            for code in SEALED:
                self.assertNotIn(code, text, f"{code} in {p.relative_to(ROOT)}")

    def test_tests_do_not_reference_snapshot(self):
        cfg_path = (ROOT / "trials" / "DEV-USDJPY-SIMPLE-01" / "config.json").read_text()
        snap = __import__("json").loads(cfg_path)["snapshot"]["path"]
        for p in (ROOT / "tests").glob("*.py"):
            text = p.read_text(encoding="utf-8")
            self.assertNotIn(snap, text)
            for c in FORBIDDEN_CALLS:
                self.assertNotIn(c, text)

    def test_no_authorization_file_committed(self):
        self.assertFalse((ROOT / "trials" / "DEV-USDJPY-SIMPLE-01" / "AUTHORIZATION.json").exists())

    def test_runtime_pinned(self):
        import re
        wf = (ROOT / ".github" / "workflows" / "dev-trial.yml").read_text()
        uses = re.findall(r"uses:\s*(\S+)", wf)
        self.assertTrue(uses)
        for u in uses:
            self.assertRegex(u, r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}$", u)
        self.assertRegex(wf, r'python-version:\s*"3\.13\.17"')
        self.assertIn("--require-hashes", wf)
        req = (ROOT / "requirements.txt").read_text()
        self.assertRegex(req, r"numpy==2\.5\.3 \\\n\s+--hash=sha256:[0-9a-f]{64}\s*$")

    def test_workflow_dispatch_only(self):
        wf = (ROOT / ".github" / "workflows" / "dev-trial.yml").read_text()
        self.assertIn("workflow_dispatch", wf)
        for bad in ("schedule:", "push:", "pull_request:", "cron"):
            self.assertNotIn(bad, wf)


if __name__ == "__main__":
    unittest.main()
