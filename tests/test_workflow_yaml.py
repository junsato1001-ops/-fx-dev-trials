"""Regression test for the DEV-USDJPY-SIMPLE-01 failure: the workflow file must be loadable YAML."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WF = ROOT / ".github" / "workflows" / "dev-trial.yml"
INSTALL = "python -m pip install --require-hashes --only-binary=:all: --no-deps -r requirements.txt"


class WorkflowYamlTests(unittest.TestCase):
    def test_plain_run_scalars_have_no_colon_space(self):
        for n, line in enumerate(WF.read_text().splitlines(), 1):
            m = re.match(r"^\s*(?:-\s+)?run:\s+(.*)$", line)
            if not m or m.group(1) in ("|", ">", "|-", ">-"):
                continue
            v = m.group(1)
            if ": " in v or v.endswith(":"):
                self.assertTrue(v[0] in "\"'" and v[-1] == v[0], f"line {n}: unquoted ': ' in run scalar")

    def test_full_parse(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed (CI); structural check above still applies")
        d = yaml.safe_load(WF.read_text())
        on = d.get("on", d.get(True))
        self.assertEqual(set(on), {"workflow_dispatch"})
        steps = d["jobs"]["trial"]["steps"]
        runs = [s.get("run") for s in steps if "run" in s]
        self.assertIn(INSTALL, runs)


if __name__ == "__main__":
    unittest.main()
