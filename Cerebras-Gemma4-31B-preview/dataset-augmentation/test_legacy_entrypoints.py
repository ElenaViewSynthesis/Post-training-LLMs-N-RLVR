"""Ensure superseded generators cannot accidentally make paid requests."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


class LegacyEntrypointTests(unittest.TestCase):
    def test_every_legacy_paid_entrypoint_is_an_error_stub(self) -> None:
        root = Path(__file__).resolve().parent
        environment = {
            **os.environ,
            "GEMINI_API_KEY": "must-not-be-used",
            "CEREBRAS_API_KEY": "must-not-be-used",
            "TOGETHER_API_KEY": "must-not-be-used",
            "SAGEMAKER_ENDPOINT_NAME": "must-not-be-used",
        }
        cases = {
            "gemini_trajectory_worker.py": "retired",
            "gemma4_31b_agent.py": "retired",
            "run_pilot.py": "refinement_pilot.py",
        }
        for script, expected in cases.items():
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(root / script)],
                    cwd=root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
