import json
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DependencyBaselineTests(unittest.TestCase):
    def test_robocasa_rl_requirements_file_exists(self):
        requirements_path = PROJECT_ROOT / "requirements-robocasa-rl.txt"
        self.assertTrue(requirements_path.exists())
        content = requirements_path.read_text(encoding="utf-8")
        self.assertIn("separate environment from the robosuite/mink baseline", content)
        self.assertIn("numpy==2.2.5", content)
        self.assertIn("torch==2.7.1", content)
        self.assertIn("torchvision==0.22.1", content)
        self.assertIn("gymnasium==0.29.1", content)
        self.assertIn("stable-baselines3", content)
        self.assertIn("mujoco==3.3.1", content)

    def test_robocasa_rl_environment_file_exists(self):
        env_path = PROJECT_ROOT / "environment-robocasa-rl.yml"
        self.assertTrue(env_path.exists())
        content = env_path.read_text(encoding="utf-8")
        self.assertIn("python=3.10", content)
        self.assertIn("- -r requirements-robocasa-rl.txt", content)
        self.assertIn("RoboCasa/RL lives in its own env", content)
        self.assertIn("Use conda only as a lightweight Python + pip shell", content)

    def test_setup_printer_runs(self):
        result = subprocess.run(
            [sys.executable, "scripts/setup/print_robocasa_rl_setup.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("RoboCasa + RL baseline setup", result.stdout)
        self.assertIn("pip install -r requirements-robocasa-rl.txt", result.stdout)
        self.assertIn("scripts/setup/check_robocasa_rl_deps.py", result.stdout)

    def test_dependency_smoke_script_reports_json(self):
        result = subprocess.run(
            [sys.executable, "scripts/setup/check_robocasa_rl_deps.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertIn("environment_strategy", payload)
        self.assertEqual(payload["environment_strategy"], "separate_robocasa_rl_env")
        self.assertFalse(payload["shared_env_supported"])
        self.assertIn("results", payload)
        self.assertIn("missing_required", payload)
        self.assertIn("robocasa_ready", payload)
        self.assertIn("robocasa", payload["results"])
        self.assertIn("mink_numpy_conflict", payload)


if __name__ == "__main__":
    unittest.main()
