"""Print the recommended RoboCasa + RL dependency setup steps."""


def main():
    print("RoboCasa + RL baseline setup")
    print("This is a separate env from the robosuite/mink baseline.")
    print("Observed RoboCasa install drift: numpy 2.2.5, torch 2.7.1, torchvision 0.22.1, gymnasium 0.29.1.")
    print("mink 0.0.5 requires numpy<2.0.0, so do not try to unify the two envs.")
    print("1) conda create -n robotic-robocasa-rl python=3.10 pip")
    print("2) conda activate robotic-robocasa-rl")
    print("3) pip install -r requirements.txt")
    print("4) copy third_party\\robocasa\\robocasa\\macros.py third_party\\robocasa\\macros.py   # run from repo root on Windows if setup_macros expects a root-level template")
    print("5) cd third_party\\robocasa")
    print("6) python -m robocasa.scripts.setup_macros")
    print("7) python -m robocasa.scripts.download_kitchen_assets")
    print("8) cd ..\\..")
    print("9) python scripts/setup/check_robocasa_rl_deps.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

