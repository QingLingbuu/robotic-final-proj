"""Print the recommended RoboCasa + RL dependency setup steps."""


def main():
    print("RoboCasa + RL baseline setup")
    print("This is a separate env from the robosuite/mink baseline.")
    print("Observed RoboCasa install drift: numpy 2.2.5, torch 2.7.1, torchvision 0.22.1, gymnasium 0.29.1.")
    print("mink 0.0.5 requires numpy<2.0.0, so do not try to unify the two envs.")
    print("1) conda env create -f environment-robocasa-rl.yml")
    print("2) conda activate robotic-robocasa-rl")
    print("3) git clone https://github.com/ARISE-Initiative/robosuite")
    print("4) pip install -e ./robosuite")
    print("5) git clone https://github.com/robocasa/robocasa")
    print("6) pip install -e ./robocasa")
    print("7) python -m robocasa.scripts.setup_macros")
    print("8) python -m robocasa.scripts.download_kitchen_assets")
    print("9) python scripts/check_robocasa_rl_deps.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
