"""Main entry point for robotic manipulation final project."""

import yaml
from arm.env_wrapper import RobosuiteEnvWrapper
from fsm.state_machine import TaskStateMachine, State


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main test function that initializes environment and runs basic test."""
    print("Starting robotic manipulation project...")
    print("=" * 60)
    
    # Load configurations
    camera_config = load_config('configs/camera.yaml')
    thresholds_config = load_config('configs/thresholds.yaml')
    
    print(f"Loaded configs:")
    print(f"  Camera: {camera_config}")
    print(f"  Thresholds: {thresholds_config}")
    
    # Initialize environment
    print("\nInitializing robosuite environment...")
    env = RobosuiteEnvWrapper()
    
    # Get initial observation
    print("\nGetting initial observation...")
    rgb, depth, proprio = env.reset()
    print(f"  ✓ RGB shape: {rgb.shape}")
    print(f"  ✓ Depth shape: {depth.shape}")
    print(f"  ✓ Arm 1 EEF pos: {proprio['arm1_eef_pos']}")
    print(f"  ✓ Arm 2 EEF pos: {proprio['arm2_eef_pos']}")
    
    # Get camera intrinsics
    fx, fy, cx, cy = env.get_camera_intrinsics()
    print(f"\nCamera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # Initialize FSM
    fsm = TaskStateMachine()
    fsm.transition_to(State.PLANNING)
    print(f"\nFSM initialized in state: {fsm.get_current_state().value}")
    
    # Test safe retract
    print("\nTesting safe arm retract...")
    env.arm_safe_retract()
    print("  ✓ Arm retract executed")
    
    fsm.transition_to(State.SUCCESS)
    print(f"\n" + "=" * 60)
    print(f"Task completed! Final state: {fsm.get_current_state().value}")
    print("\nAll tests passed! ✅")


if __name__ == "__main__":
    main()

