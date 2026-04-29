import unittest
from unittest.mock import patch

from arm.env_wrapper import BaseEnvWrapper, RobosuiteEnvWrapper, create_env_wrapper


class EnvWrapperFactoryContractTests(unittest.TestCase):
    def test_base_contract_type_exists(self):
        self.assertTrue(issubclass(RobosuiteEnvWrapper, BaseEnvWrapper))

    def test_unsupported_backend_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Unsupported environment backend"):
            create_env_wrapper(backend="unknown")

    def test_robocasa_backend_is_resolved_through_dedicated_adapter(self):
        class DummyWrapper(BaseEnvWrapper):
            backend_name = "robocasa"

            def get_observation(self):
                return None

            def step(self, action):
                return None

            def reset(self, seed=None):
                return None

            def render(self):
                return None

            def get_camera_intrinsics(self):
                return None

            def get_camera_extrinsics(self):
                return None

            def get_object_dynamics_summary(self):
                return None

            def arm_safe_retract(self):
                return None

        with patch("arm.env_wrapper._create_robocasa_wrapper", return_value=DummyWrapper()):
            wrapper = create_env_wrapper(backend="robocasa")
        self.assertEqual(wrapper.backend_name, "robocasa")


if __name__ == "__main__":
    unittest.main()
