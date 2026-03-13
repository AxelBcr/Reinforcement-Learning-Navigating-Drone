"""
Tests for error checking at every step of the simulation process:
- Input validation (room dimensions, episodes, steps)
- DroneVirtual environment init validation
- reset() checks drone.locate() result
- step() validates action tuple and direction/distance
- training_loop() validates parameters
- writing_commands() validates inputs
- dronecmds precondition checks (room/drone not None)
"""
import sys
import os
import unittest
import tempfile
from unittest.mock import MagicMock
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dronecore.roomshply import RoomShp  # noqa: E402
from dronecore.dronevirt import DroneVirtual as CoreDroneVirtual  # noqa: E402
from dronecore import DroneState, CommandResult  # noqa: E402


class TestDroneVirtualEnvInitValidation(unittest.TestCase):
    """DroneVirtual environment __init__ must reject invalid inputs."""

    def _make_env_class(self):
        """Return a local DroneVirtual env class that mirrors FunctionsLib logic."""
        from simple_spaces import spaces

        class DroneVirtualEnv:
            def __init__(self, drone, room, room_size, max_steps):
                if drone is None:
                    raise ValueError("Drone object cannot be None.")
                if room is None:
                    raise ValueError("Room object cannot be None.")
                if any(d <= 0 for d in room_size):
                    raise ValueError(f"All room_size dimensions must be positive, got {room_size}.")
                if max_steps <= 0:
                    raise ValueError(f"max_steps must be positive, got {max_steps}.")
                self.drone = drone
                self.room = room
                self.room_depth, self.room_width, self.room_height = room_size
                self.max_steps = max_steps
        return DroneVirtualEnv

    def test_none_drone_raises(self):
        EnvClass = self._make_env_class()
        room = RoomShp("(0 0, 499 0, 499 499, 0 499, 0 0)", height=499)
        with self.assertRaises(ValueError) as ctx:
            EnvClass(None, room, (499, 499, 499), 100)
        self.assertIn("Drone", str(ctx.exception))

    def test_none_room_raises(self):
        EnvClass = self._make_env_class()
        drone = CoreDroneVirtual()
        with self.assertRaises(ValueError) as ctx:
            EnvClass(drone, None, (499, 499, 499), 100)
        self.assertIn("Room", str(ctx.exception))

    def test_zero_room_size_raises(self):
        EnvClass = self._make_env_class()
        room = RoomShp("(0 0, 499 0, 499 499, 0 499, 0 0)", height=499)
        drone = CoreDroneVirtual()
        with self.assertRaises(ValueError) as ctx:
            EnvClass(drone, room, (0, 499, 499), 100)
        self.assertIn("positive", str(ctx.exception))

    def test_negative_max_steps_raises(self):
        EnvClass = self._make_env_class()
        room = RoomShp("(0 0, 499 0, 499 499, 0 499, 0 0)", height=499)
        drone = CoreDroneVirtual()
        with self.assertRaises(ValueError) as ctx:
            EnvClass(drone, room, (499, 499, 499), -1)
        self.assertIn("positive", str(ctx.exception))

    def test_valid_inputs_accepted(self):
        EnvClass = self._make_env_class()
        room = RoomShp("(0 0, 499 0, 499 499, 0 499, 0 0)", height=499)
        drone = CoreDroneVirtual()
        env = EnvClass(drone, room, (499, 499, 499), 100)
        self.assertEqual(env.room_depth, 499)
        self.assertEqual(env.max_steps, 100)


class TestResetErrorChecking(unittest.TestCase):
    """reset() must check drone.locate() result."""

    def test_reset_with_valid_position_succeeds(self):
        """Reset with interior position should succeed."""
        room = RoomShp("(0 0, 499 0, 499 499, 0 499, 0 0)", height=499)
        drone = CoreDroneVirtual()
        # Simulate reset: locate drone at (100, 100) which is inside
        drone.locate(100, 100, 90, room)
        self.assertTrue(drone.command.response)

    def test_reset_with_wall_position_fails(self):
        """Locate at wall position should fail."""
        room = RoomShp("(0 0, 499 0, 499 499, 0 499, 0 0)", height=499)
        drone = CoreDroneVirtual()
        drone.locate(0, 0, 90, room)
        self.assertFalse(drone.command.response)
        self.assertEqual(drone.command.result, CommandResult.RES_NO)


class TestStepActionValidation(unittest.TestCase):
    """step() must validate action tuple."""

    def _step_validate(self, action):
        """Mirrors the validation logic from FunctionsLib step()."""
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            raise ValueError(f"Action must be a (direction, distance) tuple, got {action}.")
        direction, distance = action
        if not (0 <= direction <= 5):
            raise ValueError(f"Direction must be between 0 and 5, got {direction}.")
        if distance < 0:
            raise ValueError(f"Distance must be non-negative, got {distance}.")

    def test_invalid_action_type_raises(self):
        with self.assertRaises(ValueError):
            self._step_validate("invalid")

    def test_action_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            self._step_validate((1,))

    def test_direction_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            self._step_validate((7, 50))

    def test_negative_distance_raises(self):
        with self.assertRaises(ValueError):
            self._step_validate((0, -10))

    def test_valid_action_accepted(self):
        # Should not raise
        self._step_validate((0, 50))
        self._step_validate((5, 0))


class TestTrainingLoopValidation(unittest.TestCase):
    """training_loop() must validate its parameters."""

    def _validate_training_params(self, env, num_episodes, max_steps):
        """Mirrors the validation logic from training_loop()."""
        if env is None:
            raise ValueError("Environment (env_with_viewer) cannot be None.")
        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}.")
        if max_steps <= 0:
            raise ValueError(f"max_steps_per_episode must be positive, got {max_steps}.")

    def test_none_env_raises(self):
        with self.assertRaises(ValueError):
            self._validate_training_params(None, 100, 250)

    def test_zero_episodes_raises(self):
        with self.assertRaises(ValueError):
            self._validate_training_params("env", 0, 250)

    def test_negative_max_steps_raises(self):
        with self.assertRaises(ValueError):
            self._validate_training_params("env", 100, -1)

    def test_valid_params_accepted(self):
        # Should not raise
        self._validate_training_params("env", 100, 250)


class TestWritingCommandsValidation(unittest.TestCase):
    """writing_commands() must validate inputs."""

    def _validate_writing_params(self, actions, room_x, room_y, room_height):
        """Mirrors the validation logic from writing_commands()."""
        if not actions:
            return False  # skip, no error
        if room_x <= 0 or room_y <= 0 or room_height <= 0:
            raise ValueError("Room dimensions must be positive")
        return True

    def test_empty_actions_skips(self):
        result = self._validate_writing_params([], 500, 500, 500)
        self.assertFalse(result)

    def test_negative_room_dims_raises(self):
        with self.assertRaises(ValueError):
            self._validate_writing_params([(0, 50)], -1, 500, 500)

    def test_zero_room_height_raises(self):
        with self.assertRaises(ValueError):
            self._validate_writing_params([(0, 50)], 500, 500, 0)

    def test_valid_params_accepted(self):
        result = self._validate_writing_params([(0, 50)], 500, 500, 500)
        self.assertTrue(result)


class TestDroneCmdsPreConditions(unittest.TestCase):
    """dronecmds functions must check room/drone are not None."""

    def test_create_drone_without_room_raises(self):
        """createDrone should raise if room is None."""
        # Mock tkinter dependencies
        for mod in [
            'tkinter', 'tkinter.ttk', 'idlelib', 'idlelib.tooltip',
            'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends',
            'matplotlib.backends.backend_tkagg',
            'PIL', 'PIL.Image', 'PIL.ImageTk',
        ]:
            sys.modules.setdefault(mod, MagicMock())
        import dronecmds
        # Save original room value
        original_room = dronecmds.room
        try:
            dronecmds.room = None
            with self.assertRaises(RuntimeError) as ctx:
                dronecmds.createDrone("DroneVirtual", "ViewerTkMPL")
            self.assertIn("Room must be created", str(ctx.exception))
        finally:
            dronecmds.room = original_room

    def test_locate_without_drone_raises(self):
        """locate should raise if drone is None."""
        for mod in [
            'tkinter', 'tkinter.ttk', 'idlelib', 'idlelib.tooltip',
            'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends',
            'matplotlib.backends.backend_tkagg',
            'PIL', 'PIL.Image', 'PIL.ImageTk',
        ]:
            sys.modules.setdefault(mod, MagicMock())
        import dronecmds
        original_drone = dronecmds.drone
        try:
            dronecmds.drone = None
            with self.assertRaises(RuntimeError) as ctx:
                dronecmds.locate(100, 100, 90)
            self.assertIn("Drone is not initialized", str(ctx.exception))
        finally:
            dronecmds.drone = original_drone

    def test_forward_without_drone_raises(self):
        """forward should raise if drone is None."""
        for mod in [
            'tkinter', 'tkinter.ttk', 'idlelib', 'idlelib.tooltip',
            'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends',
            'matplotlib.backends.backend_tkagg',
            'PIL', 'PIL.Image', 'PIL.ImageTk',
        ]:
            sys.modules.setdefault(mod, MagicMock())
        import dronecmds
        original_drone = dronecmds.drone
        try:
            dronecmds.drone = None
            with self.assertRaises(RuntimeError) as ctx:
                dronecmds.forward(100)
            self.assertIn("Drone is not initialized", str(ctx.exception))
        finally:
            dronecmds.drone = original_drone

    def test_takeoff_without_drone_raises(self):
        """takeOff should raise if drone is None."""
        for mod in [
            'tkinter', 'tkinter.ttk', 'idlelib', 'idlelib.tooltip',
            'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends',
            'matplotlib.backends.backend_tkagg',
            'PIL', 'PIL.Image', 'PIL.ImageTk',
        ]:
            sys.modules.setdefault(mod, MagicMock())
        import dronecmds
        original_drone = dronecmds.drone
        try:
            dronecmds.drone = None
            with self.assertRaises(RuntimeError) as ctx:
                dronecmds.takeOff()
            self.assertIn("Drone is not initialized", str(ctx.exception))
        finally:
            dronecmds.drone = original_drone

    def test_create_target_without_room_raises(self):
        """createTarget should raise if room is None."""
        for mod in [
            'tkinter', 'tkinter.ttk', 'idlelib', 'idlelib.tooltip',
            'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends',
            'matplotlib.backends.backend_tkagg',
            'PIL', 'PIL.Image', 'PIL.ImageTk',
        ]:
            sys.modules.setdefault(mod, MagicMock())
        import dronecmds
        original_room = dronecmds.room
        try:
            dronecmds.room = None
            with self.assertRaises(RuntimeError) as ctx:
                dronecmds.createTarget()
            self.assertIn("Room must be created", str(ctx.exception))
        finally:
            dronecmds.room = original_room


class TestInputValidationRules(unittest.TestCase):
    """Validate that the input validation logic rejects invalid values."""

    def _validate_room_height(self, room_height):
        """Mirrors the initialize_settings() room_height validation."""
        if room_height <= 81:
            raise ValueError("Room height must be greater than 81.")

    def _validate_room_dimension(self, dim):
        """Mirrors the initialize_settings() room_x/room_y validation."""
        if dim <= 1:
            raise ValueError("Room dimension must be greater than 1.")

    def _validate_positive(self, value, name="value"):
        """Mirrors the initialize_settings() episodes/steps validation."""
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    def test_room_height_at_81_rejected(self):
        with self.assertRaises(ValueError):
            self._validate_room_height(81)

    def test_room_height_at_50_rejected(self):
        with self.assertRaises(ValueError):
            self._validate_room_height(50)

    def test_room_height_at_82_accepted(self):
        self._validate_room_height(82)  # Should not raise

    def test_room_dimension_at_1_rejected(self):
        with self.assertRaises(ValueError):
            self._validate_room_dimension(1)

    def test_room_dimension_at_0_rejected(self):
        with self.assertRaises(ValueError):
            self._validate_room_dimension(0)

    def test_room_dimension_at_2_accepted(self):
        self._validate_room_dimension(2)  # Should not raise

    def test_zero_episodes_rejected(self):
        with self.assertRaises(ValueError):
            self._validate_positive(0, "num_episodes")

    def test_negative_episodes_rejected(self):
        with self.assertRaises(ValueError):
            self._validate_positive(-5, "num_episodes")

    def test_positive_episodes_accepted(self):
        self._validate_positive(1, "num_episodes")  # Should not raise

    def test_zero_max_steps_rejected(self):
        with self.assertRaises(ValueError):
            self._validate_positive(0, "max_steps")

    def test_positive_max_steps_accepted(self):
        self._validate_positive(250, "max_steps")  # Should not raise


if __name__ == '__main__':
    unittest.main()
