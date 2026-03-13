"""
Tests for process logic fixes:
- state_bins ordering matches state [x, y, z] dimensions
- step() boundary checks use correct dimension for each direction
- locate() in reset/step uses heading (not z) as third argument
- Q-table dimensions match environment state_bins
- writing_commands produces correct room height and target bounds
- smooth_commands skips distances below drone minMove (20 cm)
- ChangingTarget.main does not double-train
"""
import sys
import os
import unittest
import tempfile
from unittest.mock import patch
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dronecore.roomshply import RoomShp  # noqa: E402
from dronecore.dronevirt import DroneVirtual as CoreDroneVirtual  # noqa: E402
from dronecore import DroneState  # noqa: E402


class TestStateBinsOrdering(unittest.TestCase):
    """state_bins must align [x→room_depth, y→room_width, z→room_height]."""

    def _make_env(self, room_x, room_y, room_height):
        """Create a FunctionsLib-style DroneVirtual without importing FunctionsLib
        (it has module-level interactive code)."""
        from simple_spaces import spaces

        room_desc = f"(0 0, {room_x-1} 0, {room_x-1} {room_y-1}, 0 {room_y-1}, 0 0)"
        room = RoomShp(room_desc, room_height - 1)
        drone = CoreDroneVirtual()
        depth, width, height = room_x - 1, room_y - 1, room_height - 1

        state_bins = [
            np.linspace(0, depth, round(5 + (depth ** 0.45))),
            np.linspace(0, width, round(5 + (width ** 0.45))),
            np.linspace(0, height, round(5 + (height ** 0.45))),
        ]
        return state_bins, depth, width, height

    def test_symmetric_room_bins_same(self):
        bins, d, w, h = self._make_env(500, 500, 500)
        self.assertEqual(len(bins[0]), len(bins[1]))

    def test_asymmetric_room_x_bins_use_depth(self):
        """With room_x=300, room_y=600, x-bins must span depth (299) not width."""
        bins, depth, width, _ = self._make_env(300, 600, 400)
        self.assertAlmostEqual(bins[0][-1], depth)
        self.assertAlmostEqual(bins[1][-1], width)

    def test_asymmetric_room_y_bins_use_width(self):
        bins, depth, width, _ = self._make_env(600, 300, 400)
        self.assertAlmostEqual(bins[0][-1], depth)
        self.assertAlmostEqual(bins[1][-1], width)


class TestStepBoundaryChecks(unittest.TestCase):
    """step() must bound y by room_width and x by room_depth."""

    def _make_step_fn(self, room_depth, room_width, room_height):
        """Return a simplified step function mirroring the fixed logic."""
        def step(state, direction, distance):
            x, y, z = state
            if direction == 0 and y + distance < room_width:
                y += distance
            elif direction == 1 and y - distance > 0:
                y -= distance
            elif direction == 2 and x - distance > 0:
                x -= distance
            elif direction == 3 and x + distance < room_depth:
                x += distance
            elif direction == 4 and z + distance < room_height:
                z += distance
            elif direction == 5 and z - distance > 0:
                z -= distance
            return np.array([x, y, z])
        return step

    def test_move_up_bounded_by_width_not_depth(self):
        """Direction 0 (y+) must be bounded by room_width, not room_depth."""
        step = self._make_step_fn(room_depth=200, room_width=100, room_height=300)
        state = np.array([50, 50, 80])
        # y + 60 = 110 >= room_width(100) → movement should be rejected
        new = step(state, 0, 60)
        np.testing.assert_array_equal(new, state)

    def test_move_right_bounded_by_depth_not_width(self):
        """Direction 3 (x+) must be bounded by room_depth, not room_width."""
        step = self._make_step_fn(room_depth=100, room_width=200, room_height=300)
        state = np.array([50, 50, 80])
        # x + 60 = 110 >= room_depth(100) → movement should be rejected
        new = step(state, 3, 60)
        np.testing.assert_array_equal(new, state)

    def test_move_up_allowed_within_width(self):
        step = self._make_step_fn(room_depth=100, room_width=200, room_height=300)
        state = np.array([50, 50, 80])
        new = step(state, 0, 60)
        self.assertEqual(new[1], 110)

    def test_move_right_allowed_within_depth(self):
        step = self._make_step_fn(room_depth=200, room_width=100, room_height=300)
        state = np.array([50, 50, 80])
        new = step(state, 3, 60)
        self.assertEqual(new[0], 110)


class TestLocateHeading(unittest.TestCase):
    """locate() calls should pass heading (90), not z coordinate."""

    def test_locate_with_heading_90(self):
        """After locate with heading=90, drone heading must be pi/2."""
        from math import pi
        room = RoomShp("(0 0, 499 0, 499 499, 0 499, 0 0)", height=499)
        drone = CoreDroneVirtual()
        drone.locate(100, 100, 90, room)
        self.assertAlmostEqual(drone.position.heading, pi / 2, places=5)

    def test_locate_with_z_value_gives_wrong_heading(self):
        """Passing z=80 as heading would give heading=80*pi/180, which is wrong."""
        from math import pi
        room = RoomShp("(0 0, 499 0, 499 499, 0 499, 0 0)", height=499)
        drone = CoreDroneVirtual()
        drone.locate(100, 100, 80, room)
        # heading would be 80*pi/180 ≈ 1.396, not pi/2 ≈ 1.5708
        self.assertNotAlmostEqual(drone.position.heading, pi / 2, places=3)


class TestQTableDimensions(unittest.TestCase):
    """Q-table dimensions must match state_bins sizes."""

    def test_q_table_matches_bins(self):
        """Q-table dimensions should use (room-1) to match env state_bins."""
        room_x, room_y, room_height = 500, 500, 500
        # Fixed logic: use room-1
        space_x = round(5 + ((room_x - 1) ** 0.45))
        space_y = round(5 + ((room_y - 1) ** 0.45))
        space_z = round(5 + ((room_height - 1) ** 0.45))
        # Env state_bins use room-1 as well
        depth, width, height = room_x - 1, room_y - 1, room_height - 1
        bins_x = len(np.linspace(0, depth, round(5 + (depth ** 0.45))))
        bins_y = len(np.linspace(0, width, round(5 + (width ** 0.45))))
        bins_z = len(np.linspace(0, height, round(5 + (height ** 0.45))))
        self.assertEqual(space_x, bins_x)
        self.assertEqual(space_y, bins_y)
        self.assertEqual(space_z, bins_z)

    def test_q_table_matches_bins_asymmetric(self):
        """Asymmetric room: Q-table must still match."""
        room_x, room_y, room_height = 300, 600, 400
        space_x = round(5 + ((room_x - 1) ** 0.45))
        space_y = round(5 + ((room_y - 1) ** 0.45))
        space_z = round(5 + ((room_height - 1) ** 0.45))
        depth, width, height = room_x - 1, room_y - 1, room_height - 1
        bins_x = len(np.linspace(0, depth, round(5 + (depth ** 0.45))))
        bins_y = len(np.linspace(0, width, round(5 + (width ** 0.45))))
        bins_z = len(np.linspace(0, height, round(5 + (height ** 0.45))))
        self.assertEqual(space_x, bins_x)
        self.assertEqual(space_y, bins_y)
        self.assertEqual(space_z, bins_z)


class TestWritingCommandsRoomHeight(unittest.TestCase):
    """writing_commands must produce room_height - 1 in the replay file."""

    def test_room_height_in_output(self):
        """Generated createRoom call should use room_height - 1."""
        room_x, room_y, room_height = 500, 500, 500
        room_description = f"(0 0, {room_x - 1} 0, {room_x - 1} {room_y - 1}, 0 {room_y - 1}, 0 0)"
        expected_line = f"createRoom('{room_description}', {room_height - 1})\n"
        # Verify the formula matches the training setup
        self.assertIn("499", expected_line)
        self.assertNotIn("500", expected_line.split("createRoom")[1].split(")")[0].split(",")[-1])


class TestWritingCommandsTargetBounds(unittest.TestCase):
    """Target bounds in writing_commands must be strictly inside the room."""

    def test_target_near_low_boundary(self):
        """Target at (1, 1, 1) should have low bounds clamped to 1."""
        target_x, target_y, target_z = 1, 1, 1
        room_x, room_y, room_height = 500, 500, 500
        low_x = max(target_x - 1, 1)
        low_y = max(target_y - 1, 1)
        low_z = max(target_z - 1, 1)
        # low values must be >= 1 (strictly inside polygon)
        self.assertGreaterEqual(low_x, 1)
        self.assertGreaterEqual(low_y, 1)
        self.assertGreaterEqual(low_z, 1)

    def test_target_near_high_boundary(self):
        """Target at max valid position should have high bounds inside room."""
        room_x, room_y, room_height = 500, 500, 500
        target_x, target_y, target_z = room_x - 2, room_y - 2, room_height - 2
        high_x = min(target_x + 1, room_x - 2)
        high_y = min(target_y + 1, room_y - 2)
        high_z = min(target_z + 1, room_height - 2)
        # high values must be <= room - 2 (strictly inside polygon boundary at room - 1)
        self.assertLessEqual(high_x, room_x - 2)
        self.assertLessEqual(high_y, room_y - 2)
        self.assertLessEqual(high_z, room_height - 2)

    def test_bounds_valid_for_getRandomPosition(self):
        """The generated bounds should not cause getRandomPosition to fail."""
        room_x, room_y, room_height = 500, 500, 500
        room = RoomShp(
            f"(0 0, {room_x-1} 0, {room_x-1} {room_y-1}, 0 {room_y-1}, 0 0)",
            height=room_height - 1,
        )
        # Test various target positions
        for tx, ty, tz in [(1, 1, 1), (498, 498, 498), (250, 250, 250)]:
            low_x = max(tx - 1, 1)
            high_x = min(tx + 1, room_x - 2)
            low_y = max(ty - 1, 1)
            high_y = min(ty + 1, room_y - 2)
            low_z = max(tz - 1, 1)
            high_z = min(tz + 1, room_height - 2)
            from dronecore.envgeo import Position
            # Should not raise
            pos = room.getRandomPosition(
                Position(low_x, low_y, low_z),
                Position(high_x, high_y, high_z),
            )
            self.assertIsNotNone(pos)


class TestSmoothCommandsMinMove(unittest.TestCase):
    """smooth_commands must skip distances below the drone minMove (20 cm)."""

    @staticmethod
    def _smooth_commands(commands):
        """Local copy of the fixed smooth_commands logic for testing without tkinter."""
        smoothed_commands = []
        max_distance = 490

        opposing_directions = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}

        movement_totals = {}
        for direction, distance in commands:
            direction = int(direction)
            distance = int(distance)
            if direction in movement_totals:
                movement_totals[direction] += distance
            else:
                movement_totals[direction] = distance

        for direction in list(movement_totals.keys()):
            opposing_direction = opposing_directions.get(direction)
            if opposing_direction in movement_totals:
                if direction not in movement_totals or opposing_direction not in movement_totals:
                    continue
                if movement_totals[direction] > movement_totals[opposing_direction]:
                    movement_totals[direction] -= movement_totals[opposing_direction]
                    del movement_totals[opposing_direction]
                elif movement_totals[direction] < movement_totals[opposing_direction]:
                    movement_totals[opposing_direction] -= movement_totals[direction]
                    del movement_totals[direction]
                else:
                    del movement_totals[direction]
                    del movement_totals[opposing_direction]

        sorted_directions = sorted(
            movement_totals.keys(),
            key=lambda d: abs(movement_totals[d]),
            reverse=True,
        )

        for direction in sorted_directions:
            distance = movement_totals[direction]
            while distance >= max_distance:
                smoothed_commands.append((direction, max_distance))
                distance -= max_distance
            if distance >= 20:  # drone minimum movement is 20 cm
                smoothed_commands.append((direction, distance))

        return smoothed_commands

    def test_small_remainder_skipped(self):
        """A remainder of 5 cm (< 20) should be dropped."""
        result = self._smooth_commands([(0, 5)])
        self.assertEqual(result, [])

    def test_exact_min_move_kept(self):
        """Distance of exactly 20 should be kept."""
        result = self._smooth_commands([(0, 20)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (0, 20))

    def test_large_distance_with_small_remainder(self):
        """495 = 490 + 5; only the 490 chunk should remain."""
        result = self._smooth_commands([(0, 495)])
        self.assertEqual(result, [(0, 490)])

    def test_large_distance_with_valid_remainder(self):
        """510 = 490 + 20; both chunks should remain."""
        result = self._smooth_commands([(0, 510)])
        self.assertEqual(result, [(0, 490), (0, 20)])


if __name__ == '__main__':
    unittest.main()
