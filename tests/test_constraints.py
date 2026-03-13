"""
Tests for logical/physical constraint enforcement:
- Drone position must be strictly inside the room (not on walls)
- Target position must be strictly inside the room
- Training environment boundary checks prevent wall placement
- Takeoff altitude must not exceed room height
"""
import sys
import os
import unittest

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dronecore.roomshply import RoomShp  # noqa: E402
from dronecore.envgeo import Position  # noqa: E402
from dronecore.dronevirt import DroneVirtual  # noqa: E402
from dronecore import DroneState, CommandResult  # noqa: E402


class TestRoomIsPositionInside(unittest.TestCase):
    """Test that isPositionInside correctly rejects wall/boundary positions."""

    def setUp(self):
        self.room = RoomShp("(0 0, 500 0, 500 500, 0 500, 0 0)")

    def test_corner_positions_are_on_wall(self):
        """Corners (0,0), (500,0), (500,500), (0,500) are on the boundary."""
        self.assertFalse(self.room.isPositionInside(Position(0, 0)))
        self.assertFalse(self.room.isPositionInside(Position(500, 0)))
        self.assertFalse(self.room.isPositionInside(Position(500, 500)))
        self.assertFalse(self.room.isPositionInside(Position(0, 500)))

    def test_edge_positions_are_on_wall(self):
        """Points on edges (e.g., (250,0), (0,250)) are on the boundary."""
        self.assertFalse(self.room.isPositionInside(Position(250, 0)))
        self.assertFalse(self.room.isPositionInside(Position(0, 250)))
        self.assertFalse(self.room.isPositionInside(Position(500, 250)))
        self.assertFalse(self.room.isPositionInside(Position(250, 500)))

    def test_interior_positions_are_inside(self):
        """Points strictly inside the room are valid."""
        self.assertTrue(self.room.isPositionInside(Position(1, 1)))
        self.assertTrue(self.room.isPositionInside(Position(250, 250)))
        self.assertTrue(self.room.isPositionInside(Position(499, 499)))
        self.assertTrue(self.room.isPositionInside(Position(1, 499)))

    def test_outside_positions_are_outside(self):
        """Points outside the room are rejected."""
        self.assertFalse(self.room.isPositionInside(Position(-1, 250)))
        self.assertFalse(self.room.isPositionInside(Position(250, -1)))
        self.assertFalse(self.room.isPositionInside(Position(501, 250)))
        self.assertFalse(self.room.isPositionInside(Position(250, 501)))


class TestDroneLocateValidation(unittest.TestCase):
    """Test that locate() rejects positions on or outside room walls."""

    def setUp(self):
        self.room = RoomShp("(0 0, 500 0, 500 500, 0 500, 0 0)", height=250)
        self.drone = DroneVirtual()

    def test_locate_on_wall_corner_rejected(self):
        """Placing drone at corner (0,0) should be rejected."""
        self.drone.locate(0, 0, 90, self.room)
        self.assertFalse(self.drone.command.response)
        self.assertEqual(self.drone.command.result, CommandResult.RES_NO)

    def test_locate_on_wall_edge_rejected(self):
        """Placing drone on a wall edge should be rejected."""
        self.drone.locate(250, 0, 90, self.room)
        self.assertFalse(self.drone.command.response)
        self.assertEqual(self.drone.command.result, CommandResult.RES_NO)

    def test_locate_outside_room_rejected(self):
        """Placing drone outside the room should be rejected."""
        self.drone.locate(600, 600, 90, self.room)
        self.assertFalse(self.drone.command.response)
        self.assertEqual(self.drone.command.result, CommandResult.RES_NO)

    def test_locate_inside_room_accepted(self):
        """Placing drone inside the room should succeed."""
        self.drone.locate(250, 250, 90, self.room)
        self.assertTrue(self.drone.command.response)
        self.assertEqual(self.drone.position.x, 250)
        self.assertEqual(self.drone.position.y, 250)

    def test_locate_near_wall_accepted(self):
        """Placing drone just inside the wall (1,1) should succeed."""
        self.drone.locate(1, 1, 90, self.room)
        self.assertTrue(self.drone.command.response)
        self.assertEqual(self.drone.position.x, 1)
        self.assertEqual(self.drone.position.y, 1)

    def test_locate_rejected_does_not_change_position(self):
        """Failed locate should not change the drone's position."""
        original_x = self.drone.position.x
        original_y = self.drone.position.y
        self.drone.locate(0, 0, 90, self.room)
        self.assertEqual(self.drone.position.x, original_x)
        self.assertEqual(self.drone.position.y, original_y)


class TestTakeoffConstraints(unittest.TestCase):
    """Test that takeoff validates altitude vs room height."""

    def test_takeoff_rejected_if_room_too_short(self):
        """Takeoff should be rejected if room height <= takeoff altitude."""
        room = RoomShp("(0 0, 500 0, 500 500, 0 500, 0 0)", height=50)
        drone = DroneVirtual()
        drone.locate(250, 250, 90, room)
        drone.takeOff()
        # Takeoff altitude is 80, room height is 50 — should be rejected
        self.assertEqual(drone.state, DroneState.ONGROUND)
        self.assertFalse(drone.command.response)

    def test_takeoff_accepted_if_room_tall_enough(self):
        """Takeoff should succeed if room height > takeoff altitude."""
        room = RoomShp("(0 0, 500 0, 500 500, 0 500, 0 0)", height=250)
        drone = DroneVirtual()
        drone.locate(250, 250, 90, room)
        drone.takeOff()
        self.assertEqual(drone.state, DroneState.INFLIGHT)
        self.assertTrue(drone.command.response)
        self.assertEqual(drone.position.z, 80)


class TestGetRandomPositionConstraints(unittest.TestCase):
    """Test that getRandomPosition rejects boundary positions."""

    def setUp(self):
        self.room = RoomShp("(0 0, 500 0, 500 500, 0 500, 0 0)", height=250)

    def test_random_position_single_point_on_wall_rejected(self):
        """getRandomPosition with a point on the wall should be rejected."""
        with self.assertRaises(Exception):
            self.room.getRandomPosition(Position(0, 0, 50))

    def test_random_position_single_point_inside_accepted(self):
        """getRandomPosition with a point inside should succeed."""
        pos = self.room.getRandomPosition(Position(250, 250, 50))
        self.assertEqual(pos.x, 250)
        self.assertEqual(pos.y, 250)

    def test_random_position_range_on_wall_rejected(self):
        """getRandomPosition with boundary range should be rejected."""
        with self.assertRaises(Exception):
            self.room.getRandomPosition(Position(0, 0, 0), Position(100, 100, 100))

    def test_random_position_range_inside_accepted(self):
        """getRandomPosition with inside range should succeed."""
        pos = self.room.getRandomPosition(Position(100, 100, 50), Position(200, 200, 100))
        self.assertGreaterEqual(pos.x, 100)
        self.assertLessEqual(pos.x, 200)
        self.assertGreaterEqual(pos.y, 100)
        self.assertLessEqual(pos.y, 200)


if __name__ == '__main__':
    unittest.main()
