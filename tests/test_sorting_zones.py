import unittest

from planner.sorting_zones import choose_cup_mug_place_target


class SortingZoneTests(unittest.TestCase):
    def test_plain_cup_places_to_right_zone_without_fixture_anchors(self):
        sim_objects = [
            {"name": "mug_1", "pos": [1.0, 0.0, 0.9], "has_handle": True},
            {"name": "cup_1", "pos": [1.2, 0.1, 0.9], "has_handle": False},
            {"name": "mug_2", "pos": [1.4, 0.0, 0.9], "has_handle": True},
            {"name": "cup_2", "pos": [1.6, 0.1, 0.9], "has_handle": False},
        ]
        assignment = {"has_handle": False, "pos": [1.2, 0.1, 0.95]}

        target = choose_cup_mug_place_target(assignment, sim_objects)

        self.assertGreater(target["target_pos"][0], 1.6)
        self.assertEqual(target["zone"], "plain")

    def test_handled_mug_places_to_left_zone_without_fixture_anchors(self):
        sim_objects = [
            {"name": "mug_1", "pos": [1.0, 0.0, 0.9], "has_handle": True},
            {"name": "cup_1", "pos": [1.2, 0.1, 0.9], "has_handle": False},
            {"name": "mug_2", "pos": [1.4, 0.0, 0.9], "has_handle": True},
            {"name": "cup_2", "pos": [1.6, 0.1, 0.9], "has_handle": False},
        ]
        assignment = {"has_handle": True, "pos": [1.0, 0.0, 0.95]}

        target = choose_cup_mug_place_target(assignment, sim_objects)

        self.assertLess(target["target_pos"][0], 1.0)
        self.assertEqual(target["zone"], "handled")

    def test_handled_mug_places_in_nearest_sink_basin_when_anchor_available(self):
        assignment = {"has_handle": True, "pos": [0.7, -0.3, 0.95]}
        sim_objects = [{"name": "mug_1", "pos": [1.0, 0.0, 0.95], "has_handle": True}]
        anchors = {
            "sink": {"pos": [0.5, -0.3, 0.9], "rot": 0.0, "size": [0.5, 0.4, 0.2]},
            "counter": {"pos": [1.0, -0.3, 0.46], "rot": 0.0, "size": [2.5, 0.65, 0.92]},
        }

        target = choose_cup_mug_place_target(assignment, sim_objects, place_anchors=anchors)

        self.assertEqual(target["zone"], "sink")
        self.assertEqual(target["anchor"], "sink")
        self.assertEqual(target["placement_rule"], "nearest_sink_basin")
        self.assertGreater(target["target_pos"][0], anchors["sink"]["pos"][0])
        self.assertAlmostEqual(target["target_pos"][1], -0.3)

    def test_handled_mug_uses_left_sink_basin_when_source_is_left(self):
        assignment = {"has_handle": True, "pos": [0.2, -0.3, 0.95]}
        sim_objects = [{"name": "mug_1", "pos": [0.2, -0.3, 0.95], "has_handle": True}]
        anchors = {
            "sink": {"pos": [0.5, -0.3, 0.9], "rot": 0.0, "size": [0.5, 0.4, 0.2]},
            "counter": {"pos": [1.0, -0.3, 0.46], "rot": 0.0, "size": [2.5, 0.65, 0.92]},
        }

        target = choose_cup_mug_place_target(assignment, sim_objects, place_anchors=anchors)

        self.assertEqual(target["zone"], "sink")
        self.assertLess(target["target_pos"][0], anchors["sink"]["pos"][0])

    def test_plain_cup_places_on_counter_opposite_sink_when_anchor_available(self):
        assignment = {"has_handle": False, "pos": [1.4, 0.0, 0.95]}
        sim_objects = [{"name": "cup_1", "pos": [1.4, 0.0, 0.95], "has_handle": False}]
        anchors = {
            "sink": {"pos": [0.5, -0.3, 0.9], "rot": 0.0, "size": [0.5, 0.4, 0.2]},
            "counter": {"pos": [1.0, -0.3, 0.46], "rot": 0.0, "size": [2.5, 0.65, 0.92]},
        }

        target = choose_cup_mug_place_target(assignment, sim_objects, place_anchors=anchors)

        self.assertEqual(target["zone"], "opposite_counter")
        self.assertEqual(target["anchor"], "counter")
        self.assertGreater(target["target_pos"][0], anchors["counter"]["pos"][0])
        self.assertLess(target["target_pos"][1], anchors["counter"]["pos"][1])


if __name__ == "__main__":
    unittest.main()
