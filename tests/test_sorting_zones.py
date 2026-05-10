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

    def test_handled_mug_places_in_sink_basin_when_anchor_available(self):
        assignment = {"has_handle": True, "pos": [0.7, -0.3, 0.95], "sim_object_name": "mug_1"}
        sim_objects = [{"name": "mug_1", "pos": [1.0, 0.0, 0.95], "has_handle": True}]
        anchors = {
            "sink": {"pos": [0.5, -0.3, 0.9], "rot": 0.0, "size": [0.5, 0.4, 0.2]},
            "counter": {"pos": [1.0, -0.3, 0.46], "rot": 0.0, "size": [2.5, 0.65, 0.92]},
        }

        target = choose_cup_mug_place_target(assignment, sim_objects, place_anchors=anchors)

        self.assertEqual(target["zone"], "sink")
        self.assertEqual(target["anchor"], "sink")
        self.assertEqual(target["placement_rule"], "spaced_sink_basin")
        self.assertGreater(target["target_pos"][0], anchors["sink"]["pos"][0])
        self.assertLess(target["target_pos"][1], -0.3)

    def test_two_mugs_get_distinct_sink_y_offsets(self):
        anchors = {
            "sink": {"pos": [0.5, -0.3, 0.9], "rot": 0.0, "size": [0.5, 0.4, 0.2]},
            "counter": {"pos": [1.0, -0.3, 0.46], "rot": 0.0, "size": [2.5, 0.65, 0.92]},
        }
        mug_1 = choose_cup_mug_place_target(
            {"has_handle": True, "pos": [0.7, -0.3, 0.95], "sim_object_name": "mug_1"},
            [{"name": "mug_1", "pos": [1.0, 0.0, 0.95], "has_handle": True}],
            place_anchors=anchors,
        )
        mug_2 = choose_cup_mug_place_target(
            {"has_handle": True, "pos": [0.7, -0.3, 0.95], "sim_object_name": "mug_2"},
            [{"name": "mug_2", "pos": [1.0, 0.0, 0.95], "has_handle": True}],
            place_anchors=anchors,
        )

        self.assertNotEqual(mug_1["target_pos"][1], mug_2["target_pos"][1])

    def test_plain_cup_places_on_right_counter_when_anchor_available(self):
        assignment = {"has_handle": False, "pos": [1.4, 0.0, 0.95]}
        sim_objects = [{"name": "cup_1", "pos": [1.4, 0.0, 0.95], "has_handle": False}]
        anchors = {
            "sink": {"pos": [0.5, -0.3, 0.9], "rot": 0.0, "size": [0.5, 0.4, 0.2]},
            "counter": {"pos": [1.0, -0.3, 0.46], "rot": 0.0, "size": [2.5, 0.65, 0.92]},
        }

        target = choose_cup_mug_place_target(assignment, sim_objects, place_anchors=anchors)

        self.assertEqual(target["zone"], "right_counter")
        self.assertEqual(target["anchor"], "counter")
        self.assertEqual(target["placement_rule"], "spread_counter_edge")
        self.assertGreater(target["target_pos"][0], anchors["counter"]["pos"][0])

    def test_multiple_cups_get_distinct_counter_y_offsets(self):
        anchors = {
            "sink": {"pos": [0.5, -0.3, 0.9], "rot": 0.0, "size": [0.5, 0.4, 0.2]},
            "counter": {"pos": [1.0, -0.3, 0.46], "rot": 0.0, "size": [2.5, 0.65, 0.92]},
        }
        cup_1 = choose_cup_mug_place_target(
            {"has_handle": False, "pos": [1.4, 0.0, 0.95], "sim_object_name": "cup_1"},
            [{"name": "cup_1", "pos": [1.4, 0.0, 0.95], "has_handle": False}],
            place_anchors=anchors,
        )
        cup_2 = choose_cup_mug_place_target(
            {"has_handle": False, "pos": [1.4, 0.0, 0.95], "sim_object_name": "cup_2"},
            [{"name": "cup_2", "pos": [1.4, 0.0, 0.95], "has_handle": False}],
            place_anchors=anchors,
        )
        cup_3 = choose_cup_mug_place_target(
            {"has_handle": False, "pos": [1.4, 0.0, 0.95], "sim_object_name": "cup_3"},
            [{"name": "cup_3", "pos": [1.4, 0.0, 0.95], "has_handle": False}],
            place_anchors=anchors,
        )

        self.assertEqual(cup_1["zone"], "right_counter")
        self.assertEqual(cup_2["zone"], "right_counter")
        self.assertEqual(cup_3["zone"], "right_counter")
        self.assertNotEqual(cup_1["target_pos"][1], cup_2["target_pos"][1])
        self.assertNotEqual(cup_2["target_pos"][1], cup_3["target_pos"][1])


if __name__ == "__main__":
    unittest.main()
