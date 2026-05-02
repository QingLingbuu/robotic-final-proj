from robocasa.environments.kitchen.kitchen import *


class CupMugSorting(Kitchen):
    """
    Cup Mug Sorting: classify drinkware by handle availability.

    Initial scene contains multiple mugs and cups on one counter. The intended
    downstream policy is to move mugs with handles to the handled zone and cups
    without handles to the plain zone.
    """

    # The Panda gripper opens to about 8cm. Plain cups are the top-down baseline,
    # so keep their XY scale below that limit instead of asking top-down to grasp
    # a rim that is as wide as the fully opened gripper.
    PLAIN_CUP_OBJECT_SCALE = [0.75, 0.75, 1.0]
    PLAIN_CUP_MAX_SIZE = (0.072, 0.072, None)

    def __init__(self, num_mugs=2, num_cups=3, *args, **kwargs):
        self.num_mugs = int(num_mugs)
        self.num_cups = int(num_cups)
        self.sorting_metadata = {}
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Sort all drinkware on the counter. Use handle-top-down grasps for mugs "
            "with handles and top-down grasps for cups without handles."
        )
        ep_meta["cup_mug_sorting"] = {
            "num_mugs": self.num_mugs,
            "num_cups": self.num_cups,
            "handled_zone": "left side of the counter",
            "plain_zone": "right side of the counter",
        }
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        total = max(1, self.num_mugs + self.num_cups)
        x_positions = np.linspace(-0.55, 0.55, total)
        front_edge_y = -0.92

        object_specs = []
        for idx in range(max(self.num_mugs, self.num_cups)):
            if idx < self.num_mugs:
                object_specs.append(("mug", "mug", idx + 1, True))
            if idx < self.num_cups:
                object_specs.append(("cup", "cup", idx + 1, False))

        self.sorting_metadata = {}
        for idx, (prefix, group, object_index, has_handle) in enumerate(object_specs):
            name = f"{prefix}_{object_index}"
            self.sorting_metadata[name] = {
                "has_handle": bool(has_handle),
                "target_zone": "handled" if has_handle else "plain",
                "recommended_grasp": "handle_top_down" if has_handle else "top_down",
            }
            object_kwargs = {}
            if not has_handle:
                object_kwargs["object_scale"] = self.PLAIN_CUP_OBJECT_SCALE
                object_kwargs["max_size"] = self.PLAIN_CUP_MAX_SIZE
            cfgs.append(
                dict(
                    name=name,
                    obj_groups=group,
                    graspable=True,
                    washable=True,
                    init_robot_here=idx == 0,
                    **object_kwargs,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="right",
                        ),
                        size=(0.18, 0.18),
                        pos=(float(x_positions[idx]), front_edge_y),
                        rotation=(-np.pi, np.pi),
                    ),
                )
            )
        return cfgs

    def _check_success(self):
        handled_x = []
        plain_x = []
        for idx in range(self.num_mugs):
            obj_name = f"mug_{idx + 1}"
            if obj_name not in self.obj_body_id:
                return False
            handled_x.append(float(self.sim.data.body_xpos[self.obj_body_id[obj_name]][0]))
        for idx in range(self.num_cups):
            obj_name = f"cup_{idx + 1}"
            if obj_name not in self.obj_body_id:
                return False
            plain_x.append(float(self.sim.data.body_xpos[self.obj_body_id[obj_name]][0]))

        if not handled_x or not plain_x:
            return False
        return max(handled_x) < min(plain_x)


class CupMugSortingClean(CupMugSorting):
    """
    Clean two-object version of CupMugSorting for validating grasp and place flow.

    Places one handled mug and one plain cup in an open, front-facing row with
    extra spacing to reduce occlusion from nearby fixtures or other objects.
    """

    def __init__(self, num_mugs=1, num_cups=1, *args, **kwargs):
        super().__init__(num_mugs=num_mugs, num_cups=num_cups, *args, **kwargs)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Sort the two visible drinkware objects on the open counter. "
            "Use handle-top-down for the handled mug and top-down for the plain cup."
        )
        ep_meta["cup_mug_sorting"]["clean_scene"] = True
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        object_specs = [
            ("mug", "mug", 1, True, (-0.25, -0.92)),
            ("cup", "cup", 1, False, (0.25, -0.92)),
        ]

        self.sorting_metadata = {}
        for idx, (prefix, group, object_index, has_handle, pos) in enumerate(object_specs):
            name = f"{prefix}_{object_index}"
            self.sorting_metadata[name] = {
                "has_handle": bool(has_handle),
                "target_zone": "handled" if has_handle else "plain",
                "recommended_grasp": "handle_top_down" if has_handle else "top_down",
            }
            object_kwargs = {}
            if not has_handle:
                object_kwargs["object_scale"] = self.PLAIN_CUP_OBJECT_SCALE
                object_kwargs["max_size"] = self.PLAIN_CUP_MAX_SIZE
            cfgs.append(
                dict(
                    name=name,
                    obj_groups=group,
                    graspable=True,
                    washable=True,
                    init_robot_here=idx == 0,
                    **object_kwargs,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="right",
                        ),
                        size=(0.14, 0.14),
                        pos=(float(pos[0]), float(pos[1])),
                        rotation=(-np.pi, np.pi),
                        ensure_object_boundary_in_range=True,
                    ),
                )
            )
        return cfgs
