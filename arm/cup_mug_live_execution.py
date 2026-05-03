"""Reusable live execution helper for CupMugSorting single-target actions."""

from dataclasses import dataclass

import numpy as np

from arm.base_torso import preposition_base_torso
from arm.reach_retry import should_retry_with_preposition
from arm.reachability import diagnose_reachability
from arm.robocasa_primitives import (
    execute_close,
    execute_lift,
    execute_oriented_top_down_reach,
    execute_place,
    execute_reach,
)
from planner.sorting_zones import choose_cup_mug_place_target


@dataclass
class LiveExecutionOptions:
    render_sleep_sec: float = 0.005
    enable_base_torso_preposition: bool = False
    auto_preposition_retry: bool = False
    skip_vision_refresh_after_base_preposition: bool = False
    retry_final_error_threshold: float = 0.03
    base_torso_steps: int = 30
    base_action_start: int = 7
    base_action_end: int = 10
    torso_action_index: int | None = None
    base_desired_xy_standoff: float = 0.18
    base_xy_deadband: float = 0.06
    base_preposition_gain: float = 4.0
    base_action_limit: float = 0.35
    use_base_action_mapping: bool = False
    base_mapping_trust: float = 0.25
    base_max_world_delta: float = 0.03
    use_drinkware_classification: bool = False
    retry_place_once: bool = False


def run_live_target_execution(
    env,
    current_obs,
    candidate,
    active_candidate_summary,
    action_mapping,
    base_action_mapping,
    infer_candidate_from_obs,
    cup_mug_sorting_place_anchors,
    options: LiveExecutionOptions,
):
    """Execute one selected cup/mug target in the live RoboCasa env."""

    def run_preposition(obs_for_preposition, reason, current_candidate):
        next_obs, summary = preposition_base_torso(
            env,
            obs=obs_for_preposition,
            target_pos=current_candidate["pos"],
            action_mapping=action_mapping,
            steps=int(options.base_torso_steps),
            render_sleep_sec=float(options.render_sleep_sec),
            base_slice=(int(options.base_action_start), int(options.base_action_end)),
            torso_index=None if options.torso_action_index is None else int(options.torso_action_index),
            desired_xy_standoff=float(options.base_desired_xy_standoff),
            xy_deadband=float(options.base_xy_deadband),
            base_gain=float(options.base_preposition_gain),
            base_action_limit=float(options.base_action_limit),
            base_action_mapping=base_action_mapping if options.use_base_action_mapping else None,
            base_mapping_trust=float(options.base_mapping_trust),
            max_base_world_delta=float(options.base_max_world_delta),
        )
        summary["reason"] = reason
        return next_obs, summary

    def run_reach(obs_for_reach, current_candidate):
        if current_candidate["grasp_type"] == "handle_top_down":
            return execute_oriented_top_down_reach(
                env,
                obs=obs_for_reach,
                candidate=current_candidate,
                action_mapping=action_mapping,
                render_sleep_sec=float(options.render_sleep_sec),
            )
        return execute_reach(
            env,
            obs=obs_for_reach,
            candidate=current_candidate,
            action_mapping=action_mapping,
            render_sleep_sec=float(options.render_sleep_sec),
        )

    def refresh_assignment_pose(selected_assignment, sim_objects):
        assignment = dict(selected_assignment or {})
        selected_name = assignment.get("sim_object_name")
        if selected_name is None:
            return assignment
        for sim_object in sim_objects:
            if str(sim_object.get("name")) == str(selected_name) and sim_object.get("pos") is not None:
                assignment["pos"] = [float(value) for value in sim_object.get("pos")]
                break
        return assignment

    preposition_summary = None
    candidate_refresh_summary = None
    retry_summary = None
    candidate_rejection = None

    if options.enable_base_torso_preposition:
        candidate_before_preposition = candidate
        current_obs, preposition_summary = run_preposition(current_obs, "before_first_reach", candidate)
        if preposition_summary.get("effective_should_move") and not options.skip_vision_refresh_after_base_preposition:
            candidate, candidate_rejection, candidate_refresh_summary = infer_candidate_from_obs(
                current_obs,
                reason="after_base_preposition",
            )
            candidate_refresh_summary["previous_candidate"] = candidate_before_preposition
            active_candidate_summary = candidate_refresh_summary

    current_obs, reach_summary = run_reach(current_obs, candidate)
    reachability_diagnosis = diagnose_reachability(reach_summary, candidate=candidate)
    if options.auto_preposition_retry and should_retry_with_preposition(
        reach_summary,
        reachability_diagnosis,
        candidate,
        final_error_threshold=float(options.retry_final_error_threshold),
    ):
        first_reach_summary = reach_summary
        first_diagnosis = reachability_diagnosis
        current_obs, retry_preposition_summary = run_preposition(current_obs, "after_reach_failure", candidate)
        retry_candidate_refresh_summary = None
        if retry_preposition_summary.get("effective_should_move") and not options.skip_vision_refresh_after_base_preposition:
            previous_candidate = candidate
            candidate, candidate_rejection, retry_candidate_refresh_summary = infer_candidate_from_obs(
                current_obs,
                reason="after_retry_base_preposition",
            )
            retry_candidate_refresh_summary["previous_candidate"] = previous_candidate
            active_candidate_summary = retry_candidate_refresh_summary
        current_obs, reach_summary = run_reach(current_obs, candidate)
        reachability_diagnosis = diagnose_reachability(reach_summary, candidate=candidate)
        retry_summary = {
            "triggered": True,
            "first_reach": first_reach_summary,
            "first_reachability_diagnosis": first_diagnosis,
            "preposition": retry_preposition_summary,
            "candidate_refresh": retry_candidate_refresh_summary,
            "second_reach_success": bool(reach_summary.get("reach_success")),
        }
    elif options.auto_preposition_retry:
        retry_summary = {"triggered": False, "reason": "retry_policy_not_matched"}

    close_summary = None
    lift_summary = None
    place_plan = None
    place_summary = None
    failure_phase = None

    if reach_summary.get("reach_only"):
        failure_phase = None if reach_summary["reach_success"] else "reach"
    elif not reach_summary["reach_success"]:
        failure_phase = "reach"
    else:
        current_obs, close_summary = execute_close(
            env,
            obs=current_obs,
            render_sleep_sec=float(options.render_sleep_sec),
        )
        if not close_summary["close_success"]:
            failure_phase = "close"
        else:
            current_obs, lift_summary = execute_lift(
                env,
                obs=current_obs,
                action_mapping=action_mapping,
                render_sleep_sec=float(options.render_sleep_sec),
            )
            if not lift_summary["lift_success"]:
                failure_phase = "lift"
            elif options.use_drinkware_classification:
                classification = active_candidate_summary.get("drinkware_classification") or {}
                selected_assignment = classification.get("selected_assignment")
                sim_objects = classification.get("sim_objects", [])
                if selected_assignment is not None:
                    selected_assignment = refresh_assignment_pose(selected_assignment, sim_objects)
                    place_plan = choose_cup_mug_place_target(
                        selected_assignment,
                        sim_objects,
                        place_anchors=cup_mug_sorting_place_anchors(env),
                    )
                    current_obs, place_summary = execute_place(
                        env,
                        obs=current_obs,
                        place_target=place_plan["target_pos"],
                        action_mapping=action_mapping,
                        render_sleep_sec=float(options.render_sleep_sec),
                    )
                    if not place_summary["place_success"]:
                        if options.retry_place_once:
                            selected_assignment_retry = refresh_assignment_pose(selected_assignment, sim_objects)
                            retry_place_plan = choose_cup_mug_place_target(
                                selected_assignment_retry,
                                sim_objects,
                                place_anchors=cup_mug_sorting_place_anchors(env),
                            )
                            current_obs, retry_place_summary = execute_place(
                                env,
                                obs=current_obs,
                                place_target=retry_place_plan["target_pos"],
                                action_mapping=action_mapping,
                                render_sleep_sec=float(options.render_sleep_sec),
                            )
                            place_summary = {
                                "first_attempt": place_summary,
                                "retry_attempt": retry_place_summary,
                            }
                            place_plan = {
                                "first_attempt": place_plan,
                                "retry_attempt": retry_place_plan,
                            }
                            if not retry_place_summary["place_success"]:
                                failure_phase = "place"
                        else:
                            failure_phase = "place"

    execution_summary = {
        "candidate": {
            "id": int(candidate["id"]),
            "grasp_type": candidate["grasp_type"],
            "pos": candidate["pos"],
            "score": float(candidate.get("score", 0.0)),
            "source_handle_width": candidate.get("source_handle_width"),
            "source_handle_candidate_id": candidate.get("source_handle_candidate_id"),
            "source_handle_pos": candidate.get("source_handle_pos"),
            "source_top_down_candidate_id": candidate.get("source_top_down_candidate_id"),
        },
        "base_torso_preposition": preposition_summary,
        "candidate_after_base_preposition": candidate_refresh_summary,
        "base_action_mapping": None if base_action_mapping is None else np.asarray(base_action_mapping, dtype=float).tolist(),
        "auto_preposition_retry": retry_summary,
        "reach": reach_summary,
        "reachability_diagnosis": reachability_diagnosis,
        "close": close_summary,
        "lift": lift_summary,
        "place_plan": place_plan,
        "place": place_summary,
        "overall_success": failure_phase is None,
        "failure_phase": failure_phase,
    }
    return {
        "obs": current_obs,
        "candidate": candidate,
        "candidate_rejection": candidate_rejection,
        "active_candidate_summary": active_candidate_summary,
        "execution_summary": execution_summary,
    }
