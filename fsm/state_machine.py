"""Finite State Machine (FSM) skeleton for robotic manipulation task."""

from enum import Enum


class State(Enum):
    """Enumeration of the required task states."""

    IDLE = "IDLE"
    PLANNING = "PLANNING"
    CLEARING = "CLEARING"
    GRASPING = "GRASPING"
    VERIFYING = "VERIFYING"
    RETRY_SENSING = "RETRY_SENSING"
    RETRY_PUSH = "RETRY_PUSH"
    RETRY_GRASP = "RETRY_GRASP"
    RESET = "RESET"
    EMERGENCY = "EMERGENCY"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class TaskStateMachine:
    """Finite State Machine with explicit retry counters for N1/N2/N3."""

    def __init__(self, max_planning_retries=3, max_push_retries=2, max_grasp_retries=2):
        self.current_state = State.IDLE
        self.max_planning_retries = max_planning_retries
        self.max_push_retries = max_push_retries
        self.max_grasp_retries = max_grasp_retries
        self.retry_counts = {
            State.RETRY_SENSING: 0,
            State.RETRY_PUSH: 0,
            State.RETRY_GRASP: 0,
        }
        self.planning_candidates = []
        self.selected_candidate_index = None

    def transition_to(self, new_state: State) -> None:
        """Transition to a new state."""
        print(f"Transitioning: {self.current_state.value} -> {new_state.value}")
        self.current_state = new_state

    def is_terminal(self) -> bool:
        """Check whether the current state is terminal."""
        return self.current_state in [State.SUCCESS, State.FAILED]

    def set_planning_candidates(self, candidates) -> None:
        """Store the latest planning candidates and select the first if available."""
        self.planning_candidates = list(candidates or [])
        self.selected_candidate_index = 0 if self.planning_candidates else None

    def get_selected_candidate(self):
        """Return the currently selected candidate, if any."""
        if self.selected_candidate_index is None:
            return None
        if not 0 <= self.selected_candidate_index < len(self.planning_candidates):
            return None
        return self.planning_candidates[self.selected_candidate_index]

    def get_selected_candidate_index(self):
        """Return the selected candidate index, if any."""
        return self.selected_candidate_index

    def advance_to_next_candidate(self) -> bool:
        """Advance selection to the next candidate if one exists."""
        if self.selected_candidate_index is None:
            return False
        next_index = self.selected_candidate_index + 1
        if next_index >= len(self.planning_candidates):
            return False
        self.selected_candidate_index = next_index
        return True

    def record_retry(self, retry_state: State) -> bool:
        """Record a retry event and report whether another attempt is allowed."""
        if retry_state not in self.retry_counts:
            raise ValueError(f"Unsupported retry state: {retry_state}")

        self.retry_counts[retry_state] += 1
        self.transition_to(retry_state)

        if retry_state == State.RETRY_SENSING:
            return self.retry_counts[retry_state] < self.max_planning_retries
        if retry_state == State.RETRY_PUSH:
            return self.retry_counts[retry_state] < self.max_push_retries
        return self.retry_counts[retry_state] < self.max_grasp_retries

    def handle_perception_timeout(self) -> State:
        """Route stale perception reads to RETRY_SENSING or FAILED."""
        if self.record_retry(State.RETRY_SENSING):
            return self.current_state
        self.transition_to(State.FAILED)
        return self.current_state

    def handle_invalid_perception(self) -> State:
        """Route invalid or low-confidence perception to RETRY_SENSING or FAILED."""
        return self.handle_perception_timeout()

    def handle_push_blocked(self) -> State:
        """Route blocked clearing attempts to RETRY_PUSH or FAILED."""
        if self.record_retry(State.RETRY_PUSH):
            return self.current_state
        self.transition_to(State.FAILED)
        return self.current_state

    def handle_empty_grasp(self) -> State:
        """Route empty grasps to RETRY_GRASP or FAILED."""
        if self.advance_to_next_candidate():
            self.retry_counts[State.RETRY_GRASP] += 1
            self.transition_to(State.RETRY_GRASP)
            return self.current_state
        if self.record_retry(State.RETRY_GRASP):
            return self.current_state
        self.transition_to(State.FAILED)
        return self.current_state

    def handle_collision(self) -> State:
        """Any collision escalates immediately to EMERGENCY."""
        self.transition_to(State.EMERGENCY)
        return self.current_state

    def get_failure_counts(self):
        """Return experiment counters N1/N2/N3."""
        return {
            "n1": self.retry_counts[State.RETRY_SENSING],
            "n2": self.retry_counts[State.RETRY_PUSH],
            "n3": self.retry_counts[State.RETRY_GRASP],
        }

    def reset(self) -> None:
        """Reset FSM state and retry counters."""
        self.current_state = State.IDLE
        for state in self.retry_counts:
            self.retry_counts[state] = 0
        self.planning_candidates = []
        self.selected_candidate_index = None

    def get_current_state(self) -> State:
        """Return the current FSM state."""
        return self.current_state
