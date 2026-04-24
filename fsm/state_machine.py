"""Finite State Machine (FSM) skeleton for robotic manipulation task.

Defines all states and basic transition logic for the picking/placing workflow.
"""

from enum import Enum


class State(Enum):
    """Enumeration of all possible states in the task FSM."""
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    GRASPING = "GRASPING"
    VERIFYING = "VERIFYING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class TaskStateMachine:
    """Finite State Machine for managing task execution states."""
    
    def __init__(self):
        """Initialize FSM in IDLE state."""
        self.current_state = State.IDLE
        self.max_retries = 3
        self.retry_count = 0
    
    def transition_to(self, new_state: State) -> None:
        """
        Transition to a new state.
        
        Args:
            new_state: Target state to transition to
        """
        print(f"Transitioning: {self.current_state.value} -> {new_state.value}")
        self.current_state = new_state
    
    def is_terminal(self) -> bool:
        """
        Check if current state is terminal (SUCCESS or FAILED).
        
        Returns:
            bool: True if terminal, False otherwise
        """
        return self.current_state in [State.SUCCESS, State.FAILED]
    
    def increment_retry(self) -> bool:
        """
        Increment retry counter and check if max retries exceeded.
        
        Returns:
            bool: True if can retry, False if max retries hit
        """
        self.retry_count += 1
        return self.retry_count < self.max_retries
    
    def reset(self) -> None:
        """Reset FSM to initial idle state for new task."""
        self.current_state = State.IDLE
        self.retry_count = 0
    
    def get_current_state(self) -> State:
        """
        Get current state.
        
        Returns:
            State: Current FSM state
        """
        return self.current_state
