import numpy as np
from rlgym.utils import TerminalCondition
from rlgym.utils.gamestates import GameState
from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition, \
    TimeoutCondition



class EuropaTerminalCondition(TerminalCondition):
    """Sets the terminal conditions for an episode"""
    def __init__(self, tick_skip=8):
        super().__init__()
        fps = 120 / tick_skip
        self.no_touch = NoTouchTimeoutCondition(round(60 * 5 * fps)) # change to 1 min later in training
        self.goal_scored = GoalScoredCondition()
        self.timeout = TimeoutCondition(round(60 * 5 * fps))  # should be 5 #added for testing

    def reset(self, initial_state: GameState):
        self.no_touch.reset(initial_state)
        self.goal_scored.reset(initial_state)
        self.timeout.reset(initial_state)

    def is_terminal(self, current_state: GameState) -> bool:
        if self.no_touch.is_terminal(current_state):
            return True
        if self.goal_scored.is_terminal(current_state):
            return True
        if self.timeout.is_terminal(current_state):
            return True  
        return False