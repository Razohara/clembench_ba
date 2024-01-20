"Define Player Behaviour"

import random
from string import ascii_lowercase as letters
from typing import List

from clemgame.clemgame import Player


class Speaker(Player):
    def __init__(self, model_name: str, player: str, context: str, subjects: list, entity1_info: list, entity2_info: list):
        # always initialise the Player class with the model_name argument
        # if the player is a program and you don't want to make API calls to
        # LLMS, use model_name="programmatic"
        super().__init__(model_name)
        self.player: str = player
        self.context: str = context
        self.subjects: list = subjects
        self.entity1_info: list = entity1_info
        self.entity2_info: list = entity2_info

        # a list to keep the dialogue history
        self.history: List = []

    # implement this method as you prefer, with these same arguments
    def _custom_response(self, messages, turn_idx) -> str:
        """Return a mock message about one or both of the subjects containing information from knowledge base."""
        # introduce a small probability that the player fails
        prob_fail = 0.05
        if random.random() < prob_fail:
            return f"from {self.player}, turn {turn_idx}: FAILED."
        else:
            return f"from {self.player}, turn {turn_idx}: {random.choice(self.entity1_info)}, {random.choice(self.entity2_info)}."