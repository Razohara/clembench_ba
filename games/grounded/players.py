"""
A game probing for common ground in LLMs.
Implementation of players participating in the game. 
"""

import random
from typing import List
from backends import Model

from clemgame.clemgame import Player
from clemgame import get_logger


# framework logger independent from records/transcript
logger = get_logger(__name__)

class Speaker(Player):
    """Speakers as Players of the Grounded Game."""
    def __init__(self, backend: Model, nturns: int, player: str):
        # if player is a program and you don't want to make API calls to
        # LLMS, use model_name="programmatic"
        super().__init__(backend)
        self.nturns = nturns
        self.player = player

        # a list to keep the dialogue history
        self.history: List = []

        # call count for mocking
        self.call_count_mock = 0

    def _custom_response(self, messages, turn_idx) -> str:
        """Return a mock message, either a free string response or a list of numbers"""
        self.call_count_mock += 1
        output = ''

        # Pre, Final: Refelexive Probing Question
        if 'ME: Do you know' in messages[-1]['content']:
            # P1
            if self.player.endswith('A'):
                # pre
                output = 'ASIDE: yes'
                # final
                if self.call_count_mock >= 7:
                    output = 'ASIDE: yes'
            # P2
            else:
                # pre
                if self.call_count_mock < 7:
                    rb = random.random()
                    if rb <= 0.5:
                        output = 'ASIDE: no'
                    else:
                        output = "ASIDE: don't know"
                # final
                else:
                    # case 1
                    output = 'ASIDE: yes'

                    # case 2
                    # rb = random.random()
                    # if rb <= 0.5:
                    #     output = 'ASIDE: no'
                    # else:
                    #     output = "ASIDE: don't know"

            return output

        # Pre, Final: Shared or Symmetric Probing Question
        elif 'ME: Does your' in messages[-1]['content']: 
            # P1
            if self.player.endswith('A'):
                # pre
                if self.call_count_mock < 4: # 5 for wrong format test
                    ra = random.random()
                    if ra <= 0.5:
                        output = 'ASIDE: no'
                    else:
                        output = "ASIDE: don't know"
                # final
                else:
                    # case 1
                    output = 'ASIDE: yes'

                    # case 2
                    # ra = random.random()
                    # if ra <= 0.5:
                    #     output = 'ASIDE: no'
                    # else:
                    #     output = "ASIDE: don't know"
            # P2
            else:
                # pre
                if self.call_count_mock < 7:
                    output = "ASIDE: don't know"
                # final
                else:
                    # case 1
                    output = 'ASIDE: yes'

                    # case 2
                    # rb = random.random()
                    # if rb <= 0.5:
                    #     output = 'ASIDE: no'
                    # else:
                    #     output = "ASIDE: don't know"

            return output

        # Post: Discussed Probing Question
        elif 'ME: Have you' in messages[-1]['content']:
            # P1
            if self.player.endswith('A'):
                ra = random.random()
                if ra <= 0.5:
                    output = 'ASIDE: yes'
                else:
                    output = "ASIDE: yes"
            # P2
            else:
                rb = random.random()
                if rb <= 0.5:
                    output = 'ASIDE: yes'
                else:
                    output = "ASIDE: yes"
            
            return output

        else:
            # mock player A knowing stuff
            if self.player.endswith('A'):
                output = random.sample(['Simon has adopted Malcolm.', 'Simon has started gardening.'], 1)[0]
            # mock player B not knowing stuff
            else:
                output = random.sample(['Oh, I did not know that.', 'For real?'], 1)[0]

            return output
