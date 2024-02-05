"Master Class for Grounding Game"

import copy
from typing import List, Dict, Tuple
from string import ascii_lowercase as letters

import numpy as np

import clemgame.metrics as ms
from clemgame.clemgame import GameMaster, GameBenchmark
from clemgame import get_logger

from games.grounding.players import Speaker
from games.grounding.instancegenerator import GAME_NAME


# framework logger independent from records/transcript
logger = get_logger(__name__)


class Grounding(GameMaster):
    """Implement mechanisms for playing the grounding game."""
    def __init__(self, experiment: Dict, player_backends: List[str]):
        super().__init__(GAME_NAME, experiment, player_backends)

        # save experiment and player attributes that will be necessary later
        self.context = experiment['name']
        self.model_a = player_backends[0]
        self.model_b = player_backends[1]

        # initialise attributes that will be used for the evaluation scores
        self.complete_turns: int = 0

    def setup(self, subjects: list, entity1_info: list, entity2_info: list,
              n_turns: int, prompt_player_a: str, prompt_player_b: str,
              game_id: int) -> None:
        """Setup episode."""
        # length of episode determined by number of turns provided
        self.n_turns = n_turns

        # instantiate both players
        self.player_a = Speaker(self.model_a, 'A', self.context, subjects,
                                entity1_info, entity2_info)
        self.player_b = Speaker(self.model_b, 'B', self.context, subjects,
                                entity1_info, entity2_info)

        # initialise game variables
        self.current_turn: int = 0

        # initialise common metrics
        self.request_counts = [0] * (n_turns + 1)

        # add initial prompts to each player's messages
        self.initiate(prompt_player_a, prompt_player_b)

        # logging player details in logdoc format
        self.log_players({
            'GM': 'Game master for Grounding',
            'Player 1': f'Player A: {self.model_a}',
            'Player 2': f'Player B: {self.model_b}'
            })

        # logging evaluation keys
        self.log_key('n_turns', n_turns)

    # TO DO: begin probing, end probing in history just as in private shared?
    def probe(self):
        """Probe for common ground beliefs of players."""
        question = 'What do you believe to be shared knowledge between you' \
                   'and your conversation partner?'

        # set up dialogue history expansion for player A
        historyA = self.player_a.history.copy()
        historyA.append({'role': 'user', 'content': ''})
        historyA[-1]['content'] = question

        # log probe question for records for player A
        action = {'type': 'probe question', 'content': question}
        self.log_event(from_='GM', to='Player 1', action=action)

        # get probe question response and log it for records for player A
        prompt, raw_answer, answer = self.player_a(historyA, self.turn)
        action = {'type': 'probe response', 'content': answer}
        self.log_event(from_='GM', to='GM', action=action)

        # set up dialogue history expansion for player B
        historyB = self.player_b.history.copy()
        historyB.append({'role': 'user', 'content': ''})
        historyB[-1]['content'] = question

        # log probe question for records for player A
        action = {'type': 'probe question', 'content': question}
        self.log_event(from_='GM', to='Player 2', action=action)

        # get probe question response and log it for records for player B
        prompt, raw_answer, answer = self.player_b(historyB, self.turn)
        action = {'type': 'probe response', 'content': answer}
        self.log_event(from_='GM', to='GM', action=action)

    def play(self) -> None:
        """Play the game until end."""
        # play the game
        while self.proceed():
            self.current_turn += 1
            # log new turns upon start
            self.log_next_turn()
            # perform a turn (get utterance from each player)
            self.turn()
            # probing after every turn
            self.probe()

        # log game played successfully when n of turns reached
        if self.complete_turns == self.n_turns:
            action = {'type': 'info', 'content': 'game successful'}
            self.log_event(from_='GM', to='GM', action=action)

        # log game ending
        action = {'type': 'info', 'content': 'end game'}
        self.log_event(from_='GM', to='GM', action=action)

        # log temporary game variables needed for evaluation
        self.log_eval_assets()

    def initiate(self, prompt_player_a: str, prompt_player_b: str) -> None:
        """Initialise the dialogue history (grounding specific)."""
        # log news turns upon start
        self.log_next_turn()

        # appending initial player messages to their history
        self.player_a.history.append({'role': 'user',
                                      'content': prompt_player_a})
        self.player_b.history.append({'role': 'user',
                                      'content': prompt_player_b})

        # log messages as events for transcriptions
        action = {'type': 'send message', 'content': prompt_player_a}
        self.log_event(from_='GM', to='Player 1', action=action)
        action = {'type': 'send message', 'content': prompt_player_b}
        self.log_event(from_='GM', to='Player 2', action=action)

    def proceed(self) -> None:
        """Check if the game loop should continue or whether
        number of turns has been reached (grounding specific)."""
        return (self.current_turn < self.n_turns)

    def _get_utterance(self, player: str) -> str:
        """Get utterance from a player and log it (grounding specific)."""
        assert player in ('a', 'b')

        if player == 'a':
            # make an API call (or get programmatic response) from player a
            prompt, raw_answer, answer = self.player_a(self.player_a.history,
                                                       self.current_turn)
            # add reply to its own dialogue history
            self._append_utterance(answer, 'a', 'assistant')
            # also add reply to the records
            action = {'type': 'get message', 'content': answer}
            self.log_event(from_='Player 1', to='GM', action=action,
                           call=(copy.deepcopy(prompt), raw_answer))

        else:
            # make an API call (or get a programmatic response) from player b
            prompt, raw_answer, answer = self.player_b(self.player_b.history,
                                         self.current_turn)
            # add reply to its own dialogue history
            self._append_utterance(answer, 'b', 'assistant')
            # also add reply to the records
            action = {'type': 'get message', 'content': answer}
            self.log_event(from_='Player 2', to='GM', action=action,
                           call=(copy.deepcopy(prompt), raw_answer))

        # increase the number of API requests
        self.request_counts[self.current_turn] += 1

        return answer

    def turn(self) -> None:
        """Perform a game turn by exchanging one utterance between
           players A and B (grounding specific)."""

        # get player A's reply and add it to its history
        answer_a = self._get_utterance('a')

        # add A's reply to B's history
        self._append_utterance(answer_a, 'b', 'user')

        # also add the reply to the transcript
        action = {'type': 'send message', 'content': answer_a}
        self.log_event(from_='GM', to='Player 2', action=action)

        # get player B's reply and add it to its history
        answer_b = self._get_utterance('b')

        # add B's reply to A's history
        self._append_utterance(answer_b, 'a', 'user')

        # also add the reply to the transcript
        action = {'type': 'send message', 'content': answer_b}
        self.log_event(from_='GM', to='Player 1', action=action)

        self.complete_turns += 1

    def _append_utterance(self, utterance: str, 
                          player: str, role: str) -> None:
        """Add an utterance to the history of a player (grounding specific)."""
        assert player in ('a', 'b')
        if player == 'a':
            self.player_a.history.append({'role': role, 'content': utterance})
        else:
            self.player_b.history.append({'role': role, 'content': utterance})

    # TO DO
    def compute_scores(self, episode_interactions: Dict) -> None:
        """Compute episode-level and turn-level scores (mandatory)."""
        played_turns = episode_interactions['Played turns']

    # TO DO
    def log_eval_assets(self) -> None:
        """Aux to log variables needed for scoring (grounding specific)"""
        self.log_key('Played turns', self.current_turn)


# necessary GameBenchmark child
class GroundingGameBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""
    def __init__(self):
        super().__init__(GAME_NAME)

    # defines whether the game is single player or not
    def is_single_player(self):
        return False

    # add description of game
    def get_description(self):
        return 'A simple game in which players have a conversation' \
               'and state what they believe to be common ground.'

    # get game master
    def create_game_master(self,
                           experiment: Dict,
                           player_backends: List[str]
                           ) -> GameMaster:
        return Grounding(experiment, player_backends)
