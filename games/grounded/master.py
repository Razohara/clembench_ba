"""
A game probing for common ground in cLLMs.
Implementation of a game master that controls game mechanisms. 
"""

import copy
import re
from typing import List, Dict

import numpy as np

import clemgame.metrics as ms
from clemgame.clemgame import GameMaster, GameBenchmark, GameScorer
from clemgame import get_logger
from backends import Model

from games.grounded.players import Speaker
from games.grounded.constants import (
    GAME_NAME, LANG, N_INSTANCES, N_INFO_PER_EXP, N_TURNS_PER_DUR, SEED,
    REFLEXIVE, SHARED, WORDS_PATH, KNOW_PATH, PROMPT_PATH, SUCCESS, NOT_SUCCESS,
    NOT_PARSED, RESULT, UPDATE, INVALID, INVALID_LABEL, N_RETRIES, N_PROBINGS, 
    N_PROBINGS_FINAL, DISCUSSEDQ, EXPECTATIONS)

from games.grounded.ground_eval import ProbeEval


# framework logger independent from records/transcript
logger = get_logger(__name__)


class Grounded(GameMaster):
    """Implement mechanisms for playing the grounded game."""
    def __init__(self, experiment: Dict, player_backends: List[Model]):
        super().__init__(GAME_NAME, experiment, player_backends)

        # experiment and player attributes
        self.mode = experiment['name']
        self.model_a = player_backends[0]
        self.model_b = player_backends[1]

        # turn counters
        self.max_turns: int = 0
        self.complete_turns: int = 0
        self.complete_turns_convo: int = 0

        # initialise framework metrics
        self.request_counts: List = None
        self.parsed_request_counts: List = None
        self.violated_request_counts: List = None
        
        self.aborted: bool = False

    def setup(self, game_id: int, fact: str, fact_labels: list, n_turns: int,
              prompt_player_a: str, prompt_player_b: str,
              probe_reflexive: str, probe_shared: str, probe_discussed: str) -> None:
        """Setup episode."""
        # get turns for player conversation
        self.n_turns_convo = n_turns

        # determine maximum episode turns
        self.max_turns = (self.n_turns_convo * 2) + N_PROBINGS

        # get episode probing questions and expectations
        self.reflexive = probe_reflexive
        self.shared = probe_shared
        self.discussed_q = probe_discussed
        self.expectations = EXPECTATIONS

        # set up counters for probing questions
        # and responses matching expectations per player
        self.case1 = False
        self.case2 = False

        self.expected_responses_final_a: int = 0
        self.expected_responses_final_b: int = 0
        self.probing_questions_asked_final_a: int = 0
        self.probing_questions_asked_final_b: int = 0

        self.expected_responses_a: int = 0
        self.expected_responses_b: int = 0
        self.probing_questions_asked_a: int = 0
        self.probing_questions_asked_b: int = 0

        # set players
        self.player_a = Speaker(self.model_a, self.n_turns_convo, 'A')
        self.player_b = Speaker(self.model_b, self.n_turns_convo, 'B')

        # set game variables
        self.current_turn: int = 0
        self.current_turn_convo: int = 0

        self.discussed: bool = False
        self.not_discussed: bool = False

        self.coherent = False
        self.fact_mentioned = False
        self.fact_labels = fact_labels

        self.success: bool = False
        self.lose: bool = False

        self.probe_results = {'a': {self.reflexive: '', self.shared: ''}, 
                         'b': {self.reflexive: '', self.shared: ''}}

        # set turn variables
        self.request_counts = [0] * (self.max_turns + 1)
        self.parsed_request_counts = [0] * (self.max_turns + 1)
        self.violated_request_counts = [0] * (self.max_turns + 1)

        # grounded specific parsed request counter
        self.probing_parsed_counter = 0
        self.probing_request_counter = 0

        # set prompts for players
        self.initiate(prompt_player_a, prompt_player_b)

        # set probing words
        words = self.load_json(WORDS_PATH.format(LANG))
        self.aside = words['ASIDE']
        self.me = words['ME']
        self.yes = words['YES']
        self.no = words['NO']
        self.dunno = words['DUNNO']

        # log player details in logdoc format
        self.log_players({'GM': 'Game master for Grounded Game',
                          'Player 1': f'Player A: {self.model_a}',
                          'Player 2': f'Player B: {self.model_b}'})

        # set up evaluation
        self._log_eval_assets()

    def _probing_loop(self, probe_question: str, player: str):
        tries = 0
        successful = False

        if player == 'a':
            to_pl = 'Player 1'
            player_prompted = self.player_a
        elif player == 'b':
            to_pl = 'Player 2'
            player_prompted = self.player_b    

        while tries < N_RETRIES and successful == False:
            # set up history for probing question
            history = player_prompted.history.copy()
            history.append({'role': 'user', 'content': ''})
            history[-1]['content'] = probe_question

            # log probing question
            action = {'type': 'probing question', 
                    'content': probe_question}
            self.log_event(from_='GM', to=to_pl, action=action)

            # get probing question response
            prompt, raw_answer, answer = player_prompted(history, self.current_turn)
            action = {'type': 'probing response', 'content': answer}
            self.log_event(from_=to_pl, to='GM', action=action, 
                        call=(prompt, raw_answer))

            # increase request counters
            self.request_counts[self.current_turn] += 1
            self.probing_request_counter += 1

            # parse probing question response
            parsed_response = self._parse_probing_response(answer)

            # check if valid response, otherwise try again
            if parsed_response in (self.yes, self.no, self.dunno):
                successful = True
                self.parsed_request_counts[self.current_turn] += 1
                self.probing_parsed_counter += 1

                action = {'type': 'parse format', 'content': 'valid format'}
                self.log_event(from_='GM', to='GM', action=action)
                
                break
            
            # increase violated request counts and tries
            self.violated_request_counts[self.current_turn] += 1
            tries += 1

            # log invalid
            action = {'type': 'parse format', 'content': 'invalid format'}
            self.log_event(from_='GM', to='GM', action=action)
            # log retry if allowed
            if tries < N_RETRIES:
                action = {'type': 'info', 'content': 'retrying'}
                self.log_event(from_='GM', to='GM', action=action)
        # log no retries left
        if tries == N_RETRIES and not successful:
            action = {'type': 'info', 'content': 'no retries left'}
            self.log_event(from_='GM', to='GM', action=action)

        return probe_question, parsed_response, successful

    def proceed(self) -> None:
        """Check if the game should continue or whether
        number of turns has been reached."""
        return (self.current_turn <= self.max_turns
                and not self.aborted
                and not self.lose)

    def qna(self, question: str, questionee: str):
        """Pose question to and get answer from a player while also checking format.
        
        question: str       probing question to be posed
        questionee: str     player to be questioned ['a'|'b']        
        """
        question_posed, response_player, format_valid = self._probing_loop(question, questionee)
        self.probe_results[questionee][question_posed] = response_player
        
        if questionee == 'a':
            self.probing_questions_asked_a += 1
        elif questionee == 'b':
            self.probing_questions_asked_b += 1

        # if invalid format abort
        if not format_valid:
            self.aborted = True

        # if valid format complete turn
        else:
            self.complete_turns += 1

    def _sanity_check_response(self, player: str, question: str, type='default'):
        """Check response for sanity depending on state of the game."""
        sane = True

        # get player expectations
        if player == 'a':
            expectations = self.expectations['a']
        elif player == 'b':
            expectations = self.expectations['b']

        # get probing expectations for player
        if type == 'pre':
            expectations = expectations['pre']
        elif type == 'default':
            if self.discussed:
                expectations = expectations['final']['discussed']
            elif self.not_discussed:
                expectations = expectations['final']['not_discussed']
        
        if question == self.reflexive:
            expectations = expectations['refl']
        elif question == self.shared:
            expectations = expectations['symm']

        # post sanity check
        if type == 'post':
            # check whether response by P2 matches response by P1 already in probe results
            discussed_a = False
            discussed_b = False

            if self.probe_results['a'][self.discussed_q] == self.yes:
                discussed_a = True
            if self.probe_results['b'][self.discussed_q] == self.yes:
                discussed_b = True
            
            # only expectation during post sanity check is matching signals about discussion of fact
            if discussed_a == discussed_b:
                # increase counter of expected responses by P2 (P1 has already been increased after qna)
                self.expected_responses_b += 1

                # P1 and P2 both signal discussion of fact 
                if discussed_a and discussed_b:
                    self.discussed = True
                # P1 and P2 both signal no discussion of fact
                elif not discussed_a and not discussed_b:
                    self.not_discussed = True

            # expectation not met, no matching signals
            else:
                sane = False

            return sane

        # pre sanity check and final probing
        # check whether response by player matches expectations for player and question
        if self.probe_results[player][question] in expectations:
            # increase evaluation counters
            if player == 'a':
                self.expected_responses_a += 1
                if type == 'default':
                    self.expected_responses_final_a += 1
            elif player == 'b':
                self.expected_responses_b += 1
                if type == 'default':
                    self.expected_responses_final_b += 1
        # expectation not met
        else:
            sane = False

        return sane


    def probe(self, type='default'):
        """Probe for common ground beliefs of players."""
        probe_success = False

        if self.proceed():
            # log turn per model request for each probing question
            self.current_turn += 1
            self.log_next_turn()

            # log start of probing
            if type == 'default':
                action = {'type': 'final probing', 'content': 'start final probing'}
            elif type == 'pre':
                action = {'type': 'pre sanity check', 'content': 'start pre sanity check'}
            elif type == 'post':
                action = {'type': 'post sanity check', 'content': 'start post sanity check'}
            self.log_event(from_='GM', to='GM', action=action)

            # probing questions for post sanity check
            if type == 'post':
                # begin with discussion question P1
                # pose question and get answer while also checking format
                self.qna(self.discussed_q, 'a')

                # As P1 is the first one asked in this check of matching responses
                # they are free to say yes, no, dunno, so increase expected if no abort
                if not self.aborted:
                    self.expected_responses_a += 1
                
                # if no abort move to P2
                if not self.aborted:
                    # log turn per model request for each probing question
                    self.current_turn += 1
                    self.log_next_turn()
                    # discussion question P2
                    # pose question and get answer while also checking format
                    self.qna(self.discussed_q, 'b')

                    # if no abort conduct post sanity check after having both P1 and P2 response
                    if not self.aborted:
                        post_sane = self._sanity_check_response('b', self.discussed_q, type)
                        # if not sane lose
                        if not post_sane:
                            self.lose = True

                            action = {'type': 'post sanity check', 'content': 'expected not matched'}
                            self.log_event(from_='GM', to='GM', action=action)

                        # if sane probe success
                        else:
                            probe_success = True

                            if self.discussed:
                                content = 'expected matched: discussion of fact indicated by both P1 and P2'
                            elif self.not_discussed:
                                content = 'expected matched: no discussion of fact indicated by both P1 and P2'

                            action = {'type': 'post sanity check', 'content': content}
                            self.log_event(from_='GM', to='GM', action=action)

                # log end of post sanity check probing
                action = {'type': 'post sanity check', 'content': 'end post sanity check'}
                self.log_event(from_='GM', to='GM', action=action)

                return probe_success

            # probing questions for pre sanity check and final probing
            else:
                # reflexive probing question P1
                # pose question and get answer while also checking format
                self.qna(self.reflexive, 'a')
                if type == 'default':
                    self.probing_questions_asked_final_a += 1

                if not self.aborted:
                    # sanity check response if no abort before
                    sane_refl_a = self._sanity_check_response('a', self.reflexive, type)
                    if not sane_refl_a:
                        self.lose = True

        # if no abort proceed
        if not self.aborted:
            # log turn per model request for each probing question
            self.current_turn += 1
            self.log_next_turn()
            # shared probing question P1
            # pose question and get answer while also checking format
            self.qna(self.shared, 'a')
            if type == 'default':
                self.probing_questions_asked_final_a += 1

            if not self.aborted:
                # sanity check response if no abort before
                sane_symm_a = self._sanity_check_response('a', self.shared, type)
                if not sane_symm_a:
                    self.lose = True

        # if no abort proceed
        if not self.aborted:
            # log turn per model request for each probing question
            self.current_turn += 1
            self.log_next_turn()
            # reflexive probing question P2
            # pose question and get answer while also checking format
            self.qna(self.reflexive, 'b')
            if type == 'default':
                self.probing_questions_asked_final_b += 1

            if not self.aborted:
                # sanity check response if no abort before
                sane_refl_b = self._sanity_check_response('b', self.reflexive, type)
                if not sane_refl_b:
                    self.lose = True

        # if no abort proceed
        if not self.aborted:
            # log turn per model request for each probing question
            self.current_turn += 1
            self.log_next_turn()
            # shared probing question P2
            # pose question and get answer while also checking format
            self.qna(self.shared, 'b')
            if type == 'default':
                self.probing_questions_asked_final_b += 1

            if not self.aborted:
                # sanity check response if no abort before
                sane_symm_b = self._sanity_check_response('b', self.shared, type)
                if not sane_symm_b:
                    self.lose = True

        # get pre outcome
        if type == 'pre':
            if not self.aborted and not self.lose:
                probe_success = True
                outcome = 'expected matched: P1 informed, P2 uninformed'
            elif self.lose:
                outcome = 'expected not matched'
            if not self.aborted:
                # log pre outcome
                action = {'type': 'pre sanity check', 'content': outcome}
                self.log_event(from_='GM', to='GM', action=action)
            # log end of pre sanity check
            action = {'type': 'pre sanity check', 'content': 'end pre sanity check'}
            self.log_event(from_='GM', to='GM', action=action)

        elif type == 'default':
            # win game if not aborted and not lost through unexpected behaviour
            if not self.aborted and not self.lose:
                probe_success = True
                self.success = True
                if self.discussed:
                    outcome = 'expected matched: symmetry in knowledge of fact signaled by P1 and P2 matches indication of discussion of fact by P1 and P2'
                elif self.not_discussed:
                    outcome = 'expected matched: asymmetry in knowledge of fact signaled by P1 and P2 matches no indication of discussion of fact by P1 and P2'
            elif self.lose:
                outcome = 'expected not matched'
            if not self.aborted:
                # log final outcome
                action = {'type': 'final probing', 'content': outcome}
                self.log_event(from_='GM', to='GM', action=action)
            # log end of final probing
            action = {'type': 'final probing', 'content': 'end final probing'}
            self.log_event(from_='GM', to='GM', action=action)

        return probe_success

        # # pre sanity check
        # if self.proceed():
        #     if type == 'pre':
        #         # conduct pre sanity check
        #         pre_sane = self._pre_sanity_check()
        #         # if not pre sane lose
        #         if not pre_sane:
        #             self.lose = True
        #         # if pre sane probe success
        #         else:
        #             probe_success = True

        #     # final probing
        #     elif type == 'default':
        #         # conduct final probing check
        #         expected = self._final_probing()
        #         # if not expected lose
        #         if not expected:
        #             self.lose = True
        #         # if expected final probing success and game won
        #         else:
        #             probe_success = True
        #             self.success = True
    
        #         # # if not game lost through unexpected behaviour in final probing, game won
        #         # if not self.lose and expected:
        #         #     probe_success = True

        #         #     action = {'type': 'coherence check', 'content': 'start coherence check'}
        #         #     self.log_event(from_='GM', to='GM', action=action)

        #         #     self.coherent = self._coherence_check()

        #         #     action = {'type': 'coherence check', 'content': 'end coherence check'}
        #         #     self.log_event(from_='GM', to='GM', action=action)

        #         # if self.coherent:
        #         #     self.success = True
        #         # else:
        #         #     self.lose = True

        # # log end of probing
        # if type == 'default':
        #     action = {'type': 'final probing', 'content': 'end final probing'}
        # elif type == 'pre':
        #     action = {'type': 'pre sanity check', 'content': 'end pre sanity check'}
        # self.log_event(from_='GM', to='GM', action=action)

        # return probe_success

    # def probe(self, type='default'):
    #     """Probe for common ground beliefs of players."""
    #     probe_success = False

    #     if self.proceed():
    #         # log turn per model request for each probing question
    #         self.current_turn += 1
    #         self.log_next_turn()

    #         # log start of probing
    #         if type == 'default':
    #             action = {'type': 'final probing', 'content': 'start final probing'}
    #         elif type == 'pre':
    #             action = {'type': 'pre sanity check', 'content': 'start pre sanity check'}
    #         elif type == 'post':
    #             action = {'type': 'post sanity check', 'content': 'start post sanity check'}
    #         self.log_event(from_='GM', to='GM', action=action)

    #         # post sanity check
    #         if type == 'post':
    #             # begin with P1
    #             # discussion question P1
    #             # pose question and get answer while also checking format
    #             self.qna(self.discussed_q, 'a')
    #             self.probing_questions_asked_a += 1
                
    #             # if no abort move to P2
    #             if self.proceed():
    #                 # log turn per model request for each probing question
    #                 self.current_turn += 1
    #                 self.log_next_turn()
    #                 # discussion question P2
    #                 # pose question and get answer while also checking format
    #                 self.qna(self.discussed_q, 'b')
    #                 self.probing_questions_asked_b += 1

    #                 # if no abort conduct post sanity check
    #                 if self.proceed():
    #                     post_sane = self._post_sanity_check()
    #                     # if not sane lose
    #                     if not post_sane:
    #                         self.lose = True
    #                     # if sane probe success
    #                     else:
    #                         probe_success = True

    #                     # log end of post sanity check probing
    #                     action = {'type': 'post sanity check', 'content': 'end post sanity check'}
    #                     self.log_event(from_='GM', to='GM', action=action)

    #                     return probe_success

    #         else:
    #             # reflexive probing question P1
    #             # pose question and get answer
    #             self.qna(self.reflexive, 'a')
    #             self.probing_questions_asked_a += 1

    #     # if no abort in last question proceed
    #     if self.proceed():
    #         # log turn per model request for each probing question
    #         self.current_turn += 1
    #         self.log_next_turn()
    #         # shared probing question P1
    #         # pose question and get answer while also checking format
    #         self.qna(self.shared, 'a')
    #         self.probing_questions_asked_a += 1

    #     # if no abort in last question proceed
    #     if self.proceed():
    #         # log turn per model request for each probing question
    #         self.current_turn += 1
    #         self.log_next_turn()
    #         # reflexive probing question P2
    #         # pose question and get answer while also checking format
    #         self.qna(self.reflexive, 'b')
    #         self.probing_questions_asked_b += 1

    #     # if no abort in last question proceed
    #     if self.proceed():
    #         # log turn per model request for each probing question
    #         self.current_turn += 1
    #         self.log_next_turn()
    #         # shared probing question P2
    #         # pose question and get answer while also checking format
    #         self.qna(self.shared, 'b')
    #         self.probing_questions_asked_b += 1

    #     # pre sanity check
    #     if self.proceed():
    #         if type == 'pre':
    #             # conduct pre sanity check
    #             pre_sane = self._pre_sanity_check()
    #             # if not pre sane lose
    #             if not pre_sane:
    #                 self.lose = True
    #             # if pre sane probe success
    #             else:
    #                 probe_success = True

    #         # final probing
    #         elif type == 'default':
    #             # conduct final probing check
    #             expected = self._final_probing()
    #             # if not expected lose
    #             if not expected:
    #                 self.lose = True
    #             # if expected final probing success and game won
    #             else:
    #                 probe_success = True
    #                 self.success = True
    
    #             # # if not game lost through unexpected behaviour in final probing, game won
    #             # if not self.lose and expected:
    #             #     probe_success = True

    #             #     action = {'type': 'coherence check', 'content': 'start coherence check'}
    #             #     self.log_event(from_='GM', to='GM', action=action)

    #             #     self.coherent = self._coherence_check()

    #             #     action = {'type': 'coherence check', 'content': 'end coherence check'}
    #             #     self.log_event(from_='GM', to='GM', action=action)

    #             # if self.coherent:
    #             #     self.success = True
    #             # else:
    #             #     self.lose = True

    #     # log end of probing
    #     if type == 'default':
    #         action = {'type': 'final probing', 'content': 'end final probing'}
    #     elif type == 'pre':
    #         action = {'type': 'pre sanity check', 'content': 'end pre sanity check'}
    #     self.log_event(from_='GM', to='GM', action=action)

    #     return probe_success


    # def probe(self, type='default'):
    #     """Probe for common ground beliefs of players."""
    #     probe_success = False
    #     probe_results = {'a': {self.reflexive: '', self.shared: ''}, 
    #                      'b': {self.reflexive: '', self.shared: ''}}

    #     if self.aborted == False:
    #         # log turn per model request for each probing question
    #         self.current_turn += 1
    #         self.log_next_turn()

    #         # log start of probing
    #         if type == 'default':
    #             action = {'type': 'final probing', 'content': 'start final probing'}
    #         elif type == 'pre':
    #             action = {'type': 'pre sanity check', 'content': 'start pre sanity check'}
    #         elif type == 'post':
    #             action = {'type': 'post sanity check', 'content': 'start post sanity check'}
    #         self.log_event(from_='GM', to='GM', action=action)

    #         if type == 'post':
    #             # discussion question P1
    #             # pose question and get answer
    #             disc_question_a, disc_response_a, disc_format_a = self._probing_loop(self.discussed_q, 'a')
    #             probe_results['a'][self.discussed_q] = disc_response_a
    #             # if invalid format abort
    #             if not disc_format_a:
    #                 self.aborted = True
    #             # if valid format complete turn
    #             else:
    #                 self.complete_turns += 1
                
    #             if self.aborted == False:
    #                 # log turn per model request for each probing question
    #                 self.current_turn += 1
    #                 self.log_next_turn()

    #                 # discussion question P2
    #                 # pose question and get answer
    #                 disc_question_b, disc_response_b, disc_format_b = self._probing_loop(self.discussed_q, 'b')
    #                 probe_results['b'][self.discussed_q] = disc_response_b
    #                 # if invalid format abort
    #                 if not disc_format_b:
    #                     self.aborted = True
    #                 # if valid format complete turn
    #                 else:
    #                     self.complete_turns += 1

    #                 # conduct post sanity check
    #                 disc_sane = self._post_sanity_check(probe_results, disc_format_a and disc_format_b)
    #                 # if not sane lose
    #                 if not disc_sane:
    #                     self.lose = True
    #                 # if sane probe success
    #                 else:
    #                     probe_success = True

    #                 action = {'type': 'post sanity check', 'content': 'end post sanity check'}
    #                 self.log_event(from_='GM', to='GM', action=action)

    #                 return probe_results, probe_success

    #         else:
    #             # reflexive probing question P1
    #             # pose question and get answer
    #             refl_question_a, refl_response_a, refl_format_a = self._probing_loop(self.reflexive, 'a')
    #             probe_results['a'][self.reflexive] = refl_response_a
    #             # if invalid format abort
    #             if not refl_format_a:
    #                 self.aborted = True
    #             # if valid format complete turn
    #             else:
    #                 self.complete_turns += 1

    #     if self.aborted == False:
    #         # log turn per model request for each probing question
    #         self.current_turn += 1
    #         self.log_next_turn()

    #         # shared probing question P1
    #         # pose question and get answer
    #         shrd_question_a, shrd_response_a, shrd_format_a = self._probing_loop(self.shared, 'a')
    #         probe_results['a'][self.shared] = shrd_response_a
    #         # if invalid format abort
    #         if not shrd_format_a:
    #             self.aborted = True
    #         # if valid format complete turn
    #         else:
    #             self.complete_turns += 1

    #     if self.aborted == False:
    #         # log turn per model request for each probing question
    #         self.current_turn += 1
    #         self.log_next_turn()

    #         # reflexive probing question P2
    #         # pose question and get answer
    #         refl_question_b, refl_response_b, refl_format_b = self._probing_loop(self.reflexive, 'b')
    #         probe_results['b'][self.reflexive] = refl_response_b
    #         # if invalid format abort
    #         if not refl_format_b:
    #             self.aborted = True
    #         # if valid format complete turn
    #         else:
    #             self.complete_turns += 1

    #     if self.aborted == False:
    #         # log turn per model request for each probing question
    #         self.current_turn += 1
    #         self.log_next_turn()

    #         # shared probing question P2
    #         # pose question and get answer
    #         shrd_question_b, shrd_response_b, shrd_format_b = self._probing_loop(self.shared, 'b')
    #         probe_results['b'][self.shared] = shrd_response_b
    #         # if invalid format abort
    #         if not shrd_format_b:
    #             self.aborted = True
    #         # if valid format complete turn
    #         else:
    #             self.complete_turns += 1

    #     if self.aborted == False:
    #         if type == 'pre':
    #             # conduct pre sanity check
    #             pre_sane = self._pre_sanity_check(probe_results, refl_format_a and shrd_format_a and refl_format_b and shrd_format_b)
    #             # if not pre sane lose
    #             if not pre_sane:
    #                 self.lose = True
    #             # if pre sane probe success
    #             else:
    #                 probe_success = True
    #         elif type == 'default':
    #             # conduct final probing check
    #             expected = self._final_probing(probe_results, refl_format_a and shrd_format_a and refl_format_b and shrd_format_b)

    #             # if not game lost through unexpected behaviour in final probing, check for coherence
    #             if not self.lose and expected:
    #                 action = {'type': 'coherence check', 'content': 'start coherence check'}
    #                 self.log_event(from_='GM', to='GM', action=action)

    #                 self.coherent = self._coherence_check()

    #                 action = {'type': 'coherence check', 'content': 'end coherence check'}
    #                 self.log_event(from_='GM', to='GM', action=action)

    #             if self.coherent:
    #                 self.success = True
    #             else:
    #                 self.lose = True
        
    #     # if type == 'default' and not self.aborted:
    #     #     # default or final probing outcome check in set_probing_outcome
    #     #     probe_success = True

    #     # log end of probing
    #     if type == 'default':
    #         action = {'type': 'final probing', 'content': 'end final probing'}
    #     elif type == 'pre':
    #         action = {'type': 'pre sanity check', 'content': 'end pre sanity check'}
    #     self.log_event(from_='GM', to='GM', action=action)

    #     return probe_results, probe_success

    def _final_probing(self):
        """Determine final probing outcome."""
        expected = True

        if self.discussed:
            expectations_a = self.expectations['a']['final']['discussed']
            expectations_b = self.expectations['b']['final']['discussed']
            content = 'expected matched: symmetry in knowledge of fact signaled by P1 and P2 matches indication of discussion of fact by P1 and P2'
        elif self.not_discussed:
            expectations_a = self.expectations['a']['final']['not_discussed']
            expectations_b = self.expectations['b']['final']['not_discussed']
            content = 'expected matched: asymmetry in knowledge of fact signaled by P1 and P2 matches no indication of discussion of fact by P1 and P2'

        if self.probe_results['a'][self.reflexive] in expectations_a['refl']:
            self.expected_responses_final_a += 1
            self.expected_responses_a += 1
        else:
            expected = False

        if self.probe_results['a'][self.shared] in expectations_a['symm']:
            self.expected_responses_final_a += 1
            self.expected_responses_a += 1
        else:
            expected = False

        if self.probe_results['b'][self.reflexive] in expectations_b['refl']:
            self.expected_responses_final_b += 1
            self.expected_responses_b += 1
        else:
            expected = False

        if self.probe_results['b'][self.shared] in expectations_b['symm']:
            self.expected_responses_final_b += 1
            self.expected_responses_b += 1
        else:
            expected = False
                        
        if not expected:
            content = 'expected not matched'

        action = {'type': 'final probing', 'content': content}
        self.log_event(from_='GM', to='GM', action=action)
        logger.info(content)

        return expected

    # def _final_probing(self):
    #     """Determine final probing outcome."""
    #     expected = False
    #     content = 'expected not matched'

    #     if self.probe_results['a'][self.reflexive] == self.yes:
    #         self.expected_responses_final += 1
    #         if self.probe_results['a'][self.shared] == self.yes:
    #             self.expected_responses_final += 1
    #             if self.probe_results['b'][self.reflexive] == self.yes:
    #                 self.expected_responses_final += 1
    #                 if self.probe_results['b'][self.shared] == self.yes:
    #                     self.expected_responses_final += 1
    #                     expected = True
    #                     self.case1 = True
    #                     content = 'expected matched: knowledge signaled by P2 after same signal by informed P1'
    #                 else:
    #                     self.lose = True
    #             else:
    #                 self.lose = True
    #         elif self.probe_results['a'][self.shared] in (self.no, self.dunno):
    #             self.expected_responses_final += 1
    #             if self.probe_results['b'][self.reflexive] in (self.no, self.dunno):
    #                 self.expected_responses_final += 1
    #                 if self.probe_results['b'][self.shared] in (self.no, self.dunno):
    #                     self.expected_responses_final += 1
    #                     expected = True
    #                     self.case2 = True
    #                     content = 'expected matched: no knowledge signaled by uninformed P2 after same signal by informed P1'
    #                 else:
    #                     self.lose = True
    #             else:
    #                 self.lose = True
    #     else:
    #         self.lose = True
        
    #     action = {'type': 'final probing', 'content': content}
    #     self.log_event(from_='GM', to='GM', action=action)
    #     logger.info(content)

    #     return expected

    # def _final_probing(self, probe_results: dict, probe_success: bool):
    #     """Determine final probing outcome."""
    #     expected = False

    #     # if probe failed then abort game without outcome
    #     if probe_success == False:
    #         self.aborted = True
    #         # content = 'final probing failed'
    #         # action = {'type': 'final probing', 'content': content}
    #         # self.log_event(from_='GM', to='GM', action=action)
    #         # logger.info(content)
    #         return

    #     else:
    #         content = 'expected not matched'

    #         if probe_results['a'][self.reflexive] == self.yes:
    #             if probe_results['a'][self.shared] == self.yes:
    #                 if probe_results['b'][self.reflexive] == self.yes:
    #                     if probe_results['b'][self.shared] == self.yes:
    #                         expected = True
    #                         self.case1 = True
    #                         content = 'expected matched: knowledge signaled by P2 after same signal by informed P1'
    #                     else:
    #                         self.lose = True
    #                 else:
    #                     self.lose = True
    #             elif probe_results['a'][self.shared] in (self.no, self.dunno):
    #                 if probe_results['b'][self.reflexive] in (self.no, self.dunno):
    #                     if probe_results['b'][self.shared] in (self.no, self.dunno):
    #                         expected = True
    #                         self.case2 = True
    #                         content = 'expected matched: no knowledge signaled by uninformed P2 after same signal by informed P1'
    #                     else:
    #                         self.lose = True
    #                 else:
    #                     self.lose = True
    #         else:
    #             self.lose = True
            
    #         action = {'type': 'final probing', 'content': content}
    #         self.log_event(from_='GM', to='GM', action=action)
    #         logger.info(content)

    #         return expected

    # def _coherence_check(self):
    #     """check whether post sanity check matched final probing outcome"""
    #     content = 'incoherence: signals about player knowledge and player discussion incoherent'
    #     coherent = False

    #     if self.discussed and self.case1:
    #         coherent = True
    #         content = 'coherence: signals about player knowledge and player discussion coherent (discussed)'

    #     elif self.not_discussed and self.case2:
    #         coherent = True
    #         content = 'coherence: signals about player knowledge and player discussion coherent (not discussed)'

    #     action = {'type': 'coherence check', 'content': content}
    #     self.log_event(from_='GM', to='GM', action=action)
    #     logger.info(content)

    #     return coherent


    def _pre_sanity_check(self):
        """Pre Conversation Sanity Check"""
        sane = True
        content = 'expected matched: P1 informed, P2 uninformed'

        expectations_a = self.expectations['a']['pre']
        expectations_b = self.expectations['b']['pre']

        # action = {'type': 'pre sanity check', 'content': self.probe_results['a'][self.reflexive] + expectations_a['refl']}
        # self.log_event(from_='GM', to='GM', action=action)

        if self.probe_results['a'][self.reflexive] in expectations_a['refl']:
            self.expected_responses_a += 1
        else:
            sane = False

        if self.probe_results['a'][self.shared] in expectations_a['symm']:
            self.expected_responses_a += 1
        else:
            sane = False

        if self.probe_results['b'][self.reflexive] in expectations_b['refl']:
            self.expected_responses_b += 1
        else:
            sane = False

        if self.probe_results['b'][self.shared] in expectations_b['symm']:
            self.expected_responses_b += 1
        else:
            sane = False
        
        if not sane:
            content = 'expected not matched'

        action = {'type': 'pre sanity check', 'content': content}
        self.log_event(from_='GM', to='GM', action=action)

        return sane


    # def _pre_sanity_check(self):
    #     """Pre Conversation Sanity Check"""
    #     sane = False
    #     content = 'failed'
    #     action = {'type': 'pre sanity check', 'content': 'expected not matched'}

    #     if not self.aborted and not self.lose:
    #         if self.probe_results['a'][self.reflexive] == self.yes:
    #             if self.probe_results['a'][self.shared] in (self.no, self.dunno):
    #                 # P1 has passed pre
    #                 if self.probe_results['b'][self.reflexive] in (self.no, self.dunno):
    #                     if self.probe_results['b'][self.shared] == self.dunno:
    #                         # P2 has passed pre
    #                         content = 'successful'
    #                         sane = True
    #                         action = {'type': 'pre sanity check', 'content': 'expected matched: P1 informed, P2 uninformed'}
    #                         self.log_event(from_='GM', to='GM', action=action)
    #                     else:
    #                         # P2 fails pre shared
    #                         self.lose = True
    #                 else:
    #                     # P2 fails pre reflexive
    #                     self.lose = True 
    #             else:
    #                 # P1 fails pre shared
    #                 self.lose = True
    #         else:
    #         # P1 fails pre reflexive
    #             self.lose = True

    #     if not sane:
    #         self.log_event(from_='GM', to='GM', action=action)
    #         logger.info(content)

    #     return sane


    # def _pre_sanity_check(self, probe_results: dict, probe_success: bool):
    #     """Pre Conversation Sanity Check"""
    #     sane = False
    #     content = 'failed'
    #     action = {'type': 'pre sanity check', 'content': 'expected not matched'}

    #     if probe_success:
    #         if probe_results['a'][self.reflexive] == self.yes:
    #             if probe_results['a'][self.shared] in (self.no, self.dunno):
    #                 # P1 has passed pre
    #                 if probe_results['b'][self.reflexive] in (self.no, self.dunno):
    #                     if probe_results['b'][self.shared] == self.dunno:
    #                         # P2 has passed pre
    #                         content = 'successful'
    #                         sane = True
    #                         action = {'type': 'pre sanity check', 'content': 'expected matched: P1 informed, P2 uninformed'}
    #                         self.log_event(from_='GM', to='GM', action=action)
    #                     else:
    #                         # P2 fails pre shared
    #                         self.lose = True
    #                 else:
    #                     # P2 fails pre reflexive
    #                     self.lose = True 
    #             else:
    #                 # P1 fails pre shared
    #                 self.lose = True
    #         else:
    #         # P1 fails pre reflexive
    #             self.lose = True

    #     if not sane:
    #         self.log_event(from_='GM', to='GM', action=action)
    #         logger.info(content)

    #     return sane

    def _post_sanity_check(self):
        """Post Conversation Sanity Check"""
        sane = False
        content = 'expected not matched: no matching indications of discussion of fact by P1 and P2'
        discussed_a = False
        discussed_b = False

        if self.probe_results['a'][self.discussed_q] == self.yes:
            discussed_a = True
        if self.probe_results['b'][self.discussed_q] == self.yes:
            discussed_b = True
        
        # only expectation during post sanity check is matching signals about discussion of fact
        if discussed_a == discussed_b:
            sane = True
            self.expected_responses_a += 1
            self.expected_responses_b += 1

            # P1 and P2 signal discussion of fact 
            if discussed_a and discussed_b:
                self.discussed = True
                content = 'expected matched: discussion of fact indicated by both P1 and P2'

            # P1 and P2 signal no discussion of fact
            elif not discussed_a and not discussed_b:
                self.not_discussed = True
                content = 'expected matched: no discussion of fact indicated by both P1 and P2'

        action = {'type': 'post sanity check', 'content': content}
        self.log_event(from_='GM', to='GM', action=action)

        return sane

    # def _post_sanity_check(self, probe_results: dict, probe_success: bool):
    #     """Post Conversation Sanity Check"""
    #     sane = False
    #     content = 'failed'
    #     discussed_a = False
    #     discussed_b = False

    #     if probe_success and not self.aborted and not self.lose:
    #         if probe_results['a'][self.discussed_q] == self.yes:
    #             discussed_a = True
    #         if probe_results['b'][self.discussed_q] == self.yes:
    #             discussed_b = True
        
    #     if discussed_a == discussed_b:
    #         sane = True
    #         if discussed_a and discussed_b:
    #             self.discussed = True
    #             action = {'type': 'post sanity check', 'content': 'expected matched: discussion signaled by both P1 and P2'}
    #             self.log_event(from_='GM', to='GM', action=action)
    #         elif not discussed_a and not discussed_b:
    #             self.not_discussed = True
    #             action = {'type': 'post sanity check', 'content': 'expected matched: no discussion signaled by both P1 and P2'}
    #             self.log_event(from_='GM', to='GM', action=action)

    #     else:
    #         self.lose = True
    #         action = {'type': 'post sanity check', 'content': 'expected not matched: no matching discussion signals by P1 and P2'}
    #         self.log_event(from_='GM', to='GM', action=action)

    #     return sane

    def _parse_probing_response(self, response: str) -> str:
        """Extract parsed answer in probing turn."""
        if (not response.startswith(self.aside.strip())
            or self._has_continuation(response)):
            return INVALID

        clean_response = self._filter_tag(response, self.aside.strip())
        if clean_response.lower() == self.yes:
            return self.yes
        if clean_response.lower() == self.no:
            return self.no
        if clean_response.lower() == self.dunno:
            return self.dunno

        logger.warning(NOT_PARSED)
        return INVALID

    @staticmethod
    def _filter_tag(answer: str, tag: str) -> str:
        """Remove a tag from a utterance."""
        filtered = answer.replace(tag, '')
        return filtered.strip()

    @classmethod
    def applies_to(cls, game_name: str) -> bool:
        return game_name == GAME_NAME

    def _has_continuation(self, response: str) -> bool:
        """Return True if the response continues after what is needed."""
        # if the answer contains a line break with some continuation after it,
        # we consider it to be an invalid response
        # we strip first to account for cases where it ends in one or many \n
        # without producing anything after it, and then check if the remaining
        # text still contains a line break
        if '\n' in response.strip('\n'):
            return True
        return False

    def play(self) -> None:
        """Play the game until end."""
        # pre sanity check
        pre_success = self.probe(type='pre')

        # conversation if pre sane
        if pre_success and self.proceed():
            start_convo_logged = False
            alternation = False
            # turn-taking for player dialogue
            while self.proceed_convo():
                self.current_turn += 1
                self.current_turn_convo += 1
                # log new turns upon start
                self.log_next_turn()

                # start of convo and fixing dialogue history
                if not start_convo_logged:
                    action = {'type': 'info', 'content': 'start conversation'}
                    self.log_event(from_='GM', to='GM', action=action)
                    start_convo_logged = True
                if not alternation:
                    # append a "fake" turn to avoid adjacent user turns in their history
                    self.player_a.history.append({'role': 'user', 'content': "Let's go!"})
                    alternation = True

                # turn P1
                if self.current_turn_convo % 2 == 1:
                    self.turn_convo('a', 'b')
                # turn P2
                else:
                    self.turn_convo('b', 'a')
                self.complete_turns += 1
                self.complete_turns_convo += 1

            # end of conversation
            if self.complete_turns_convo == (self.n_turns_convo * 2):
                # append a "fake" turn to avoid adjacent user turns in their history
                self.player_a.history.append({'role': 'assistant', 'content': "Aha."})
                # log a message informing convo is over
                action = {'type': 'info', 'content': 'end conversation'}
                self.log_event(from_='GM', to='GM', action=action)

            # post sanity check after convo and to determine expectations for final probing
            post_success = self.probe(type='post')

            # final probing if post sane and to determine whether game won/lost according to
            # expectations formulated by post sanity check
            if post_success and self.proceed():
                final_success = self.probe()

        # log game ending
        if self.success:
            outcome = 'game won'
        elif self.aborted:
            outcome = 'game aborted'
        elif self.lose:
            outcome = 'game lost'
        action = {'type': 'info', 'content': 'end game: ' + outcome}
        self.log_event(from_='GM', to='GM', action=action)

        # log temporary game variables needed for evaluation
        self._log_eval_assets()

    def initiate(self, prompt_player_a: str, prompt_player_b: str) -> None:
        """Initialise the dialogue history (grounded specific)."""
        # log new turns upon start
        self.log_next_turn()
        # appending initial player messages to their history
        self.player_a.history.append({'role': 'user',
                                      'content': prompt_player_a})
        self.player_b.history.append({'role': 'user',
                                      'content': prompt_player_b})
        # append a "fake" turn to avoid adjacent user turns in their history
        self.player_a.history.append({'role': 'assistant', 'content': "Ok."})
        self.player_b.history.append({'role': 'assistant', 'content': "Ok."})

        # log prompts for records
        action = {'type': 'send message', 'content': prompt_player_a}
        self.log_event(from_='GM', to='Player 1', action=action)
        action = {'type': 'send message', 'content': prompt_player_b}
        self.log_event(from_='GM', to='Player 2', action=action)

    def proceed_convo(self) -> None:
        """Check if the loop should continue or whether number of turns has been reached."""
        return (self.current_turn_convo < (self.n_turns_convo * 2)
                and not self.aborted
                and not self.lose)

    def _get_utterance(self, player: str) -> str:
        """Get utterance from a player and log it."""
        assert player in ('a', 'b')
        if player == 'a':
            from_pl = 'Player 1'
            player_prompted = self.player_a
        elif player == 'b':
            from_pl = 'Player 2'
            player_prompted = self.player_b

        # make an API call or get programmatic response
        prompt, raw_answer, answer = player_prompted(player_prompted.history,
                                        self.current_turn)
        # add reply to the records
        action = {'type': 'get message', 'content': answer}
        self.log_event(from_=from_pl, to='GM', action=action,
                       call=(copy.deepcopy(prompt), raw_answer))
        # add reply to its own dialogue history
        self._append_utterance(answer, player, 'assistant')

        # increase the number of API requests
        self.request_counts[self.current_turn] += 1

        # increase parsed requests, since all we need is two utterances
        self.parsed_request_counts[self.current_turn] += 1

        return answer

        #     # make API call or get programmatic response from player a
        #     prompt, raw_answer, answer = self.player_a(self.player_a.history,
        #                                                self.current_turn)
        #     # add reply to the records
        #     action = {'type': 'get message', 'content': answer}
        #     self.log_event(from_='Player 1', to='GM', action=action,
        #                    call=(copy.deepcopy(prompt), raw_answer))
        #     # add reply to its own dialogue history
        #     self._append_utterance(answer, 'a', 'assistant')

        # else:
        #     # make an API call or get programmatic response from player b
        #     prompt, raw_answer, answer = self.player_b(self.player_b.history,
        #                                  self.current_turn)
        #     # add reply to the records
        #     action = {'type': 'get message', 'content': answer}
        #     self.log_event(from_='Player 2', to='GM', action=action,
        #                    call=(copy.deepcopy(prompt), raw_answer))
        #     # add reply to its own dialogue history
        #     self._append_utterance(answer, 'b', 'assistant')

        # # increase the number of API requests
        # self.request_counts[self.current_turn] += 1
        # self.own_request_counter += 1
        # # increase parsed requests, since all we need is two utterances
        # self.parsed_request_counts[self.current_turn] += 1

        # return answer
    
    def turn_convo(self, sender, receiver) -> None:
        """Perform one dialogue turn by getting utterance of
           one player."""
        if receiver == 'a':
            to_player = 'Player 1'
        else:
            to_player = 'Player 2'

        # get sender's utterance and add it to its own history as assistant
        message_sender = self._get_utterance(sender)
        # add senders's utterance to receiver's history as user
        self._append_utterance(message_sender, receiver, 'user')
        # also add the reply to the transcript
        action = {'type': 'send message', 'content': message_sender}
        self.log_event(from_='GM', to=to_player, action=action)

    def _append_utterance(self, utterance: str, 
                          player: str, role: str) -> None:
        """Add an utterance to the history of a player."""
        assert player in ('a', 'b')

        if player == 'a':
            self.player_a.history.append({'role': role, 'content': utterance})
        else:
            self.player_b.history.append({'role': role, 'content': utterance})

    def _log_eval_assets(self) -> None:
        """Aux to log variables needed for scoring."""
        # check to log whether fact labels have been mentioned
        utts_a = [utt['content'] for utt in self.player_a.history if utt['role'] == 'assistant']
        utts_b = [utt['content'] for utt in self.player_b.history if utt['role'] == 'assistant']
        fact_mentioned_a = any([label.lower() in utte.lower() for utte in utts_a for label in self.fact_labels])
        fact_mentioned_b = any([label.lower() in utte.lower() for utte in utts_b for label in self.fact_labels])

        # framework metrics
        self.log_key(ms.METRIC_REQUEST_COUNT,
                     self.request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_PARSED,
                     self.parsed_request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_VIOLATED,
                     self.violated_request_counts)
        
        self.log_key(ms.METRIC_LOSE, self.lose)
        self.log_key(ms.METRIC_SUCCESS, self.success)
        self.log_key(ms.METRIC_ABORTED, self.aborted)

        # grounded specific metrics
        self.log_key('Turns complete', self.complete_turns)
        self.log_key('Turn maximum', self.max_turns)

        self.log_key('Requests parsed in probings', self.probing_parsed_counter)
        self.log_key('Requests in probings', self.probing_request_counter)

        self.log_key('Fact mentioned by P1', fact_mentioned_a)
        self.log_key('Fact mentioned by P2', fact_mentioned_b)
        self.log_key('Discussion of fact indicated', self.discussed)

        self.log_key('Probing questions matched by P1', self.expected_responses_a)
        self.log_key('Probing questions matched by P2', self.expected_responses_b)
        self.log_key('Probing questions posed for P1', self.probing_questions_asked_a)
        self.log_key('Probing questions posed for P2', self.probing_questions_asked_b)
        self.log_key('Probing questions matched by P1 in final', self.expected_responses_final_a)
        self.log_key('Probing questions matched by P2 in final', self.expected_responses_final_b)
        self.log_key('Probing questions posed for P1 in final', self.probing_questions_asked_final_a)
        self.log_key('Probing questions posed for P2 in final', self.probing_questions_asked_final_b)

class GroundedGameScorer(GameScorer):
    """Implement mechanisms to score the played game."""
    def __init__(self, experiment: Dict, game_instance: Dict):
        super().__init__(GAME_NAME, experiment, game_instance)
        self.mode = experiment['name']
    
    def compute_scores(self, episode_interactions: Dict) -> None:
        """Compute episode-level and turn-level scores (mandatory)."""
        played_turns = episode_interactions['Turns complete']
        max_turns = episode_interactions['Turn maximum']

        probings_parsed_counter = episode_interactions['Requests parsed in probings']
        probings_requests_counter = episode_interactions['Requests in probings']

        expected_responses_a = episode_interactions['Probing questions matched by P1']
        expected_responses_b = episode_interactions['Probing questions matched by P2']
        posed_questions_a = episode_interactions['Probing questions posed for P1']
        posed_questions_b = episode_interactions['Probing questions posed for P2']

        expected_responses_final_a = episode_interactions['Probing questions matched by P1 in final']
        expected_responses_final_b = episode_interactions['Probing questions matched by P2 in final']
        posed_questions_final_a = episode_interactions['Probing questions posed for P1 in final']
        posed_questions_final_b = episode_interactions['Probing questions posed for P2 in final']

        discussion_indicated = episode_interactions['Discussion of fact indicated']
        fact_mentioned_a = episode_interactions['Fact mentioned by P1']
        fact_mentioned_b = episode_interactions['Fact mentioned by P2']

        # framework metrics
        # turn scores (turn 0 was initial prompts)
        reqs = episode_interactions[ms.METRIC_REQUEST_COUNT][1:]
        p_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_PARSED][1:]
        v_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_VIOLATED][1:]

        for turn in range(0, played_turns):
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT, reqs[turn])
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_PARSED, p_reqs[turn])
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_VIOLATED, v_reqs[turn])
        # episode scores
        self.log_episode_score(ms.METRIC_REQUEST_COUNT, sum(reqs))
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, sum(p_reqs))
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, sum(v_reqs))

        # grounded specific metrics
        # episode scores
        compl_turn_ratio = played_turns / max_turns

        # ratio of expectations met and probing questions asked
        ratio_expected_posed_a = expected_responses_a / posed_questions_a if posed_questions_a != 0 else 0
        ratio_expected_posed_b = expected_responses_b / posed_questions_b if posed_questions_b != 0 else 0
        ratio_expected_posed = (expected_responses_a + expected_responses_b) / (posed_questions_a + posed_questions_b) if (posed_questions_a + posed_questions_b) != 0 else 0

        # ratio of expectations met and probing questions asked in final
        ratio_expected_posed_final_a = expected_responses_final_a / posed_questions_final_a if posed_questions_final_a != 0 else 0
        ratio_expected_posed_final_b = expected_responses_final_b / posed_questions_final_b if posed_questions_final_b != 0 else 0
        ratio_expected_posed_final = (expected_responses_final_a + expected_responses_final_b) / (posed_questions_final_a + posed_questions_final_b) if (posed_questions_final_a + posed_questions_final_b) != 0 else 0
        
        ratio_expected_all_final_a = expected_responses_final_a / (N_PROBINGS_FINAL/2)
        ratio_expected_all_final_b = expected_responses_final_b / (N_PROBINGS_FINAL/2)
        ratio_expected_all_final = (expected_responses_final_a + expected_responses_final_b) / N_PROBINGS_FINAL

        ratio_expected_all_a = expected_responses_a / (N_PROBINGS/2)
        ratio_expected_all_b = expected_responses_b / (N_PROBINGS/2)
        ratio_expected_all = (expected_responses_a + expected_responses_b) / N_PROBINGS

        # game success, lose, abort
        aborted = int(episode_interactions[ms.METRIC_ABORTED])
        lose = int(episode_interactions[ms.METRIC_LOSE]) if not aborted else 0
        success =  1 - lose if not aborted else 0

        # quality score 
        bench_score = ratio_expected_all_final if not aborted else np.nan

        # parsed requests success ratio for probings
        self.log_episode_score('Request success ratio in probings', probings_parsed_counter / probings_requests_counter if probings_requests_counter != 0 else 0)

        self.log_episode_score('Turns completed', played_turns)
        self.log_episode_score('Turn maximum', max_turns)
        self.log_episode_score('Ratio of turns completed', compl_turn_ratio)

        self.log_episode_score('Probing questions matched by P1', expected_responses_a)
        self.log_episode_score('Probing questions matched by P2', expected_responses_b)
        self.log_episode_score('Probing questions posed to P1', posed_questions_a)
        self.log_episode_score('Probing questions posed to P2', posed_questions_b)
        self.log_episode_score('Ratio of probing questions posed and matched by P1', ratio_expected_posed_a)
        self.log_episode_score('Ratio of probing questions posed and matched by P2', ratio_expected_posed_b)
        self.log_episode_score('Ratio of probing questions posed and matched', ratio_expected_posed)

        self.log_episode_score('Probing questions matched by P1 in final', expected_responses_final_a)
        self.log_episode_score('Probing questions matched by P2 in final', expected_responses_final_b)
        self.log_episode_score('Probing questions posed to P1 in final', posed_questions_final_a)
        self.log_episode_score('Probing questions posed to P2 in final', posed_questions_final_b)
        self.log_episode_score('Ratio of probing questions posed and matched by P1 in final', ratio_expected_posed_final_a)
        self.log_episode_score('Ratio of probing questions posed and matched by P2 in final', ratio_expected_posed_final_b)
        self.log_episode_score('Ratio of probing questions posed and matched in final', ratio_expected_posed_final)
        
        self.log_episode_score('Ratio of probing questions matched of final by P1', ratio_expected_all_final_a)
        self.log_episode_score('Ratio of probing questions matched of final by P2', ratio_expected_all_final_b)
        # benchscore but for overview
        self.log_episode_score('Ratio of probing questions matched of final', ratio_expected_all_final)

        self.log_episode_score('Ratio of probing questions matched of game by P1', ratio_expected_all_a)
        self.log_episode_score('Ratio of probing questions matched of game by P2', ratio_expected_all_b)
        self.log_episode_score('Ratio of probing questions matched of game', ratio_expected_all)

        self.log_episode_score('Discussion of fact indicated', discussion_indicated)
        self.log_episode_score('Fact mentioned by P1', fact_mentioned_a)
        self.log_episode_score('Fact mentioned by P2', fact_mentioned_b)

        self.log_episode_score(ms.METRIC_ABORTED, aborted)
        self.log_episode_score(ms.METRIC_LOSE, lose)
        self.log_episode_score(ms.METRIC_SUCCESS, success)
        self.log_episode_score(ms.BENCH_SCORE, bench_score)


# required GameBenchmark child
class GroundedGameBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""
    def __init__(self):
        super().__init__(GAME_NAME)

    # defines whether the game is single player or not
    def is_single_player(self):
        return False

    # add description of game
    def get_description(self):
        return 'A game in which players have a conversation ' \
               'and are then questioned about their self-awareness of knowledge' \
               'in addition to that of their conversation partner.'

    # get game master
    def create_game_master(self,
                           experiment: Dict,
                           player_backends: List[str]
                           ) -> GameMaster:
        return Grounded(experiment, player_backends)

    # get game scorer
    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return GroundedGameScorer(experiment, game_instance)
