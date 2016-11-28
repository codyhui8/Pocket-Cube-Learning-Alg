'''
Cody Hui 1338250
CSE 415, Spring 2016, University of Washington
MDP.py
Instructor: S. Tanimoto, May 2016.

Provides representations for Markov Decision Processes, plus
functionality for running the transitions.

The transition function should be a function of three arguments:
T(s, a, sp), where s and sp are states and a is an action.
The reward function should also be a function of the three same
arguments.  However, its return value is not a probability but
a numeric reward value -- any real number.

operators:  state-space search objects consisting of a precondition
 and deterministic state-transformation function.
 We assume these are in the "QUIET" format used in earlier assignments.

actions:  objects (for us just Python strings) that are 
 stochastically mapped into operators at runtime according 
 to the Transition function.


Status:
 As of May 14 at 11:00 AM:
   Basic methods have been prototyped.

 Implemented the ValueIteration method, the QLearning method, and the extractPolicy method.
 All implemented methods are working.

 Implemented various timing features to show user how much time has elapsed.
'''

import random
import timeit

REPORTING = True


class MDP:
    ACTION_POLICY = {}

    def __init__(self):
        self.known_states = set()
        self.succ = {}  # hash of adjacency lists by state.

    def register_start_state(self, start_state):
        self.start_state = start_state
        self.known_states.add(start_state)

    def register_actions(self, action_list):
        self.actions = action_list

    def register_operators(self, op_list):
        self.ops = op_list

    def register_transition_function(self, transition_function):
        self.T = transition_function

    def register_reward_function(self, reward_function):
        self.R = reward_function

    def register_action_policy(self):
        self.ACTION_POLICY = {'epsilon-greedy': self.greedy_epsilon,
                              'epsilon-soft': self.soft_epsilon}

    def register_known_states(self, ks):
        self.known_states = ks

    def state_neighbors(self, state):
        neighbors = self.succ.get(state, False)
        if neighbors == False:
            neighbors = [op.apply(state) for op in self.ops if op.is_applicable(state)]
            self.succ[state] = neighbors
            self.known_states.update(neighbors)
        return neighbors

    def take_action(self, a, state):
        s = state
        neighbors = self.state_neighbors(s)
        threshold = 0.0
        rnd = random.uniform(0.0, 1.0)
        r = 0
        already_chose = False
        for sp in neighbors:
            threshold += self.T(s, a, sp)
            if threshold > rnd:
                r = self.R(s, a, sp)
                already_chose = True
                break
        self.current_state = sp
        if not REPORTING: print("After action " + a + ", moving to state " + str(sp) + \
                                "; reward is " + str(r))
        if not already_chose:
            return None
        else:
            return sp

    def generateAllStates(self):
        queue = [self.start_state]
        self.current_state = self.start_state
        tic = timeit.default_timer()
        temp_time = tic
        print('Seconds: ' + str(temp_time))
        num_states = 0
        while queue:
            value = queue.pop(0)
            if self.succ.get(value, False) == False:
                neighbors = [op.apply(value) for op in self.ops if op.is_applicable(value)]
                for op in neighbors:
                    self.current_state = op
                    if op not in self.known_states:
                        queue.append(op)
                        self.known_states.add(op)

                        num_states += 1
                        temp_toc = timeit.default_timer()
                        if temp_toc - temp_time >= 10:
                            print('Seconds: ' + "{0:.3f}".format(temp_toc) + ', States: ' + str(num_states))
                            temp_time = temp_toc
                self.succ[value] = neighbors
        toc = timeit.default_timer()
        print('Time Elapsed: ' + str(toc - tic))
        if not REPORTING:
            print("Length: " + str(len(self.known_states)))
            print(self.known_states)
        return self.known_states

    def ValueIterations(self, discount, niterations):
        self.V = dict([(s, 0) for s in self.known_states])
        for i in range(niterations):
            tic = timeit.default_timer()
            temp_V = self.V.copy()
            time_counter = 0
            for s in self.known_states:
                temp_value = []
                neighbors = [op.apply(s) for op in self.ops if op.is_applicable(s)]
                for t in self.actions:
                    temp_sum = 0
                    for j in neighbors:
                        temp_sum += self.T(s, t, j) * (self.R(s, t, j) + discount * temp_V[j])
                    temp_value.append(temp_sum)
                self.V[s] = max(temp_value)
                temp_tic = timeit.default_timer()
                if temp_tic - time_counter >= 10:
                    time_counter = temp_tic
                    print('Seconds: ' + str(temp_tic - tic))
            toc = timeit.default_timer()
            print('Iteration: ' + str(i) + ', Seconds: ' + str(toc - tic))
        return self.V

    def QLearning(self, discount, nEpisodes, epsilon, action_policy, show_state):
        self.QValues = dict()
        for states in self.known_states:
            for action in self.actions:
                self.QValues[(states, action)] = 0
        self.current_policy = self.extractPolicy()
        self.QCount = dict()
        for states in self.known_states:
            for action in self.actions:
                self.QCount[(states, action)] = 0

        actions = self.actions
        for i in range(nEpisodes):
            state = self.start_state

            extracted_policy = self.extractPolicy()

            neighbors = self.state_neighbors(state)
            q_val_copy = self.QValues.copy()

            keep_instructions = False
            if i % show_state == 0 and i != 0:
                keep_instructions = True
            instructions = []
            tic = timeit.default_timer()
            time_counter = tic
            while len(neighbors) > 0:
                policy = self.ACTION_POLICY.get(action_policy)(epsilon, extracted_policy, state, actions)

                sp = self.take_action(policy, state)
                self.QCount[(state, policy)] += 1

                if sp != None:
                    alpha = 1 / self.QCount[(state, policy)]
                    reward = self.R(state, policy, sp)
                    max_q = max([q_val_copy[(sp, a)] for a in self.actions])
                    sample = reward + discount * max_q
                    self.QValues[(state, policy)] = (1 - alpha) * self.QValues[(state, policy)] + alpha * sample

                    state = sp

                    neighbors = self.state_neighbors(sp)
                    if keep_instructions:
                        instructions.append([(state, policy), self.QValues[(state, policy)]])

                temp_tic = timeit.default_timer()
                if temp_tic - time_counter >= 10:
                    print('Seconds: ' + str(temp_tic - tic))
                    time_counter = temp_tic

            if keep_instructions:
                for j in instructions:
                    print("State: " + str(j[0][0]) + ", Policy: " + str(j[0][1]) + ", Q-Value: " + str(j[1]))

            toc = timeit.default_timer()
            print("Iteration: " + str(i + 1) + ", Time Elapsed: " + str(toc - tic))

        return (self.QValues, extracted_policy)

    def extractPolicy(self):
        self.OptimalPolicy = dict()
        already_set = False
        actions = self.actions[0:(len(self.actions) - 1)]
        for states in self.known_states:
            temp_action = [self.QValues.get((states, action)) for action in actions]
            return_action = ''

            max_value = max(temp_action)
            for action in actions:
                if self.QValues.get((states, action)) == max_value and not already_set:
                    return_action = action
                    already_set = True
            self.OptimalPolicy[states] = return_action
            already_set = False
        return self.OptimalPolicy

    def greedy_epsilon(self, epsilon, extracted_policy, state, actions):
        greedy = random.random()
        policy = ''
        if greedy > epsilon:
            policy = extracted_policy[state]
        else:
            policy = random.choice(actions)
        return policy

    def soft_epsilon(self, epsilon, extracted_policy, state, actions):
        greedy = random.random()
        policy = ''
        if greedy < epsilon:
            policy = extracted_policy[state]
        else:
            policy = random.choice(actions)
        return policy
