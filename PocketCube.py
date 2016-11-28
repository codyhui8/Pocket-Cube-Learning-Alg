'''
Cody Hui 1338250
CSE 415, Spring 2016, University of Washington
PocketCube.py
Instructor:  S. Tanimoto.
'''

# <METADATA>
QUIET_VERSION = "0.1"
PROBLEM_NAME = "Pocket Cube"
PROBLEM_VERSION = "0.1"
PROBLEM_AUTHORS = ['C. Hui']
PROBLEM_CREATION_DATE = "23-MAY-2016"
PROBLEM_DESC = ''
# </METADATA>

import copy
import random
import math
import csv
import MDP


# <COMMON_CODE>
class PocketCube:
    NUM_SIDE = 0
    NUM_EPISODES = 0
    NUMBER_COMBINATIONS = []
    ACTION_POLICY = "epsilon-greedy"
    MAX_MOVES = 100
    EPSILON = 0.2
    SHOW_STATE = 50

    GOAL_STATE = [[1] * 4, [2] * 4, [3] * 4, [2] * 4, [3] * 4, [1] * 4]

    INITIAL_STATE = GOAL_STATE

    # <CUBE ENCODING>
    FACE_TO_EDGE = [[[], [0, 1], [0, 1], [0, 1], [0, 1], []],  # 0
                    [[0, 2], [], [0, 2], [], [3, 1], [0, 2]],  # 1
                    [[2, 3], [3, 1], [], [0, 2], [], [1, 0]],  # 2
                    [[3, 1], [], [3, 1], [], [0, 2], [3, 1]],  # 3
                    [[1, 0], [0, 2], [], [3, 1], [], [2, 3]],  # 4
                    [[], [2, 3], [2, 3], [2, 3], [2, 3], []]]  # 5

    FTE_ORDER = [[1, 2, 3, 4],  # 0
                 [0, 4, 5, 2],  # 1
                 [1, 5, 3, 0],  # 2
                 [2, 5, 4, 0],  # 3
                 [0, 3, 5, 1],  # 4
                 [1, 4, 3, 2]]  # 5

    FACE_ROTATE_CORNER = [[0, 2, 3, 1],  # 0
                          [0, 4, 6, 2],  # 1
                          [2, 6, 7, 3],  # 2
                          [1, 3, 7, 5],  # 3
                          [0, 1, 5, 4],  # 4
                          [6, 4, 5, 7]]  # 5

    FACE_ROTATE = [2, 0, 3, 1]

    FIRST_LAYER = [[None, 3, 2, None, None, 0],
                   [None, None, 3, 2, None, 1],
                   [None, 2, None, None, 3, 2],
                   [None, None, None, 3, 2, 3]]

    # </CUBE ENCODING>

    # <INITIALIZING>
    def __init__(self):
        return

    def init_numside(self, num_side):
        self.NUM_SIDE = num_side

    def init_numcolor(self, num_color):
        self.NUM_COLOR = num_color

    def init_numepisodes(self, num_episodes):
        self.NUM_EPISODES = num_episodes

    def init_actions(self, actions):
        self.NUMBER_COMBINATIONS = actions

    def init_action_policy(self, action_policy):
        self.ACTION_POLICY = action_policy

    def init_max_moves(self, max_moves):
        self.MAX_MOVES = max_moves

    def init_epsilon(self, epsilon):
        self.EPSILON = epsilon

    def init_show_state(self, show_state):
        self.SHOW_STATE = show_state
    # </INITIALIZING>

    # <CUBE MANIPULATION>
    def move(self, s, step):
        if s == self.GOAL_STATE or s == 'Solved':
            return 'Solved'
        news = self.copy_state(s)  # start with a deep copy.
        if step > 0:
            news[step - 1] = self.rotate_face(news[step - 1], "clockwise")
        else:
            news[abs(step) - 1] = self.rotate_face(news[abs(step) - 1], "cc")
        news[0:6] = self.rotate_edges(news[0:6], step)
        # news[6] = rotate_cubies(news[6], step)
        return news  # return new state

    # def rotate_cubies(self, s, step):
    #     news = s.copy()
    #     corner_rotate = self.FACE_ROTATE_CORNER[abs(step) - 1]
    #     temp = news.copy()
    #     for i in range(4):
    #         if step > 0:
    #             news[corner_rotate[(i - 1) % 4]] = temp[corner_rotate[i]]
    #         else:
    #             news[corner_rotate[(i + 1) % 4]] = temp[corner_rotate[i]]
    #     return news

    def rotate_face(self, face, direction):
        temp_value = [None] * 4
        if direction == "clockwise":
            for i in range(4):
                temp_value[i] = face[self.FACE_ROTATE[i]]
        else:
            for i in range(4):
                temp_value[i] = face[self.FACE_ROTATE[-1 - i]]
        return temp_value

    def rotate_edges(self, s, step):
        temp_copy = self.copy_state(s[0:6])
        face = abs(step) - 1
        order = self.FTE_ORDER[face]
        edges = self.FACE_TO_EDGE[face]
        temp = self.copy_state(temp_copy)
        for i in list(range(0, 4)):
            for j in range(2):
                if step > 0:
                    temp_copy[order[(i - 1) % 4]][edges[order[(i - 1) % 4]][j]] = temp[order[i]][edges[order[i]][j]]
                else:
                    temp_copy[order[(i + 1) % 4]][edges[order[(i + 1) % 4]][j]] = temp[order[i]][edges[order[i]][j]]
        return temp_copy

    # </CUBE MANIPULATION>

    # <MDP FUNCTIONS>
    def R(self, s, a, sp):
        s = self.change_set_to_list(s)
        if sp == 'Solved':
            return 1000000
        result = self.check_first_layer(s)
        for i in range(6):
            for j in range(4):
                if s[i][j] == i:
                    result += 1
        return result * 100

    def T(self, s, a, sp):
        if self.move(self.change_set_to_list(s), a) == sp:
            return 1.1 - len(self.NUMBER_COMBINATIONS) * 0.1
        elif (1.1 - len(self.NUMBER_COMBINATIONS) * 0.1) == 1:
            return 0
        else:
            return 0.1

    # </MDP FUNCTIONS>

    # <HELPER METHODS>
    def check_first_layer(self, s):
        return_value = 0
        for pieces in self.FIRST_LAYER:
            piece_correct = True
            for i in range(6):
                if pieces[i] != None:
                    if s[i][pieces[i]] != self.GOAL_STATE[i][pieces[i]]:
                        piece_correct = False
            if piece_correct:
                return_value += 2500
        return return_value

    def HASHCODE(self, s):
        txt = ''
        for i in s:
            for j in i:
                txt += str(j)
        return txt

    def copy_state(self, s):
        news = []
        for i in range(6):
            temp = []
            for j in range(4):
                temp.append(s[i][j])
            news.append(temp)
        return news

    def change_state_to_set(self, s):
        if s == 'Solved':
            return s
        return self.HASHCODE(s)

    def change_set_to_list(self, s):
        if s == 'Solved':
            return s
        return_value = []
        count = 0
        for i in range(6):
            temp_value = []
            for j in range(4):
                temp_value.append(int(s[(i * 4) + j]))
                count += 1
            return_value.append(temp_value)
        # temp_value = []
        # for i in range(count,len(s)):
        #     temp_value.append(int(s[i]))
        # return_value.append(temp_value)
        return return_value

    def UNHASH(self, state):
        return_value = []
        count = 0
        for i in range(6):
            temp_value = []
            for j in range(4):
                temp_value.append(state[(i * 4) + j])
                count += 1
            return_value.append(temp_value)
        # temp_value = []
        # for i in range(count, len(state)):
        #     temp_value.append(state[i])
        # return_value.append(temp_value)
        return return_value

    # </HELPER METHODS>

    # <INITIALIZE STATE>
    def generate_goal_state(self):
        side = self.NUM_SIDE**2
        if self.NUM_COLOR == 2:
            self.GOAL_STATE = [[1] * side, [1] * side, [1] * side, [2] * side, [2] * side, [2] * side]
        elif self.NUM_COLOR == 3:
            self.GOAL_STATE = [[1] * side, [2] * side, [3] * side, [2] * side, [3] * side, [1] * side]
        elif self.NUM_COLOR == 6:
            self.GOAL_STATE = [[1] * side, [2] * side, [3] * side, [4] * side, [5] * side, [6] * side]
        else:
            print("You have have a " + str(self.NUM_COLOR) + " colored cube. Please try again.")
            quit()

    # Examples of different scrambles.
    # Scramble numbers have to be in between -6 to 6 and does not include 0, 2, 3, 6, -2, -3, -6.
    # 1. [-4, -1, 5]
    def CREATE_INITIAL_STATE(self):
        # Rubik's Cube Scramble Generation
        initial = self.GOAL_STATE
        # initial.append(list(range(8)))
        scramble = [1, 5, -1, -5]
        newstate = self.INITIALIZE_MOVE(initial, scramble[0])
        print("Initial Scramble to initialize state generation and Q-Learning")
        print(self.DESCRIBE_STATE(newstate) + ': Move: ' + str(scramble[0]) + '\n')
        # temp = self.change_state_to_set(newstate)
        # print(self.change_state_to_set(newstate))
        # print(self.change_set_to_list(temp))
        for nums in scramble[1:]:
            newstate = self.INITIALIZE_MOVE(newstate, nums)
            print(self.DESCRIBE_STATE(newstate) + ': Move: ' + str(nums) + '\n')
            # temp = self.change_state_to_set(newstate)
            # print(self.change_state_to_set(newstate))
            # print(self.change_set_to_list(temp))
        return newstate

    def INITIALIZE_MOVE(self, s, step):
        news = self.copy_state(s)  # start with a deep copy.
        if step > 0:
            news[step - 1] = self.rotate_face(news[step - 1], "clockwise")
        else:
            news[abs(step) - 1] = self.rotate_face(news[abs(step) - 1], "cc")
        news[0:6] = self.rotate_edges(news[0:6], step)
        # news[6] = rotate_cubies(news[6], step)
        return news  # return new state

    # </INITIALIZE STATE>

    # <OPERATORS>
    class Operator:
        def __init__(self, name, precond, state_transf):
            self.name = name
            self.precond = precond
            self.state_transf = state_transf

        def is_applicable(self, s):
            return self.precond(s)

        def apply(self, s):
            return self.state_transf(s)

    def can_move(self, s):
        if s == 'Solved':
            return False
        return True

    OPERATORS = []

    def INITIALIZE_OPERATORS(self):
        global OPERATORS
        for i in range(len(self.NUMBER_COMBINATIONS)):
            self.OPERATORS.append(self.Operator("Move " + str(self.NUMBER_COMBINATIONS[i]) + " face.",
                                                lambda s, i=i: self.can_move(s),
                                                lambda s, i=i: self.change_state_to_set(self.move(
                                                    self.change_set_to_list(s), self.NUMBER_COMBINATIONS[i])))
                                  )
        return self.OPERATORS

    # </OPERATORS>

    # <FILE WRITING AND READING>
    def write_known_states(self, states, name):
        NUM_ROWS = 20
        count = 0
        separated_states = []
        print(len(states))
        set_to_list = list(states)
        for i in range(math.ceil(len(states) / NUM_ROWS)):
            temp_array = []
            for j in range(NUM_ROWS):
                if j + count < len(states):
                    temp_array.append(set_to_list[j + count])
            count += NUM_ROWS
            separated_states.append(temp_array)
        print("Separating Finished")

        cw = open(name, 'w')
        for i in separated_states:
            for j in i:
                cw.write("%s," % (str(j)))
            cw.write('\n')
        cw.close()

    def read_known_states(self, file_name):
        return_list = []
        with open(file_name, 'r') as f:
            # reader = csv.reader(f)
            data = list(list(rec) for rec in csv.reader(f, delimiter=','))
            for row in data:
                for values in row[0:(len(row) - 1)]:
                    return_list.append(str(values))
        return return_list

    # </FILE WRITING AND READING>

    # <PRINTING AND VISUALIZATION>
    def DESCRIBE_STATE(self, state):
        txt = self.describe_layer(state, 0)

        c = 0
        for i in range(self.NUM_SIDE):
            for value in state[1:5]:
                for j in range(self.NUM_SIDE):
                    txt += str(value[c + j]) + " "
                txt += " "
            c += 2
            txt += "\n"
        txt += "\n"

        txt += self.describe_layer(state, 5)
        return txt

    # Helper for DESCRIBE_STATE
    def describe_layer(self, state, pos):
        txt = ""
        a = 0
        for i in range(self.NUM_SIDE):
            for j in range(5):
                txt += " "
            for value in state[pos][a:(a + self.NUM_SIDE)]:
                txt += str(value) + " "
            a += self.NUM_SIDE
            txt += "\n"
        txt += "\n"
        return txt

    def print_solution(self, state, policies):
        temp_state = state
        print(temp_state)
        count = 0
        count_repeat = 0
        old_policy = policies.get(temp_state)
        while temp_state != self.HASHCODE(self.GOAL_STATE) and count < self.MAX_MOVES:
            set_list = self.UNHASH(temp_state)
            print(self.DESCRIBE_STATE(set_list) + 'Move: ' + str(old_policy) + '\n')
            new_state = self.move(set_list, policies.get(temp_state))

            if policies.get(self.HASHCODE(new_state)) == old_policy:
                count_repeat += 1
                old_policy = policies.get(temp_state)
                temp_state = self.HASHCODE(new_state)

            # Under the condition that you repeat the same move more than 3 times, choose a random action.
            if policies.get(self.HASHCODE(new_state)) == -old_policy or count_repeat > 4:
                count_repeat = 0
                temp_actions = self.NUMBER_COMBINATIONS.copy()
                temp_actions.remove(old_policy)
                rand_action = random.choice(temp_actions)
                temp_state = self.HASHCODE(self.move(set_list, rand_action))
                old_policy = rand_action
            count += 1

        if temp_state == self.HASHCODE(self.GOAL_STATE):
            print("\nPuzzle solved!")
        else:
            print("\nPuzzle not solved. Maybe try more episodes!")

    # </PRINTING AND VISUALIZATION>

    # <RUNNING>
    GOAL_STATE_SET = ''

    def run(self, load_from_file, save_to_file, LOAD_FILE_NAME, SAVE_FILE_NAME):
        global GOAL_STATE_SET
        grid_MDP = MDP.MDP()
        self.generate_goal_state()
        GOAL_STATE_SET = self.change_state_to_set(self.GOAL_STATE)
        self.INITIAL_STATE = self.change_state_to_set(self.CREATE_INITIAL_STATE())
        grid_MDP.register_start_state(self.INITIAL_STATE)
        grid_MDP.register_actions(self.NUMBER_COMBINATIONS)
        grid_MDP.register_operators(self.INITIALIZE_OPERATORS())
        grid_MDP.register_transition_function(self.T)
        grid_MDP.register_reward_function(self.R)
        grid_MDP.register_action_policy()

        if load_from_file:
            # To read the known states in
            known_states = self.read_known_states(LOAD_FILE_NAME)
            grid_MDP.register_known_states(set(known_states))
        else:
            # To generate the known states
            known_states = grid_MDP.generateAllStates()
            if save_to_file:
                # Write the known states to a .csv file
                self.write_known_states(known_states, SAVE_FILE_NAME)

        print('Length: ' + str(len(known_states)))

        q_values, policy = grid_MDP.QLearning(0.9, self.NUM_EPISODES, self.EPSILON, self.ACTION_POLICY, self.SHOW_STATE)

        self.print_solution(self.INITIAL_STATE, policy)
        return

    # </RUNNING>
