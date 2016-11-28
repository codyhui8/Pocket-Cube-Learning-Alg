'''
Cody Hui 1338250
CSE 415, Spring 2016, University of Washington
RunProgram.py
Instructor:  S. Tanimoto.

Implementation of the code transparency functions.
'''


import PocketCube


def run():
    NUM_SIDE = input("Please enter the size of the side. For example, 3 for a 3x3 and 2 for a 2x2. \n>>> ")

    NUM_COLORS = input("\nPlease enter the number of colors that you want for the cube. There can be 2, 3, or 6 colors.\n"
                       ">>> ")

    NUM_EPISODES = input("\nPlease enter the number of episodes that you would like to run the program for.\n"
                         "For a 2x2, it is recommended that you do 10,000 and for a 3x3, it is recommended that you do 1,000,000.\n"
                         ">>> ")

    ACTIONS = input("\nPlease enter the actions to test. The action should be a number between -5 to 5, "
                    "not include +-2, +-3, and 0.\n"
                    "These values should be values separated by a space, for example: 1 4 5 -1 -4 -5\n"
                    ">>> ")
    ACTIONS = list(ACTIONS.split(" "))
    ACTIONS = map(int, ACTIONS)

    ACTION_POLICY = input("\nPlease enter the action policy method. Enter one of the following:"
                          " epsilon-greedy or epsilon-soft.\n"
                          ">>> ")

    EPSILON = input("\nPlease enter the epsilon value. This number can be any number between 0 and 1. The default is 0.2.\n"
                    ">>> ")

    MAX_MOVES = input("\nPlease enter the maximum number of moves that you would like to have for the solution.\n"
                      "The default number is 500.\n"
                      ">>> ")

    SHOW_STATE = input("\nHow often do you want to show the state and action pairs and their corresponding Q-Values?\n"
                       "These can be any whole numbers. The default number is 50.\n"
                       ">>> ")

    LOAD_NOLOAD = input("\nPlease specify if you would like to load a known_states file or not.\n"
                        "Enter a Y as a Yes and a N as a No.\n"
                        ">>> ")

    LOAD_FILE_NAME = ''
    SAVE_FILE_NAME = ''
    save_to_file = False
    load_from_file = False
    if LOAD_NOLOAD == 'Y':
        load_from_file = True
        LOAD_FILE_NAME = input("\nPlease input the file name. For example, 'known_states.csv' without the quotation marks.\n"
                          ">>> ")
    else:
        save_value = input("\nPlease specify if you would like to save the known states to a file.\n"
                           "Enter a Y as a Yes and a N as a No.\n"
                           ">>> ")
        if save_value == 'Y':
            save_to_file = True
            SAVE_FILE_NAME = input("\nPlease input the file name. For example, 'known_states.csv' without the quotation marks.\n"
                          ">>> ")

    cube = PocketCube.PocketCube()
    cube.init_numside(int(NUM_SIDE))
    cube.init_numcolor(int(NUM_COLORS))
    cube.init_numepisodes(int(NUM_EPISODES))
    cube.init_actions(list(ACTIONS))
    cube.init_epsilon(float(EPSILON))
    cube.init_action_policy(ACTION_POLICY)
    cube.init_max_moves(int(MAX_MOVES))
    cube.init_show_state(int(SHOW_STATE))

    cube.run(load_from_file, save_to_file, LOAD_FILE_NAME, SAVE_FILE_NAME)

run()
