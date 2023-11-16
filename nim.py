import math
import random
import time


class Nim():

    def __init__(self, initial=[1,3,5,7]):
        """
        Initialize game board.
        Each game board has
            - `piles`: a list of how many elements remain in each pile
            - `player`: 0 or 1 to indicate which player's turn
            - `winner`: None, 0, or 1 to indicate who the winner is
        """
        self.piles = initial.copy()
        # print(f"self.piles={self.piles}")
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """
        Nim.available_actions(piles) takes a `piles` list as input
        and returns all of the available actions `(i, j)` in that state.

        Action `(i, j)` represents the action of removing `j` items
        from pile `i` (where piles are 0-indexed).
        """
        # print(f"+++available_actions")
        actions = set()
        for i, pile in enumerate(piles):
            # print(i, pile  )
            for j in range(1, pile + 1):
                # print(f":{j}")
                actions.add((i, j))
        # print(f"+++>>>available actions={actions}")
        return actions

    @classmethod
    def other_player(cls, player):
        """
        Nim.other_player(player) returns the player that is not
        `player`. Assumes `player` is either 0 or 1.
        """
        return 0 if player == 1 else 1

    def switch_player(self):
        """
        Switch the current player to the other player.
        """
        self.player = Nim.other_player(self.player)

    def move(self, action):
        """
        Make the move `action` for the current player.
        `action` must be a tuple `(i, j)`.
        """
        # print(f"\n+++move: state b4 action={self.piles}")
        pile, count = action
        # print(f"---action={action} for player {self.player}")

        # Check for errors
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Update pile
        self.piles[pile] -= count
        self.switch_player()
        # print(f"---state after action={self.piles}")

        # print(f"# Check for a winner")
        # print(f"---self.piles={self.piles}")
        if all(pile == 0 for pile in self.piles):
            # print(f"---all piles empty")
            self.winner = self.player
            # print(f"---winner={self.winner}")


class NimAI():

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)`
        pairs to a Q-value (a number).
         - `state` is a tuple of remaining piles, e.g. (1, 1, 4, 4)
         - `action` is a tuple `(i, j)` for an action
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Update Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received
        from taking that action.
        """
        # print(f"+++update")
        # print(f"---old_state={old_state}, action={action}, new_state={new_state}, reward={reward}")
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)
        # print(f"+++>>>update: self.q={self.q}")

    def get_q_value(self, state, action):
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in `self.q`, return 0.
        """
        # self.q = ((0, 0, 0, 2), (3, 2): -1,
        # print
        if not state or not action:
            return 0   
        # print(f"---state type= {type(state)}")
        statetuple = tuple(state)
        # print(f"---state type= {type(statetuple)}")

        
        q =  self.q.get((statetuple, action), 0)
        # print(f"+++>>>get_q_value: {state}, {action} = q {q}")
        return q 

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estiamte of future rewards `future_rewards`.

        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """
        if state is None or action is None:
            # print(f"---!!!!!state or action is None")
            return
   
        statetuple=tuple(state)
     
        newvalest = reward + future_rewards
 
        result = old_q + (self.alpha * (newvalest - old_q))
        result = round(result, 2)
        self.q[statetuple, action] = result
        # print(f"---updateq self.q = {self.q[statetuple, action]}")
        

    def best_future_reward(self, state):
        """
        Given a state `state`, consider all possible `(state, action)`
        pairs available in that state and return the maximum of all
        of their Q-values.

        Use 0 as the Q-value if a `(state, action)` pair has no
        Q-value in `self.q`. If there are no available actions in
        `state`, return 0.

        =epsilon is the exploration rate (probability of taking a random action))]
        ='Q-table' is a dictionary mapping state-action pairs to Q-values
        """
        actions =  Nim.available_actions(state)
        # print(f"+++best_future: actions={actions} len={len(actions)})") 
        if not actions:
            # print(f"---no actions")
            return 0
        # get q value for actions
        qlist = []
        for action in actions:
            # print(f"---action={action}")
            # print(f"---q = {self.get_q_value(state, action)})")
            qlist.append(self.get_q_value(state, action))
        # print(f"---qlist={qlist} len={len(qlist)}")
        best = max(qlist)
        # print(f"---best={best}")
        # gamma = (1 - self.epsilon)
        gamma = 1
        result =  gamma * best
        # print(f"---result={result}")
        # print(f">>>best={best}")
        return result
        
    

        # get max of the q values

    def choose_action(self, state, epsilon=True):
        """
        Given a state `state`, return an action `(i, j)` to take.

        If `epsilon` is `False`, then return the best action
        available in the state (the one with the highest Q-value,
        using 0 for pairs that have no Q-values).

        If `epsilon` is `True`, then with probability
        `self.epsilon` choose a random available action,
        otherwise choose the best action available.

        If multiple actions have the same Q-value, any of those
        options is an acceptable return value.
        """
        actions = Nim.available_actions(state)
        # print(f"---actions = {actions}")
        if len(actions) == 1:
            return list(actions)[0]
        # print(f"+++choose_action: {actions}")
        if epsilon == True:
        # With Epsilon True: choose random action with epsilon prob
            # print(f"---epsilon = {epsilon}")
            x = random.random()
            # print(f"---random x = {x}")
            if x < self.epsilon:
                action = random.choice(list(actions))
                # print(f">>>EPs=True : random action= {action}")
                return action

        # else choose best action
        max = 0
        bestaction = None
        for action in actions:
            if not bestaction:
                bestaction = action
            # print(f"---action in loop={action}")
            q = self.get_q_value(state, action)
            # print(f"---q={q}")
            if q >= max:
                max = q
                bestaction = action
        # print(f">>>chooseaction={bestaction}")
        return bestaction

    # def printq(self,x):
    #     for key,val in self.q.items():
    #         # print(f"///{key[0][1]}")
    #         if key[0] == x:  # enter state you want to inspect

    #             print(f"result = {key,':',val}")
    #             print(f"result = {key} : {val}")

def train(n):
    """
    Train an AI by playing `n` games against itself.
    """

    player = NimAI()
    # player0wins = 0

    # Play n games
    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()
        # print(f"---q dict={player.q}")

        # Keep track of last move made by either player
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        # Game loop
        while True:
            # print(f"\n---player = {game.player}")

            # Keep track of current state and action
            state = game.piles.copy()
            # print(f"---state={state}")
            action = player.choose_action(game.piles)
            # print(f"---action={action}")

            # Keep track of last state and action
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            # print(f"---{game.player}'s last = {last[game.player]}")

            # Make move
            game.move(action)
            new_state = game.piles.copy()
            # print(f"---new_state={new_state}")

            # When game is over, update Q values with rewards
            if game.winner is not None:
                # print(f"\n***winner = {game.winner} so update q")
                # if game.winner == 0:
                #     player0wins += 1 
                player.update(state, action, new_state, -1)
                # print(f"---state action = {state, action}")
                # print(f"---{state}{action} = {player.q[(tuple(state), action)]}")
                

                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1
                )
                # print(f"---q table size = {len(player.q)}")
                # print(f"---q table: {player.q}")
                # player.printq((1,3,5,7))
                break

            # If game is continuing, no rewards yet
            elif last[game.player]["state"] is not None:
                # print(f"---continue game")
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0
                )
            # print(f">>>endloop Tain")
        # firstplayerwins = player0wins/n
        # secondplayerwins = 1 - firstplayerwins
        # print(f"\n---{n} games played")
        # print(f"first player wins {firstplayerwins}")
        # print(f"second player wins {secondplayerwins}\n")

    print("Done training")

    # Return the trained AI
    return player


def play(ai, human_player=1):
    """
    Play human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether
    human player moves first or second.
    """

    # If no player order set, choose human's order randomly
    if human_player is None:
        human_player = random.randint(0, 1)

    # Create new game
    game = Nim()

    # def xor(numbers):
    #     xor_result = 0
    #     for number in numbers:
    #         print(number)
    #         xor_result ^= number
    #         print(f"x={xor_result}")

    #     return xor_result

    # xored = xor(game.piles)
    # # print(f"xored={xored}")



    # Game loop
    while True:

        # Print contents of piles
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        # Compute available actions
        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        # Let human make a move
        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))
                if (pile, count) in available_actions:
                    break
                print("Invalid move, try again.")

        # Have AI make a move
        else:
            print("AI's Turn")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        # Make move
        game.move((pile, count))

        # Check for winner
        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return
