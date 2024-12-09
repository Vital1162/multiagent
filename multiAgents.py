# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        legalMoves = gameState.getLegalActions()

        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # rand best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        "*** YOUR CODE HERE ***"

        # Maximize distance from ghosts and minimize distance to food
        return self.dist_nearest_ghost(successorGameState) / (
            self.dist_nearest_food(successorGameState, currentGameState) + 0.1
        )

    def dist_nearest_food(self, nextState, currentState):
        pacmanPos = nextState.getPacmanPosition()
        foodsPos = currentState.getFood().asList()

        return min(
            [
                ((pacmanPos[0] - foodPos[0]) ** 2 + (pacmanPos[1] - foodPos[1]) ** 2)
                ** 0.5
                for foodPos in foodsPos
            ]
        )

    def dist_nearest_ghost(self, gameState):
        pacmanPos = gameState.getPacmanPosition()
        ghostsPos = gameState.getGhostPositions()

        return min(
            [
                ((pacmanPos[0] - ghostPos[0]) ** 2 + (pacmanPos[1] - ghostPos[1]) ** 2)
                ** 0.5
                for ghostPos in ghostsPos
            ]
        )


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        action, _ = self.min_max_value(gameState, 0, 0)
        return action

    def min_max_value(self, game_state, agent_index, depth):
        # check stop
        if (
            game_state.isWin()
            or game_state.isLose()
            # limit
            or depth >= self.depth * game_state.getNumAgents()
        ):
            return "Stop", self.evaluationFunction(game_state)
        elif agent_index == 0:  # pacman turn
            return self.compute_best_option(game_state, 0, depth, best_function=max)
        else:  # ghost turn
            return self.compute_best_option(
                game_state, agent_index, depth, best_function=min
            )

    def compute_best_option(self, game_state, agent_index, depth, best_function):
        """Get all the legal action in present
        then get all next states
        using min max to find next state
        """
        next_actions = game_state.getLegalActions(agent_index)
        next_states = [
            game_state.generateSuccessor(agent_index, action) for action in next_actions
        ]

        next_agent_index = (agent_index + 1) % game_state.getNumAgents()
        values = [
            self.min_max_value(next_state, next_agent_index, depth + 1)[1]
            for next_state in next_states
        ]
        best_value = best_function(values)
        best_action = next_actions[values.index(best_value)]
        return best_action, best_value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        action, _ = self.min_max_value(
            game_state=gameState,
            agent_index=0,
            alpha=float("-inf"),
            beta=float("inf"),
            depth=0,
        )
        return action

    def min_max_value(self, game_state, agent_index, alpha, beta, depth):
        if (
            game_state.isWin()
            or game_state.isLose()
            or depth >= self.depth * game_state.getNumAgents()
        ):
            return "Stop", self.evaluationFunction(game_state)
        elif agent_index == 0:
            return self.max_value(game_state, 0, alpha, beta, depth)
        else:
            return self.min_value(game_state, agent_index, alpha, beta, depth)

    def max_value(self, game_state, agent_index, alpha, beta, depth):
        # init
        best_value = float("-inf")
        best_action = "Stop"

        next_agent_index = (agent_index + 1) % game_state.getNumAgents()

        for next_action in game_state.getLegalActions(agent_index):
            next_state = game_state.generateSuccessor(agent_index, next_action)
            next_value = self.min_max_value(
                next_state, next_agent_index, alpha, beta, depth + 1
            )[1]
            if next_value > best_value:
                best_action, best_value = next_action, next_value
            if best_value > beta:
                return best_action, best_value
            alpha = max(alpha, best_value)
        return best_action, best_value

    def min_value(self, game_state, agent_index, alpha, beta, depth):
        # init
        best_value = float("inf")
        best_action = "Stop"

        next_agent_index = (agent_index + 1) % game_state.getNumAgents()

        for next_action in game_state.getLegalActions(agent_index):
            next_state = game_state.generateSuccessor(agent_index, next_action)
            next_value = self.min_max_value(
                next_state, next_agent_index, alpha, beta, depth + 1
            )[1]
            if next_value < best_value:
                best_action, best_value = next_action, next_value
            if best_value < alpha:
                return best_action, best_value
            beta = min(beta, best_value)
        return best_action, best_value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        action, _ = self.expect_max_value(gameState, 0, 0)
        return action

    def expect_max_value(self, game_state, agent_index, depth):
        if (
            game_state.isWin()
            or game_state.isLose()
            or depth >= self.depth * game_state.getNumAgents()
        ):
            return "Stop", self.evaluationFunction(game_state)
        elif agent_index == 0:
            return self.max_value(game_state, 0, depth)
        else:
            return self.expect_value(game_state, agent_index, depth)

    def max_value(self, game_state, agent_index, depth):
        next_actions = game_state.getLegalActions(agent_index)
        next_states = [
            game_state.generateSuccessor(agent_index, action) for action in next_actions
        ]
        next_agent_index = (agent_index + 1) % game_state.getNumAgents()
        values = [
            self.expect_max_value(next_state, next_agent_index, depth + 1)[1]
            for next_state in next_states
        ]
        best_value = max(values)
        best_action = next_actions[values.index(best_value)]
        return best_action, best_value

    def expect_value(self, game_state, agent_index, depth):
        next_actions = game_state.getLegalActions(agent_index)
        next_states = [
            game_state.generateSuccessor(agent_index, action) for action in next_actions
        ]
        next_agent_index = (agent_index + 1) % game_state.getNumAgents()
        values = [
            self.expect_max_value(next_state, next_agent_index, depth + 1)[1]
            for next_state in next_states
        ]
        mean_value = sum(values) / len(values)
        return None, mean_value


def betterEvaluationFunction(game_state: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    Init the weight for wining, losing penalty, food weight, food distant weight,
    and

    Point of evaluation will higher if pacman near the food
    and stay out of ghost more

    capsules will be an option but not prior for pacman

    losing score will remove nearly all point of evaluation

    wining also get some point

    Evalutae by {
        food_count_score
        ghost_distance_score
        food_distance_score
        capsules_score
        outcome_score
    }
    """
    "*** YOUR CODE HERE ***"

    WINNING_SCORE = 1e5
    LOSING_PENALTY = -1e5
    FOOD_COUNT_WEIGHT = 1e6
    FOOD_DISTANCE_WEIGHT = 1e4  # don't make it to high this
    CAPSULES_WEIGHT = 1e5  # don't get below to 1e

    nearest_food_distance = distance_to_nearest(
        game_state, game_state.getFood().asList()
    )
    nearest_ghost_distance = distance_to_nearest(
        game_state, game_state.getGhostPositions()
    )
    remaining_food_count = game_state.getNumFood()
    remaining_capsules_count = len(game_state.getCapsules())

    food_count_score = 1 / (remaining_food_count + 1) * FOOD_COUNT_WEIGHT
    ghost_distance_score = nearest_ghost_distance
    food_distance_score = 1 / nearest_food_distance * FOOD_DISTANCE_WEIGHT
    capsules_score = 1 / (remaining_capsules_count + 1) * CAPSULES_WEIGHT

    outcome_score = 0
    if game_state.isLose():
        outcome_score += LOSING_PENALTY
    elif game_state.isWin():
        outcome_score += WINNING_SCORE

    return (
        food_count_score
        + ghost_distance_score
        + food_distance_score
        + capsules_score
        + outcome_score
    )


def distance_to_nearest(game_state, positions):
    if len(positions) == 0:
        return math.inf
    pacman_pos = game_state.getPacmanPosition()
    # using manhattanDistance in utils.py
    distances = [manhattanDistance(pacman_pos, position) for position in positions]
    return min(distances)


better = betterEvaluationFunction
