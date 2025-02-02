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
import math


import random, util

from game import Agent, Actions
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
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newCapsules = successorGameState.getCapsules()

        "*** YOUR CODE HERE ***"
        minDistFood = 1
        if newFood.asList():
            minDistFood = min([util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()])

        distanceOfGhosts = min(util.manhattanDistance(newPos, ghostPos) for ghostPos in successorGameState.getGhostPositions())

        ghostReallyClose = 1 if min(util.manhattanDistance(newPos, ghostPos) for ghostPos in successorGameState.getGhostPositions()) <= 2 else 0

        distanceOfCapsules = 1
        if newCapsules:
            distanceOfCapsules = min(util.manhattanDistance(newPos, capsulePos) for capsulePos in newCapsules)

        if newScaredTimes[0] != 0:
            ghostReallyClose = 0
            distanceOfGhosts = -distanceOfGhosts

        return successorGameState.getScore() + 1 / minDistFood - 1 / distanceOfGhosts - ghostReallyClose * 1000 + 1 / distanceOfCapsules * 0.1


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        gameState.getLegalActions(agentIndex):
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
        act = gameState.getLegalActions(0)[0]
        mx = float('-inf')
        for i in gameState.getLegalActions(0):
            eval = self.recur(gameState.generateSuccessor(0, i), 0, 1)
            if mx < eval:
                mx = eval
                act = i

        return act

    def recur(self, gameState, depth, agent):
        if agent == gameState.getNumAgents():
            depth += 1
            agent = 0
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:
            return max([(self.recur(gameState.generateSuccessor(agent, nextState), depth, 1)) for nextState in gameState.getLegalActions(agent)])
        else:
            return min([(self.recur(gameState.generateSuccessor(agent, nextState), depth, agent + 1)) for nextState in gameState.getLegalActions(agent)])





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        act = gameState.getLegalActions(0)[0]
        mx = float('-inf')
        alpha = float("-inf")
        beta = float("inf")
        for i in gameState.getLegalActions(0):
            eval = self.minimax(gameState.generateSuccessor(0, i), 0, 1, alpha, beta)
            if mx < eval:
                mx = eval
                act = i
            alpha = max(alpha, mx)
            beta = max(beta, mx)
        return act

    def minimax(self, gameState, depth, agent, alfa, beta):
        if agent == gameState.getNumAgents():
            agent = 0
            depth += 1

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agent == 0:
            maxEva = float("-inf")
            for action in gameState.getLegalActions(agent):
                eva = self.minimax(gameState.generateSuccessor(agent, action), depth, agent + 1, alfa, beta)
                maxEva = max(maxEva, eva)
                alfa = max(alfa, maxEva)
                if beta < alfa:
                    break
            return maxEva

        else:
            minEva = float("inf")
            for action in gameState.getLegalActions(agent):
                eva = self.minimax(gameState.generateSuccessor(agent, action), depth, agent + 1, alfa, beta)
                minEva = min(minEva, eva)
                beta = min(beta, minEva)
                if beta < alfa:
                    break
            return minEva


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
        act = gameState.getLegalActions(0)[0]
        mx = float('-inf')
        for i in gameState.getLegalActions(0):
            eval = self.eptiMax(gameState.generateSuccessor(0, i), 0, 1)
            if mx < eval:
                mx = eval
                act = i
        return act

    def eptiMax(self, gameState, depth, agent):
        if agent == gameState.getNumAgents():
            agent = 0
            depth += 1

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agent == 0:
            maxEva = float("-inf")
            for action in gameState.getLegalActions(agent):
                eva = self.eptiMax(gameState.generateSuccessor(agent, action), depth, agent + 1)
                maxEva = max(maxEva, eva)
            return maxEva

        else:
            summ = 0
            for action in gameState.getLegalActions(agent):
                eva = self.eptiMax(gameState.generateSuccessor(agent, action), depth, agent + 1)
                summ += eva
            return summ / len(gameState.getLegalActions(agent))

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsules = currentGameState.getCapsules()

    minDistFood = 1
    if newFood.asList():
        minDistFood = min([util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()])

    distanceOfGhosts = min(
        util.manhattanDistance(newPos, ghostPos) for ghostPos in currentGameState.getGhostPositions())

    ghostReallyClose = 1 if min(
        util.manhattanDistance(newPos, ghostPos) for ghostPos in currentGameState.getGhostPositions()) <= 2 else 0

    distanceOfCapsules = 1
    if newCapsules:
        distanceOfCapsules = min(util.manhattanDistance(newPos, capsulePos) for capsulePos in newCapsules)

    if newScaredTimes[0] != 0:
        ghostReallyClose = -1/20
        distanceOfGhosts = -distanceOfGhosts

    if distanceOfGhosts == 0:
        distanceOfGhosts = 1000
    return currentGameState.getScore() + 1 / minDistFood - 1 / distanceOfGhosts - ghostReallyClose * 1000 + 1 / distanceOfCapsules * 0.1

# Abbreviation
better = betterEvaluationFunction
