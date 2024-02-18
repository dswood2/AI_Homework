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
import random, util

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

        "*** YOUR CODE HERE ***"
        # Calculate distances to the closest food and ghosts
        food_Distances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        ghost_Distances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        
        # Check if there are any ghosts nearby
        if min(ghost_Distances) < 2:
            return float('-inf')  # Avoid ghosts if too close

        # Calculate the closest food distance
        if food_Distances:
            reciprocal_Food_Distance = 1.0 / min(food_Distances)
        else:
            reciprocal_Food_Distance = 0

        # Consider the remaining scared time of ghosts
        scared_Ghosts = []
        for ghost_State, scared_Time in zip(newGhostStates, newScaredTimes):
            if scared_Time > 0:
                scared_Ghosts.append(ghost_State)
                
        # Adjust the score based on the reciprocal of the closest food distance and scared ghosts
        score = successorGameState.getScore() + reciprocal_Food_Distance - len(scared_Ghosts)

        return score
        return successorGameState.getScore()

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
        def minimaxValue(state, agent_Index, depth):
            # Base case: terminal state or reached maximum depth
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            
            # Maximize for Pacman (agentIndex = 0)
            if agent_Index == 0: 
                current_minimax_val = float('-inf')
                for action in state.getLegalActions(agent_Index):
                    successor = state.generateSuccessor(agent_Index, action)
                    # Recursive call to get the value for the successor state
                    current_minimax_val = max(current_minimax_val, minimaxValue(successor, 1, depth))
                return current_minimax_val
            
            # Minimize for ghosts (agentIndex > 0)
            else:
                current_minimax_val = float('inf') 
                for action in state.getLegalActions(agent_Index):
                    successor = state.generateSuccessor(agent_Index, action)
                    # If the last ghost is reached, move to the next depth level
                    if agent_Index == state.getNumAgents() - 1:
                        current_minimax_val = min(current_minimax_val, minimaxValue(successor, 0, depth-1))  
                    else:
                        # Move to the next ghost
                        current_minimax_val = min(current_minimax_val, minimaxValue(successor, agent_Index+1, depth))
                return current_minimax_val
        legal_Actions = gameState.getLegalActions(0)  # Get legal actions for Pacman

        # Initialize variables to keep track of the best action and its minimax value
        best_Action = None
        best_Minimax_Value = float('-inf')

        # Iterate through legal actions to find the one with the maximum minimax value
        for action in legal_Actions:
            successor_State = gameState.generateSuccessor(0, action)
            current_Minimax_Value = minimaxValue(successor_State, 1, self.depth)

            # Update best action if the current action leads to a state with a higher minimax value
            if current_Minimax_Value > best_Minimax_Value:
                best_Minimax_Value = current_Minimax_Value
                best_Action = action
        return best_Action

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.maxValue(gameState, 0, 0, -float("inf"), float("inf"))[0]

    
    def alphaBetaSearch(self, gameState, agent_Index, depth, alpha, beta):
        # Perform alpha-beta pruning search to find the best move.
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        if agent_Index == 0:
            return self.maxValue(gameState, agent_Index, depth, alpha, beta)[1]
        else:
            return self.minValue(gameState, agent_Index, depth, alpha, beta)[1]

    def maxValue(self, gameState, agent_Index, depth, alpha, beta):
        # Maximize the value for the maximizing player
        best_Action = None
        value = -float("inf")

        for action in gameState.getLegalActions(agent_Index):
            # Generate successor state after taking the action
            successor = self.alphaBetaSearch(
                gameState.generateSuccessor(agent_Index, action),
                (depth + 1) % gameState.getNumAgents(),
                depth + 1,
                alpha,
                beta
            )

            # Update bestAction and value based on the successor's value
            if successor > value:
                value = successor
                best_Action = action

            # Perform alpha-beta pruning if value exceeds beta
            if value > beta:
                return best_Action, value
            
            # Update alpha to be the maximum of alpha and value
            alpha = max(alpha, value)

        return best_Action, value

    
    def minValue(self, gameState, agent_Index, depth, alpha, beta):
        # Minimize the value for the minimizing player
        best_Action = None
        value = float("inf")

        for action in gameState.getLegalActions(agent_Index):
            # Generate successor state after taking the action
            successor = self.alphaBetaSearch(
                gameState.generateSuccessor(agent_Index, action),
                (depth + 1) % gameState.getNumAgents(),
                depth + 1,
                alpha,
                beta
            )

            # Update bestAction and value based on the successor's value
            if successor < value:
                value = successor
                best_Action = action
            
            # Perform alpha-beta pruning if value is less than alpha
            if value < alpha:
                return best_Action, value

            # Update beta to be the minimum of beta and value
            beta = min(beta, value)

        return best_Action, value

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.

          The expectimax function returns a tuple of (actions,
        """
        "*** YOUR CODE HERE ***"
        # calling expectimax with the depth we are going to investigate
        max_Depth = self.depth * gameState.getNumAgents()
        return self.expectimax(gameState, max_Depth, 0)[0]

    def expectimax(self, gameState, depth, agent_Index):
        # Recursively computes the expectimax value for a given state.
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return (None, self.evaluationFunction(gameState))
        
        legal_Actions = gameState.getLegalActions(agent_Index)
        
        if agent_Index == 0: # Pacman
            return self.maxValue(gameState, depth, agent_Index, legal_Actions)
        else: # Ghosts
            return self.expectedValue(gameState, depth, agent_Index, legal_Actions)

    def maxValue(self, gameState, depth, agent_Index, legal_Actions):
        # Computes the maximum value for Pacman.
        best_Action = None
        best_Value = -float("inf")
        
        for action in legal_Actions:
            # Generate successor state after Pacman's action
            successor = gameState.generateSuccessor(agent_Index, action)
            # Get the expectimax value for the successor state
            value = self.expectimax(successor, depth-1, (agent_Index+1) % gameState.getNumAgents())[1]
            
            if value > best_Value:
                best_Value = value
                best_Action = action
                
        return best_Action, best_Value

    def expectedValue(self, gameState, depth, agent_Index, legal_Actions):
        # Computes the expected value for ghosts.
        avg_Score = 0
        
        for action in legal_Actions:
            # Generate successor state after ghost's action
            successor = gameState.generateSuccessor(agent_Index, action)
            # Get the expectimax value for the successor state     
            value = self.expectimax(successor, depth-1, (agent_Index+1) % gameState.getNumAgents())[1]
            avg_Score += value
        
        avg_Score /= len(legal_Actions)
        return None, avg_Score

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This definition considers a more comprehensive set of 
    factors influencing Pacman's performance. It takes into account 
    ghosts, remaining food, the distance to the center, and the game state, 
    providing a more strategic evaluation. Adjusting weights allows
    fine tuning the agent's behavior.
    """
    "*** YOUR CODE HERE ***"
    pacman_Pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    ghost_States = currentGameState.getGhostStates()

    # Evaluate the distance to the closest ghost
    closest_Ghost_Distance = min(manhattanDistance(pacman_Pos, ghost.getPosition()) for ghost in ghost_States)

    # Evaluate the distance to the closest scared ghost
    scared_Ghosts = [ghost for ghost in ghost_States if ghost.scaredTimer > 0]
    closest_Scared_Ghost_Distance = min(manhattanDistance(pacman_Pos, ghost.getPosition()) for ghost in scared_Ghosts) if scared_Ghosts else float('inf')

    # Evaluate the number of remaining food
    remaining_Food = food.asList()
    food_Count = len(remaining_Food)

    # Evaluate the number of remaining power capsules
    capsule_Count = len(capsules)

    # Evaluate Pacman's distance to the center
    center_Pos = (currentGameState.data.layout.width // 2, currentGameState.data.layout.height // 2)
    distance_To_Center = manhattanDistance(pacman_Pos, center_Pos)

    # Evaluate the game state (win, lose, or ongoing)
    if currentGameState.isWin():
        game_Score = float('inf')
    elif currentGameState.isLose():
        game_Score = float('-inf')
    else:
        game_Score = currentGameState.getScore()

    # Combine the components into an overall evaluation score
    evaluation = 0
    # Weights for different components
    weight_ghost = -50
    weight_scared_ghost = -20
    weight_food = -5
    weight_capsule = -10
    weight_center = -2
    weight_game_state = 100

    # Add contributions from different components based on their weights
    evaluation += weight_ghost * closest_Ghost_Distance
    evaluation += weight_scared_ghost / (closest_Scared_Ghost_Distance + 1)  # Add 1 to avoid division by zero
    evaluation += weight_food * food_Count
    evaluation += weight_capsule * capsule_Count
    evaluation += weight_center / (distance_To_Center + 1)  # Add 1 to avoid division by zero
    evaluation += weight_game_state * game_Score

    return evaluation

# Abbreviation
better = betterEvaluationFunction
