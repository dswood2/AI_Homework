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
        def maxValue(state, depth):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            v = float('-inf')
            legalActions = state.getLegalActions(0)  # 0 represents Pacman agent index
            for action in legalActions:
                successorState = state.generateSuccessor(0, action)
                v = max(v, minValue(successorState, 1, depth))

            return v

        def minValue(state, ghostIndex, depth):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            v = float('inf')
            legalActions = state.getLegalActions(ghostIndex)
            for action in legalActions:
                successorState = state.generateSuccessor(ghostIndex, action)
                if ghostIndex == state.getNumAgents() - 1:
                    v = min(v, maxValue(successorState, depth - 1))
                else:
                    v = min(v, minValue(successorState, ghostIndex + 1, depth))

            return v

        legalActions = gameState.getLegalActions(0)
        bestAction = max(legalActions, key=lambda action: minValue(gameState.generateSuccessor(0, action), 1, self.depth))
        return bestAction
        util.raiseNotDefined()
