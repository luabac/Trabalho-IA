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
from multiAgents import MultiAgentSearchAgent


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Agente Minimax: o Pac-Man (agente 0) maximiza e os fantasmas
    (agentes 1..n-1) minimizam a utilidade do estado.
    """

    def getAction(self, gameState: GameState):
        """
        Retorna a melhor ação do Pac-Man usando Minimax com profundidade
        self.depth (uma "profundidade" = 1 jogada do Pac-Man + 1 jogada de
        cada fantasma).
        """

        def minimax(agentIndex=0, depth=0, state=gameState):
            # Jogo acabou ou profundidade máxima atingida (caso base)
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Próximo agente- depois do último fantasma volta para o Pac-Man
            lastAgent = state.getNumAgents() - 1
            nextAgent = 0 if agentIndex == lastAgent else agentIndex + 1

            # Próxima profundidade- só aumenta quando o último fantasma joga
            nextDepth = depth + 1 if agentIndex == lastAgent else depth

            actions = state.getLegalActions(agentIndex)
            # Agente sem ações (encurralado): trata o estado como terminal
            if not actions:
                return self.evaluationFunction(state)

            # Nó MAX (Pac-Man)
            if agentIndex == self.index:
                bestScore = -float('inf')
                bestAction = None
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score = minimax(nextAgent, nextDepth, successor)
                    if score > bestScore:
                        bestScore = score
                        bestAction = action
                # Na raiz da árvore devolve a AÇÃO; nas chamadas recursivas, o VALOR
                if depth == 0 and state is gameState:
                    return bestAction
                return bestScore

            # Nó MIN (Fantasmas)
            bestScore = float('inf')
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                score = minimax(nextAgent, nextDepth, successor)
                if score < bestScore:
                    bestScore = score
            return bestScore

        return minimax()


def betterEvaluationFunction(currentGameState: GameState):
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    # Calcula a distância de Manhattan para a comida mais próxima
    foodDistances = [manhattanDistance(pos, f) for f in food]
    if len(foodDistances) > 0:
        minFoodDistance = min(foodDistances)
    else:
        minFoodDistance = 0

    # Distância para o fantasma mais próximo
    ghostDistances = [manhattanDistance(pos, ghost.getPosition()) for ghost in ghostStates]
    minGhostDistance = min(ghostDistances)

    # Aumenta a pontuação se o fantasma estiver assustado, mas penaliza se estiver muito perto
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    if min(scaredTimes) > 0:
        minGhostDistance = 0  # Ignora fantasmas assustados

    return currentGameState.getScore() - (1.5 / (minFoodDistance + 1)) + (2 / (minGhostDistance + 1))

# Abbreviation
better = betterEvaluationFunction