# `multiAgents.py` Documentation

This document explains the key classes and functions in the `multiAgents.py` file. It provides a breakdown of each agent class and its purpose, as well as how they interact with the Pacman game environment.

---

## **1. ReflexAgent**

### **Description**

The `ReflexAgent` is a simple agent that decides its actions based on a state evaluation function. It evaluates all possible actions and selects one that maximizes its score.

### **Key Methods**

- **`getAction(gameState)`**
  - Chooses an action by evaluating all legal moves and selecting the best one based on the evaluation function.
  - Returns a direction from `{NORTH, SOUTH, EAST, WEST, STOP}`.
- **`evaluationFunction(currentGameState, action)`**
  - A customizable function to evaluate the desirability of a state resulting from an action.
  - Example logic in the file:
    - Prefers actions that minimize the distance to food.
    - Maximizes the distance to ghosts.

---

## **2. MultiAgentSearchAgent**

### **Description**

`MultiAgentSearchAgent` serves as an abstract base class for adversarial agents like `MinimaxAgent`, `AlphaBetaAgent`, and `ExpectimaxAgent`. It includes utility methods and initialization for all multi-agent search strategies.

### **Attributes**

- `evalFn`: A reference to the evaluation function (default: `scoreEvaluationFunction`).
- `depth`: The search depth.

---

## **3. MinimaxAgent**

### **Description**

Implements the minimax algorithm, where Pacman (the maximizing agent) tries to maximize its score, while ghosts (minimizing agents) try to minimize Pacman's score.

### **Key Methods**

- **`getAction(gameState)`**
  - Returns the best action using the minimax strategy with the specified depth.
- **`min_max_value(game_state, agent_index, depth)`**
  - Recursively evaluates game states for maximizing (Pacman) and minimizing (ghosts) agents.
  - Stops evaluation if the game is won, lost, or the depth limit is reached.
- **`compute_best_option(game_state, agent_index, depth, best_function)`**
  - Calculates the best action for either the maximizing or minimizing agent.

---

## **4. AlphaBetaAgent**

### **Description**

A variation of the minimax agent that uses alpha-beta pruning to eliminate branches that cannot influence the final decision. This reduces the number of nodes evaluated and improves efficiency.

### **Key Methods**

- **`getAction(gameState)`**
  - Returns the best action using alpha-beta pruning.
- **`min_max_value(game_state, agent_index, alpha, beta, depth)`**
  - Evaluates game states while applying pruning based on the `alpha` (max bound) and `beta` (min bound) values.
- **`max_value` and `min_value`**
  - Specialized methods for handling maximizing and minimizing agents, respectively.

---

## **5. ExpectimaxAgent**

### **Description**

Implements the expectimax algorithm, which is similar to minimax but assumes that ghosts act randomly instead of adversarially. Useful for modeling scenarios where opponents don't optimize their strategy.

### **Key Methods**

- **`getAction(gameState)`**
  - Returns the best action using the expectimax strategy.
- **`expect_max_value(game_state, agent_index, depth)`**
  - Evaluates game states using the expectimax approach.
- **`expect_value(game_state, agent_index, depth)`**
  - Computes the expected value for a ghost's random actions.

---

## **6. `betterEvaluationFunction`**

### **Description**

A sophisticated evaluation function for optimizing Pacman's gameplay. It considers multiple factors:

- Distance to the nearest food and ghosts.
- Number of remaining food and capsules.
- Winning or losing states.

### **Key Features**

- Rewards Pacman for:
  - Collecting food quickly.
  - Staying far from ghosts.
  - Winning the game.
- Penalizes Pacman for:
  - Losing the game.
  - Delays in collecting food or capsules.

---

## **7. Utility Functions**

### **`scoreEvaluationFunction`**

- The default evaluation function, which simply returns the game state's score.

### **`distance_to_nearest(game_state, positions)`**

- Computes the Manhattan distance to the nearest position (e.g., food or ghost).

### **`manhattan_distance(pos1, pos2)`**

- Helper function to calculate Manhattan distance between two positions.

---

## **Abbreviations**

- `better`: Refers to the `betterEvaluationFunction`.

---

This file forms the backbone for implementing various AI strategies for Pacman, including reflex agents, adversarial search, and probabilistic planning. Each class and function is designed to enable experimentation and customization.
