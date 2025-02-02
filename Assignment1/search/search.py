# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    initial_state = problem.getStartState()

    depth_stack = util.Stack()
    visited_nodes = []

    current_pair = (initial_state, [])
    depth_stack.push(current_pair)

    while not depth_stack.isEmpty():

        node_state, move_sequence = depth_stack.pop()

        if problem.isGoalState(node_state):
            return move_sequence

        if node_state in visited_nodes:
            continue

        visited_nodes.append(node_state)
        successor_states = problem.getSuccessors(node_state)
        for next_state, action, step_cost in successor_states:
            new_moves = move_sequence + [action]
            new_node = (next_state, new_moves)
            depth_stack.push(new_node)

    return -1



def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    start_node = problem.getStartState()

    bfs_queue = util.Queue()
    explored_set = []

    initial_pair = (start_node, [])
    bfs_queue.push(initial_pair)

    while not bfs_queue.isEmpty():

        state, path = bfs_queue.pop()

        if problem.isGoalState(state):
            return path

        if state in explored_set:
            placeholder = 0
            continue

        explored_set.append(state)
        successors = problem.getSuccessors(state)
        for successor, action, cost in successors:
            new_path = path + [action]
            successor_pair = (successor, new_path)
            bfs_queue.push(successor_pair)

    return -1


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    initial_state = problem.getStartState()

    priority_queue = util.PriorityQueue()
    visited_states = []
    cost_map = {}

    start_tuple = (initial_state, [], 0)
    priority_queue.push(start_tuple, 0)

    while not priority_queue.isEmpty():
        current_state, move_list, total_cost = priority_queue.pop()

        if problem.isGoalState(current_state):
            return move_list

        if current_state in visited_states and total_cost > cost_map.get(current_state, float('inf')):
            continue

        visited_states.append(current_state)
        cost_map[current_state] = total_cost

        successor_nodes = problem.getSuccessors(current_state)

        for next_node, action, step_cost in successor_nodes:
            updated_moves = move_list + [action]
            new_total_cost = total_cost + step_cost
            next_tuple = (next_node, updated_moves, new_total_cost)
            priority_queue.push(next_tuple, new_total_cost)

    return -1

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    priority_queue = util.PriorityQueue()
    initial_state = problem.getStartState()
    priority_queue.push((initial_state, 0, []), heuristic(initial_state, problem))

    # Dictionary to store the best g(n) value for each state
    g_score = {initial_state: 0}
    visited_states = set()  # To track visited states

    while not priority_queue.isEmpty():
        current_state, current_cost, path_taken = priority_queue.pop()

        # If we reach the goal state, return the path
        if problem.isGoalState(current_state):
            return path_taken

        # If the state has already been visited, skip it
        if current_state in visited_states:
            continue
        visited_states.add(current_state)

        # Expand the successors of the current state
        successors = problem.getSuccessors(current_state)
        for successor_state, action_taken, cost_to_successor in successors:
            new_cost = current_cost + cost_to_successor
            # If this is a better path (lower cost), update the g_score and push to the queue
            if successor_state not in g_score or new_cost < g_score[successor_state]:
                g_score[successor_state] = new_cost
                f_score = new_cost + heuristic(successor_state, problem)
                new_path = path_taken + [action_taken]
                priority_queue.push((successor_state, new_cost, new_path), f_score)

    # If no solution found, return empty path (failure)
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
