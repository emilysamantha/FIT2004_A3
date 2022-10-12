"""
FIT2004 Assignment 3
Author: Emily Samantha Zarry
ID: 32558945

"""
import math
from collections import deque


# Task 1 - Sharing Meals
def allocate(availability):
    """

    """
    pass


def bfs(network, source, sink, parent):
    """

    :Input:
        network: adjacency matrix representation of a network
        source:
        sink:
        parent:
    """
    # Array to keep track of which nodes have been visited
    # Initialize all nodes as not visited
    visited = [False] * len(network)

    # Queue to keep track of which nodes to visit first
    queue = deque()

    # Visit the source first
    # Append to the queue
    queue.append(source)
    # Mark as visited
    visited[source] = True

    # Breadth First Search
    while queue:
        # Pop the node to visit
        curr = queue.popleft()

        # Looping through curr's adjacent nodes
        for i in range(len(network[curr])):
            # If an adjacent node of curr has not been visited
            if network[curr][i] > 0 and not visited[i]:
                # Checking so that we do not select a path which includes
                # breakfast and dinner of the same day


                # Append to the queue
                queue.append(i)
                # Mark as visited
                visited[i] = True
                # Mark curr as parent of the adjacent node
                parent[i] = curr
                # Checking if we have reached the sink node
                if i == sink:
                    # If yes, return True to indicate that we have found an augmenting path
                    return True

    # If we have done BFS and did not reach the sink node
    # Return False to indicate that we did not find an augmenting path
    return False


def ford_fulkerson(network, source, sink):
    """

    :Input:
        network: adjacency matrix representation of a residual network
        source: integer representing the source node in the network
        sink: integer representation representing the sink node in the network
    """
    # Array to store the parent of each node in the path
    parent = [-1] * len(network)

    # Initialize maximum flow to 0
    max_flow = 0

    # While there is an augmenting path in the network
    while bfs(network, source, sink, parent):
        # Find residual capacity (minimum capacity) of augmenting path in the residual network
        residual_cap = float("Inf")

        # Starting from the sink node and going backwards until source
        curr = sink
        while curr != source:
            # Compare current minimum capacity with
            residual_cap = min(residual_cap, network[parent[curr]][curr])
            # Move backwards
            curr = parent[curr]

        # Add the residual capacity found to the maximum flow of the network
        max_flow += residual_cap

        # Updating the residual network by augmenting with the found residual capacity
        curr = sink
        while curr != source:
            par = parent[curr]
            network[par][curr] -= residual_cap      # Reducing the residual edge
            network[curr][par] += residual_cap      # Incrementing the backward edge

            # Move backwards
            curr = parent[curr]

    print(network)

    return max_flow


def circulation_lb(availability):
    # n is the number of days
    n = len(availability)

    # Calculating demand for super source and super sink nodes
    # The demand represents the number of meals to be prepared (num of days * 2)
    demand = n * 2

    # Calculating the number of nodes in the network
    # super sink + meals + (5 people + order option) + (num of days * 2) + super sink
    # 1 + 1 + (5 + 1) + (demand) + 1
    num_nodes = 1 + 1 + (5 + 1) + demand + 1

    # Generate graph
    graph = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

    # Fill graph
    # Edge from source node (0) to meals (1)
    graph[0][1] = demand

    # Edge from meals to each person
    for i in range(5):
        graph[1][i+2] = math.ceil(0.44 * n)

    # Edge from meals to order option
    graph[1][7] = math.ceil(0.1 * n)

    # Edge from order option to every meal
    for i in range(n * 2):
        graph[7][i + 8] = 1

    # Edge from each person to each meal they can prepare
    for day in range(n):
        for person in range(5):
            # If the person can prepare breakfast for that day
            if availability[day][person] == 1:
                graph[person + 2][(day * 2) + 8] = 1    # Capacity is 1 (each meal is prepared by one person only)
            # If the person can prepare dinner for that day
            elif availability[day][person] == 2:
                graph[person + 2][(day * 2) + 9] = 1
            # If the person can prepare breakfast and dinner for that day
            elif availability[day][person] == 3:
                graph[person + 2][(day * 2) + 8] = 1
                graph[person + 2][(day * 2) + 9] = 1

    # Edge from each meal to super sink node
    for i in range(n * 2):
        graph[i + 8][num_nodes - 1] = 1

    ford_fulkerson(graph, 0, num_nodes - 1)


# Testing task 1
network = [[0, 0, 3, 5, 0, 0],
           [0, 0, 0, 3, 0, 0],
           [0, 0, 0, 0, 5, 0],
           [0, 0, 3, 0, 0, 3],
           [0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 0]]

network2 = [[0, 4, 3, 0, 0, 0, 0],
            [0, 0, 3, 0, 3, 0, 0],
            [0, 0, 0, 0, 2, 3, 0],
            [0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 2],
            [0, 0, 0, 2, 0, 0, 5],
            [0, 0, 0, 0, 0, 0, 0]]

availability = [[2, 0, 2, 1, 2],
                [3, 3, 1, 0, 0],
                [0, 1, 0, 3, 0],
                [0, 0, 2, 0, 3],
                [1, 0, 0, 2, 1],
                [0, 0, 3, 0, 2],
                [0, 2, 0, 1, 0],
                [1, 3, 3, 2, 0],
                [0, 0, 1, 2, 1],
                [2, 0, 0, 3, 0]]

circulation_lb(availability)


# Task 2
