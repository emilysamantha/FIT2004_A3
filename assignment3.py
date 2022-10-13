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
    # n is the number of days
    # n = len(availability)

    # Initializing breakfast and dinner lists
    # breakfast = []
    # dinner = []

    max_flow, residual_network, breakfast, dinner = circulation_lb(availability)

    # If the max_flow found satisfies the number of meals to be prepared
    # if max_flow == len(availability) * 2:
    #     # Generate the breakfast and dinner lists based on the residual network
    #
    #     # Iterating through each meal node
    #     for i in range(18, len(residual_network) - 1):
    #         # Checking if the meal is connected to the order node (node 7)
    #         if residual_network[i][7] == 1:
    #             # If the node is even, it is a breakfast node
    #             if i % 2 == 0:
    #                 breakfast.append(5)
    #             # If the node is even, it is a dinner node
    #             else:
    #                 dinner.append(5)
    #
    #     # Iterating through each day node
    #     for i in range(8, 8 + n):
    #         # Iterating through each person node
    #         for j in range(2, 7):
    #             # If there is an edge between the person and the day
    #             if residual_network[i][j] == 1:
    #                 pass

    print(max_flow)
    return breakfast, dinner


def dfs(network, source, sink, parent):
    # Array to keep track of which nodes have been visited
    # Initialize all nodes as not visited
    visited = [False] * len(network)

    # Stack to keep track of which nodes to visit first
    # Visit the source first
    stack = [source]
    visited[source] = True

    # Depth First Search
    while stack:
        # Pop the node to visit
        curr = stack.pop()

        # If the node has been visited, move to the next node to visit
        if visited[curr]:
            continue

        # Looping through curr's adjacent node
        for i in range(len(network[curr])):
            # If an adjacent node of curr has not been visited
            if network[curr][i] > 0 and not visited[i]:
                # Append to the stack
                stack.append(i)
                # Mark as visited
                visited[i] = True
                # Mark curr as parent of the adjacent node
                parent[i] = curr
                # Checking if we have reached the sink node
                if i == sink:
                    # If yes, return True to indicate that we have found an augmenting path
                    return True

    # If we have done DFS and did not reach the sink node
    # Return False to indicate that we did not find an augmenting path
    return False


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


def ford_fulkerson(network, source, sink, n):
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

    # Breakfast and dinner arrays
    breakfast = [-1] * n
    dinner = [-1] * n

    # While there is an augmenting path in the network
    while bfs(network, source, sink, parent):
        # Find residual capacity (minimum capacity) of augmenting path in the residual network
        # residual_cap = float("Inf")

        # Starting from the sink node and going backwards until source
        curr = sink
        while curr != source:
            # Compare current minimum capacity with the current edge
            # residual_cap = min(residual_cap, network[parent[curr]][curr])

            # If curr is a meal node (between (n*2 + 2) and last node - 1)
            if 7 + n + 1 <= curr <= len(network) - 2:
                # If the parent is the order node (node 7)
                if parent[curr] == 7:
                    day = curr - (7+n) - 1
                    # If curr is even, it is a breakfast node
                    if curr % 2 == 0:
                        breakfast[day] = 5
                    # If curr is odd, it is a dinner node
                    else:
                        dinner[day] = 5
                else:
                    # Record that the person will be preparing the corresponding meal
                    day = parent[curr] - 8
                    person = parent[parent[curr]] - 2

                    # If curr is even, it is a breakfast node
                    if curr % 2 == 0:
                        breakfast[day] = person
                    # If curr is odd, it is a dinner node
                    else:
                        dinner[day] = person

            # Move backwards
            curr = parent[curr]

        # Add the residual capacity found to the maximum flow of the network
        max_flow += 1

        # Updating the residual network by augmenting with the found residual capacity
        curr = sink
        while curr != source:
            par = parent[curr]
            network[par][curr] -= 1      # Reducing the residual edge
            network[curr][par] += 1      # Incrementing the backward edge

            # Move backwards
            curr = parent[curr]

    return max_flow, network, breakfast, dinner


def circulation_lb(availability):
    # n is the number of days
    n = len(availability)

    # Calculating demand for super source and super sink nodes
    # The demand represents the number of meals to be prepared (num of days * 2)
    demand = n * 2

    # Calculating the number of nodes in the network
    # super sink + meals + (5 people + order option) + (num of days * 2) + super sink
    # 1 + 1 + (5 + 1) + (demand) + 1
    num_nodes = 1 + 1 + (5 + 1) + (n * 3) + 1

    # Generate empty graph
    graph = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

    # Fill graph
    # Edge from source node (0) to meals (1)
    graph[0][1] = demand

    # Edge from meals to each person
    for i in range(5):
        graph[1][i+2] = math.ceil(0.44 * n)

    # Edge from meals to order option
    graph[1][7] = math.ceil(0.1 * n)

    # Edge from order option (node 7) to every meal (starting from node 18)
    for i in range(n * 2):
        graph[7][i + 18] = 1

    # Edge from each person to each day they can prepare a meal
    # and edge from each day to each meal
    for day in range(n):
        for person in range(5):
            # Add edge from person to the day
            # Capacity is one since each person can only prepare one meal for each day
            graph[person + 2][day + 8] = 1

            # If the person can prepare breakfast for that day
            if availability[day][person] == 1:
                # Add edge from the day to the corresponding meal of the day (i.e. breakfast)
                graph[day + 8][((day + 8) * 2) + 2] = 1

            # If the person can prepare dinner for that day
            elif availability[day][person] == 2:
                # Add edge from the day to the corresponding meal of the day (i.e. dinner)
                graph[day + 8][((day + 8) * 2) + 3] = 1

            # If the person can prepare breakfast and dinner for that day
            elif availability[day][person] == 3:
                # Add edge to both breakfast and dinner for the day
                graph[day + 8][((day + 8) * 2) + 2] = 1
                graph[day + 8][((day + 8) * 2) + 3] = 1

    # Edge from each meal to super sink node
    for i in range(n * 2):
        graph[i + 18][num_nodes - 1] = 1

    return ford_fulkerson(graph, 0, num_nodes - 1, n)


# Testing task 1
network1 = [[0, 0, 3, 5, 0, 0],
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

availability1 = [[2, 0, 2, 1, 2],
                [3, 3, 1, 0, 0],
                [0, 1, 0, 3, 0],
                [0, 0, 2, 0, 3],
                [1, 0, 0, 2, 1],
                [0, 0, 3, 0, 2],
                [0, 2, 0, 1, 0],
                [1, 3, 3, 2, 0],
                [0, 0, 1, 2, 1],
                [2, 0, 0, 3, 0]]

print(allocate(availability1))

# Task 2
