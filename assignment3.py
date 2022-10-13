"""
FIT2004 Assignment 3
Author: Emily Samantha Zarry
ID: 32558945

"""
import math
from collections import deque


# Task 1 - Sharing Meals
def allocate(availability):
    graph = generate_network(availability)
    return ford_fulkerson(graph, 0, len(graph) - 1, len(availability))


def generate_network(availability):
    # GENERAL INFORMATION
    num_persons = 5
    n = len(availability)       # n is the number of days
    demand = n * 2              # represents the number of meals to be prepared (num of days * 2)
    num_nodes = 1 + 1 + (num_persons + 1) + (n * 3) + 1
    # source + meals + (num of people + order option) + (num of days * 3) + sink

    # REQUIREMENTS
    min_meals = math.floor(0.36 * n)
    max_meals = math.ceil(0.44 * n)
    max_order = math.ceil(0.1 * n)

    # NODE INFORMATION
    start_person_nodes = 2
    order_option_node = 7
    start_day_nodes = 8
    start_meal_nodes = start_day_nodes + n

    # Generate empty graph
    graph = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

    # Fill graph
    # Edge from source node (0) to meals (1)
    graph[0][1] = demand

    # Edge from meals to each person
    for i in range(num_persons):
        graph[1][i + 2] = max_meals

    # Edge from meals to order option
    graph[1][7] = max_order

    # Edge from order option to every meal (starting from node 18)
    for i in range(n * 2):
        graph[order_option_node][i + start_meal_nodes] = 1

    # Edge from each person to each day they can prepare a meal
    # and edge from each day to each meal
    for day in range(n):
        for person in range(num_persons):
            # Add edge from person to the day
            # Capacity is one since each person can only prepare one meal for each day
            graph[person + start_person_nodes][day + start_day_nodes] = 1

            # If the person can prepare breakfast for that day
            if availability[day][person] == 1:
                # Add edge from the day to the corresponding meal of the day (i.e. breakfast)
                graph[day + start_day_nodes][(day * 2) + start_meal_nodes] = 1

            # If the person can prepare dinner for that day
            elif availability[day][person] == 2:
                # Add edge from the day to the corresponding meal of the day (i.e. dinner)
                graph[day + start_day_nodes][(day * 2) + start_meal_nodes + 1] = 1

            # If the person can prepare breakfast and dinner for that day
            elif availability[day][person] == 3:
                # Add edge to both breakfast and dinner for the day
                graph[day + start_day_nodes][(day * 2) + start_meal_nodes] = 1
                graph[day + start_day_nodes][(day * 2) + start_meal_nodes + 1] = 1

    # Edge from each meal to super sink node
    for i in range(n * 2):
        graph[i + start_meal_nodes][num_nodes - 1] = 1

    return graph


def bfs(network, source, sink, parent):
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
    # NODE INFORMATION
    start_person_nodes = 2
    order_option_node = 7
    start_day_nodes = 8
    start_meal_nodes = start_day_nodes + n

    # Initialize array to store parent of each node in the path
    parent = [-1] * len(network)

    # Initialize maximum flow to 0
    max_flow = 0

    # Initialize breakfast and dinner arrays
    breakfast = [-1] * n
    dinner = [-1] * n

    # While there is an augmenting path in the network
    while bfs(network, source, sink, parent):
        # Starting from the sink node and going backwards until source
        curr = sink
        while curr != source:
            # If curr is a meal node
            if start_meal_nodes <= curr <= len(network) - 2:
                # If the parent is the order node
                if parent[curr] == order_option_node:
                    day = (curr - start_meal_nodes) // 2
                    # If curr is even, it is a breakfast node
                    if curr % 2 == 0:
                        breakfast[day] = 5
                    # If curr is odd, it is a dinner node
                    else:
                        dinner[day] = 5
                # Otherwise, if the parent is a day node
                else:
                    # Record that the person will be preparing the corresponding meal
                    day = parent[curr] - start_day_nodes
                    person = parent[parent[curr]] - start_person_nodes

                    # If curr is even, it is a breakfast node
                    if curr % 2 == 0:
                        breakfast[day] = person
                    # If curr is odd, it is a dinner node
                    else:
                        dinner[day] = person

            # Move backwards
            curr = parent[curr]

        # Add the residual capacity found to the maximum flow of the network
        max_flow += 1       # Residual capacity will always be 1

        # Updating the residual network by augmenting with the found residual capacity
        curr = sink
        while curr != source:
            par = parent[curr]
            network[par][curr] -= 1      # Reducing the residual edge
            network[curr][par] += 1      # Incrementing the backward edge

            # Move backwards
            curr = parent[curr]

    return breakfast, dinner


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
