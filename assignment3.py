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
    Function to determine allocation for preparing breakfast and dinner between 5 roommates (numbered 0, 1, 2, 3, 4)
    in the next n days (numbered 0, 1, ..., n - 1). Ideally divided such that each meal is assigned to one person
    and each person is assigned exactly 2n/5 meals. Allocation is done based on each person's availability.

    However, a perfect allocation might not be possible. So a meal can be ordered from a restaurant if needed (the
    number of meals that are ordered should not exceed 0.1n) and each person should be allocated to at least 0.36n
    meals and at most 0.44n meals. No person should be allocated to both meals of the same day.

    :Input:
        availability: list of lists that contains data about the time availability of each person.
                      For each day numbered i and each person numbered j, availability[i][j] is:
                      - 0, if person j is not able to prepare both breakfast and dinner on day i
                      - 1, if person j is only able to prepare breakfast on day i
                      - 2, if person j is only able to prepare dinner on day i
                      - 3, if person j is able to prepare both breakfast and dinner on day i

    :Output:
        - None, if an allocation that satisfies all constraints does not exist.
        - (breakfast, dinner), where lists breakfast and dinner each specify a valid allocation.
          breakfast[i] = j, if person j is allocated to prepare breakfast on day i
          breakfast[i] = 5, if breakfast on day i will be ordered from a restaurant
          dinner[i] = j, if person j is allocated to prepare dinner on day i
          dinner[i] = 5, if dinner on day i will be ordered from a restaurant

    :Time Complexity: O(n^2), where n is the number of days
    :Aux Space Complexity: O(n^2), where n is the number of days

    :Approach:
    This function uses the concept of circulation with demands and lower bounds problem. To solve it, the network flow
    is broken down into two graphs, graph_min_flow and graph_adjusted. graph_min_flow has capacity equal to the
    lower bound for the flow going into each person, while graph_adjusted has capacity equal to the remainder of
    flow available from the minimum network flow. We use Ford-Fulkerson's algorithm to find the maximum flow in
    both networks.

    A valid allocation exists if all outgoing edges from the source node in both graphs are saturated (i.e. has full
    capacity). We then merge the findings from these two graphs to construct the breakfast and dinner arrays.
    """
    # GENERAL INFORMATION
    num_persons = 5
    n = len(availability)  # n is the number of days

    # REQUIREMENTS
    min_meals = math.floor(0.36 * n)
    demand_min_flow = min_meals * num_persons             # represents the minimum meals to be prepared by all roommates
    demand_adjusted = (n * 2) - (min_meals * num_persons) # represents the rest of meals to be prepared

    # Generating network flow with capacity equal to the lower bound
    graph_min_flow = generate_network_min_flow(availability)
    breakfast_min_flow, dinner_min_flow, max_flow_min = ford_fulkerson(availability, graph_min_flow)

    # Generating network flow with adjusted capacities
    graph_adjusted = generate_network_adjusted(availability, breakfast_min_flow, dinner_min_flow)
    breakfast_adjusted, dinner_adjusted, max_flow_adjusted = ford_fulkerson(availability, graph_adjusted)

    # Check if the outgoing edges from the source are saturated in both graph_min_flow and graph_adjusted
    if max_flow_min < demand_min_flow or max_flow_adjusted < demand_adjusted:
        # Then it means no valid allocation exists and return None
        return None

    # Merge breakfast_min_flow and breakfast_adjusted into final breakfast array
    breakfast = merge_results(breakfast_min_flow, breakfast_adjusted)

    # Merge dinner_min_flow and dinner_adjusted into final dinner array
    dinner = merge_results(dinner_min_flow, dinner_adjusted)

    return breakfast, dinner


def generate_network_min_flow(availability):
    """
    Function that returns a network in adjacency list format.

    :Input:
        availability: list of lists that contains data about the time availability of each person.

    :Output:
        graph: an adjacency matrix that represents the network flow for calculating the meal allocation,
               this network flow has capacities equal to the lower bound needed in the allocation

    :Time Complexity: O(n)
    :Aux Space Complexity: O(v + e), where v is the number of vertices in the network and e is the number of edges
                           in the network (Here, both v and e are bounded by n)

    Nodes:
        - Source node
        - Meals node
        - 5 person nodes + 1 order option node
        - n day nodes
        - (n * 2) meal nodes
        - Sink node

    Network Visualization:
    n = 5 days

                 <<Edge for each      <<Edge for every
                 day that person      meal that can be prepared
                 is able to prepare   by a person>>
                 a meal>>
           |-->P0------------------>D0--------------->M0----|
           |                           |------------->M1----|
           |-->P1                   D1                M2----|   <<Edge from every
           |                                          M3----|   meal to sink node>>
           |-->P2                   D2                M4----|
    S-->M--|                                          M5----------->T
           |-->P3                   D3                M6----|
           |                                          M7----|
           |-->P4                   D4                M8----|
                                                      M9----|
               P5                   D5                M10---|
           <<Here, P5 which is the order option node,
           does not have any edges connected to it
           because the lower bound for it is 0>>

    :Approach:
    We add an edge from source node to meals node with capacity of the minimum meals that need to be prepared by all
    roommates. Then we add an edge from the meals to each person with capacity of the minimum number of meals prepared
    by each person. Then from each person, we add an edge to each day they are able to prepare a meal with capacity
    of 1. We add an edge from each day to its corresponding breakfast and dinner nodes with capacity of 1. And finally
    we add an edge from every meal node to the sink node with capacity of 1.

    """
    # GENERAL INFORMATION
    num_persons = 5
    n = len(availability)   # n is the number of days

    # REQUIREMENTS
    min_meals = math.floor(0.36 * n)
    demand = min_meals * num_persons  # represents the minimum meals to be prepared by all roommates

    # NODE INFORMATION
    source_node = 0
    meals_node = 1
    start_person_nodes = 2
    start_day_nodes = 8
    start_meal_nodes = start_day_nodes + n
    num_nodes = 1 + 1 + (num_persons + 1) + (n * 3) + 1
    # source + meals + (num of people + order option) + (num of days * 3) + sink

    # Generate empty adjacency list
    graph = [[] for _ in range(num_nodes)]                                              # O(n)

    # Fill adjacency list
    # Edge from source node to meals node
    graph[source_node].append([meals_node, demand])

    # Edge from meals to each person (with capacity equal to the lower bound)
    for i in range(num_persons):                                                        # O(1)
        graph[meals_node].append([i + start_person_nodes, min_meals])

    # No edge from meals to order option (Not required)

    # No edge from order option to every meal (Not required)

    # Edge from each person to each day they can prepare a meal
    # and edge from each day to each meal
    for day in range(n):                                                                # O(n)
        for person in range(num_persons):                                               # O(1)
            # If the person can prepare a meal for that day
            if availability[day][person] > 0:
                # Add edge from person to the day
                # Capacity is one since each person can only prepare one meal for each day
                graph[person + start_person_nodes].append([day + start_day_nodes, 1])

    # Edge from every day to its breakfast and dinner nodes
    for day in range(n):                                                                # O(n)
        graph[day + start_day_nodes].append([(day * 2) + start_meal_nodes, 1])
        graph[day + start_day_nodes].append([(day * 2) + start_meal_nodes + 1, 1])

    # Edge from each meal to super sink node
    for i in range(n * 2):                                                              # O(n)
        graph[i + start_meal_nodes].append([num_nodes - 1, 1])

    return graph


def generate_network_adjusted(availability, breakfast_min_flow, dinner_min_flow):
    """
    Function that returns a network in adjacency list format.

    :Input:
        availability: list of lists that contains data about the time availability of each person.

    :Output:
        graph: an adjacency matrix that represents the network flow for calculating the meal allocation,
               this network flow has capacities equal to the remainder flow available after allocation
               using the min flow network

    :Time Complexity: O(n)
    :Aux Space Complexity: O(v + e), where v is the number of vertices in the network and e is the number of edges
                           in the network (Here, both v and e are bounded by n)

    Nodes:
        - Source node
        - Meals node
        - 5 person nodes + 1 order option node
        - n day nodes
        - (n * 2) meal nodes
        - Sink node

    Network Visualization:
    n = 5 days

                 <<Edge for each      <<Edge for every
                 day that person      meal that can be prepared
                 is able to prepare   by a person>>
                 a meal>>
           |-->P0------------------>D0--------------->M0----|
           |                           |------------->M1----|
           |-->P1                   D1                M2----|   <<Edge from every
           |                                          M3----|   meal to sink node>>
           |-->P2                   D2                M4----|
    S-->M--|                                          M5----------->T
           |-->P3                   D3                M6----|
           |                                          M7----|
           |-->P4                   D4                M8----|
           |                                          M9----|
           |-->P5                   D5                M10---|
           <<Here, we add an edge to P5 which is
           the order option node, >>

    :Approach:
    We add an edge from source node to meals node with capacity of the remainder of meals that need to be prepared by
    all roommates after accounting for the minimum. Then we add an edge from the meals to each person with capacity of
    the remainder number of meals that can be prepared by each person (i.e. max meals - min meals). Then from each
    person, we add an edge to each day they are able to prepare a meal with capacity of 1. However, here we do an
    extra check and only add the edge if their availability fills an unallocated meal from the previous network.
    We add an edge from each day to its corresponding breakfast and dinner nodes with capacity of 1. And finally
    we add an edge from the meal node to the sink node with capacity of 1.
    """
    # GENERAL INFORMATION
    num_persons = 5
    n = len(availability)  # n is the number of days

    # REQUIREMENTS
    min_meals = math.floor(0.36 * n)
    max_meals = math.ceil(0.44 * n)
    max_order = math.ceil(0.1 * n)
    demand = (n * 2) - (min_meals * num_persons)  # represents the rest of meals to be prepared

    # NODE INFORMATION
    source_node = 0
    meals_node = 1
    start_person_nodes = 2
    order_option_node = 7
    start_day_nodes = 8
    start_meal_nodes = start_day_nodes + n
    num_nodes = 1 + 1 + (num_persons + 1) + (n * 3) + 1
    # source + meals + (num of people + order option) + (num of days * 3) + sink

    # Generate empty graph
    graph = [[] for _ in range(num_nodes)]              # O(n^2)

    # Fill graph
    # Edge from source node to meals node
    graph[source_node].append([meals_node, demand])

    # Edge from meals to each person (with capacity equal to the remainder of meals possible for the person)
    for i in range(num_persons):
        graph[meals_node].append([i + start_person_nodes, max_meals - min_meals])

    # Edge from meals to order option
    graph[meals_node].append([order_option_node, max_order])

    # Edge from order option to every meal
    # Taking into account the allocated meals given by breakfast_min_flow and dinner_min_flow
    # Only make an edge if that meal is not allocated a person yet
    for meal in range(n * 2):
        day = meal // 2
        # If meal is equal to modulo 2 of n, it is a breakfast node
        if meal % 2 == (n % 2):
            # If the breakfast for that day has been allocated, do not add an edge
            if breakfast_min_flow[day] != -1:
                continue
        # Else if meal is not equal to modulo 2 of n, it is a dinner node
        else:
            # If the dinner for that day has been allocated, do not add an edge
            if dinner_min_flow[day] != -1:
                continue

        graph[order_option_node].append([meal + start_meal_nodes, 1])
        graph[meal + start_meal_nodes].append([num_nodes - 1, 1])

    # Edge from each person to each day they can prepare a meal
    # and edge from each day to each meal
    # Taking into account the allocated meals given by breakfast_min_flow and dinner_min_flow
    # Only make an edge if that meal is not allocated a person yet
    for day in range(n):
        for person in range(num_persons):
            # If the person can prepare breakfast for that day and breakfast for that day is not allocated yet
            if availability[day][person] == 1 and breakfast_min_flow[day] == -1:
                # Add edge from person to the day
                # Capacity is one since each person can only prepare one meal for each day
                graph[person + start_person_nodes].append([day + start_day_nodes, 1])
                # If the meal node is not connected to sink yet, add edge from day to meal
                if len(graph[(day * 2) + start_meal_nodes]) == 0:
                    graph[day + start_day_nodes].append([(day * 2) + start_meal_nodes, 1])
                    # Edge from the meal to super sink node
                    graph[(day * 2) + start_meal_nodes].append([num_nodes - 1, 1])

            # If the person can prepare dinner for that day and dinner for that day is not allocated yet
            elif availability[day][person] == 2 and dinner_min_flow[day] == -1:
                # Add edge from person to the day
                graph[person + start_person_nodes].append([day + start_day_nodes, 1])
                # If the meal node is not connected to sink yet, add edge from day to meal
                if len(graph[(day * 2) + start_meal_nodes + 1]) == 0:
                    graph[day + start_day_nodes].append([(day * 2) + start_meal_nodes + 1, 1])
                    # Edge from the meal to super sink node
                    graph[(day * 2) + start_meal_nodes + 1].append([num_nodes - 1, 1])

            # If the person can prepare breakfast and dinner for that day
            # and only dinner has been allocated (breakfast is not allocated yet)
            elif availability[day][person] == 3 and dinner_min_flow[day] > -1 and breakfast_min_flow[day] == -1\
                    and dinner_min_flow[day] != person:
                # Add edge from person to the day
                graph[person + start_person_nodes].append([day + start_day_nodes, 1])
                # If the meal node is not connected to sink yet, add edge from day to meal
                if len(graph[(day * 2) + start_meal_nodes]) == 0:
                    graph[day + start_day_nodes].append([(day * 2) + start_meal_nodes, 1])
                    # Edge from the meal to super sink node
                    graph[(day * 2) + start_meal_nodes].append([num_nodes - 1, 1])

            # If the person can prepare breakfast and dinner for that day
            # and only breakfast has been allocated (dinner is not allocated yet)
            elif availability[day][person] == 3 and breakfast_min_flow[day] > -1 and dinner_min_flow[day] == -1\
                    and breakfast_min_flow[day] != person:
                # Add edge from person to the day
                graph[person + start_person_nodes].append([day + start_day_nodes, 1])
                # If the meal node is not connected to sink yet, add edge from day to meal
                if len(graph[(day * 2) + start_meal_nodes + 1]) == 0:
                    graph[day + start_day_nodes].append([(day * 2) + start_meal_nodes + 1, 1])
                    # Edge from the meal to super sink node
                    graph[(day * 2) + start_meal_nodes + 1].append([num_nodes - 1, 1])

            # If the person can prepare breakfast and dinner for that day
            # and both breakfast and dinner is not allocated yet
            elif availability[day][person] == 3 and breakfast_min_flow[day] == -1 and dinner_min_flow[day] == -1:
                # Add edge from person to the day
                graph[person + start_person_nodes].append([day + start_day_nodes, 1])
                # If the meal node is not connected to sink yet, add edge from day to meal
                if len(graph[(day * 2) + start_meal_nodes]) == 0:
                    graph[day + start_day_nodes].append([(day * 2) + start_meal_nodes, 1])
                    # Edge from the meal to super sink node
                    graph[(day * 2) + start_meal_nodes].append([num_nodes - 1, 1])
                # If the meal node is not connected to sink yet, add edge from day to meal
                if len(graph[(day * 2) + start_meal_nodes + 1]) == 0:
                    graph[day + start_day_nodes].append([(day * 2) + start_meal_nodes + 1, 1])
                    # Edge from the meal to super sink node
                    graph[(day * 2) + start_meal_nodes + 1].append([num_nodes - 1, 1])

    return graph


def bfs(availability, network, parent):
    """
    Function to perform breadth-first-search on a network.

    :Input:
        availability: list of lists that contains data about the time availability of each person.
        network: adjacency list representation of network to apply BFS to
        parent: list that stores the immediate parent of a node in the BFS path
    :Output:
        Returns True if there exists a simple path from the source node to the sink node.
        Returns False otherwise.

    :Time Complexity: O(v + e), where v is the number of vertices in the network and e is the number of edges
                      in the network (Here, both v and e are bounded by n)
    :Aux Space Complexity: O(n)

    :Approach:
    We do regular breadth first search, but we do a check to see if we have reached the sink node. If so, return True
    and stop the search. This way, this bfs method is able to tell whether there is an augmenting path inside the
    network.

    We also do an extra check everytime we reach a meal node to check if the connected person is actually available for
    that meal (breakfast or dinner), because a person might be connected to the day but not connected to either the
    breakfast or the dinner. If they are not available for that meal, then we ignore that path.
    """
    n = len(availability)
    source = 0
    sink = len(network) - 1

    # NODE INFORMATION
    start_person_nodes = 2
    start_day_nodes = 8
    start_meal_nodes = start_day_nodes + n

    # Array to keep track of which nodes have been visited
    # Initialize all nodes as not visited
    visited = [False] * len(network)                            # O(n) space

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
            # If an adjacent node has a weight of more than 0 and of curr has not been visited
            adjacent_node = network[curr][i][0]
            weight_adjacent_node = network[curr][i][1]
            if weight_adjacent_node > 0 and not visited[adjacent_node]:
                # If curr is a meal node
                if start_meal_nodes <= curr < start_meal_nodes + (n * 2):
                    # Check if the person is available for the meal
                    day = parent[curr] - start_day_nodes
                    person = parent[parent[curr]] - start_person_nodes

                    # If the person is not the order option node
                    if parent[curr] != 7:
                        # If curr is a breakfast node and the person can only prepare dinner or
                        # if curr is a dinner node and the person can only prepare breakfast
                        if (curr % 2 == (n % 2) and (availability[day][person] == 2)) or \
                                (curr % 2 != (n % 2) and (availability[day][person] == 1)):
                            # Ignore the path
                            continue

                # Append to the queue
                queue.append(adjacent_node)
                # Mark as visited
                visited[adjacent_node] = True
                # Mark curr as parent of the adjacent node
                parent[adjacent_node] = curr
                # Checking if we have reached the sink node
                if adjacent_node == sink:
                    # If yes, return True to indicate that we have found an augmenting path
                    return True

    # If we have done BFS and did not reach the sink node
    # Return False to indicate that we did not find an augmenting path
    return False


def ford_fulkerson(availability, network):
    """
    Function that implements Ford-Fulkerson algorithm for finding the maximum flow of a network.
    Contains modifications to record each time a path is augmented and stores the corresponding person
    inside breakfast and dinner arrays. The residual capacity of each augmentation is always 1
    since the minimum capacity is found from the person nodes to the day nodes (Flow can only either be
    0 or 1, i.e. available or not available).

    :Input:
        availability: list of lists that contains data about the time availability of each person.
        network: adjacency matrix representation of a residual network
    :Output:
        (breakfast, dinner, max_flow), where
        breakfast: array containing the person responsible to prepare each breakfast over n days
        dinner: array containing the person responsible to prepare each dinner over n days
        max_flow: the maximum flow found by the algorithm

    :Time Complexity: O(n^2)
    :Aux Space Complexity: O(n)

    :Approach:
    We do regular Ford-Fulkerson algorithm by augmenting a path while there exists one. We add a modification to
    record the breakfast and dinner allocations everytime we augment a path.
    """
    n = len(availability)
    source = 0
    sink = len(network) - 1

    # NODE INFORMATION
    start_person_nodes = 2
    order_option_node = 7
    start_day_nodes = 8
    start_meal_nodes = start_day_nodes + n

    # Initialize array to store parent of each node in the path
    parent = [-1] * len(network)                                    # O(n) space

    # Initialize maximum flow to 0
    max_flow = 0

    # Initialize breakfast and dinner arrays
    breakfast = [-1] * n                                            # O(n) space
    dinner = [-1] * n                                               # O(n) space

    # While there is an augmenting path in the network              # Total: O(n^2)
    while bfs(availability, network, parent):                       # Cost of running bfs once: O(n)
        # Starting from the sink node and going backwards until source
        curr = sink
        while curr != source:                                       # O (n)
            # If curr is a meal node
            if start_meal_nodes <= curr <= len(network) - 2:
                # If the parent of the meal is the order option node
                if parent[curr] == order_option_node:
                    day = (curr - start_meal_nodes) // 2
                    # If curr is equal to modulo 2 of n, it is a breakfast node
                    if curr % 2 == (n % 2):
                        breakfast[day] = 5
                    # If curr is not equal to modulo 2 of n, it is a dinner node
                    else:
                        dinner[day] = 5
                else:
                    # Record that the person will be preparing the corresponding meal
                    day = parent[curr] - start_day_nodes
                    person = parent[parent[curr]] - start_person_nodes

                    # If curr is equal to modulo 2 of n, it is a breakfast node
                    if curr % 2 == (n % 2):
                        breakfast[day] = person
                    # If curr is not equal to modulo 2 of n, it is a dinner node
                    else:
                        dinner[day] = person

            # Move backwards
            curr = parent[curr]

        # Add the residual capacity found to the maximum flow of the network
        max_flow += 1       # Residual capacity will always be 1

        # Updating the residual network by augmenting with the residual capacity
        curr = sink
        while curr != source:                                           # O(n)
            par = parent[curr]
            # Reducing the residual edge
            for i in range(len(network[par])):
                if network[par][i][0] == curr:
                    network[par][i][1] -= 1
                    break
            # Incrementing the backward edge
            for i in range(len(network[curr])):
                if network[curr][i][0] == par:
                    network[curr][i][1] += 1
                    break

            # Move backwards
            curr = parent[curr]

    return breakfast, dinner, max_flow


def merge_results(min_flow_results, adjusted_results):
    """
    Function to merge the results of maximizing the flow of network with min flow and
    maximizing the flow of network with adjusted capacities.

    :Input:
        min_flow_results: list of length n which stores the person responsible for preparing a meal
        adjusted_results: list of length n which stores the person responsible for preparing a meal
    :Output:

    :Time Complexity: O(n)
    :Aux Space Complexity: O(n)
    """
    res = []

    # Iterating through each value in the results
    for i in range(len(min_flow_results)):
        # If the value is filled (i.e. not the default value -1)
        # Append the value to results
        if min_flow_results[i] > -1:
            res.append(min_flow_results[i])
        elif adjusted_results[i] > -1:
            res.append(adjusted_results[i])

    return res


# Task 2 - Similarity Detector
def compare_subs(submission1, submission2):
    """
    Function that uses a retrieval data structure to compare two submissions and determine their similarity.
    :Input:
        submission1: first string to compare containing characters in the range [a-z] or space
        submission2: second string to compare containing characters in the range [a-z] or space
    :Output:
        res: list of findings with three elements:
            - the longest common substring between submission1 and submission2
            - the similarity score for submission1 (expressed as the percentage of submission1
              that belongs to the longest common substring, rounded to the nearest integer)
            - the similarity score for submission2 (expressed as the percentage of submission2
              that belongs to the longest common substring, rounded to the nearest integer)

    :Time Complexity: O((N + M)^2), where N is the length of the first string and M is the length of the second string
    :Aux Space Complexity: O((N + M)^2), where N is the length of the first string and M is the length of the second
                           string

    :Approach:
    We create a suffix trie that stores only the start index and length of each substring in each node. We then
    compress this suffix trie into a suffix tree so that the space complexity taken by this tree is O(n).

    This suffix tree will store two strings. We add all the suffixes of the first string to the tree and then add
    all the suffixes of the second string.

    We then traverse the suffix tree using DFS (which will cost O(n) after compressing) to find the node with the
    maximum common length. We then build the longest common substring from this node. And finally we just calculate
    the similarity scores for both strings.
    """
    # Creating empty suffix tree
    suffix_tree = SuffixTree(submission1, submission2)

    # Build the suffix tree
    suffix_tree.build()

    # Traversing the suffix tree to find the node with largest commonLength
    max_node = findMaxNode(suffix_tree)

    # Building the longest common substring
    longest_common_substring = build_longest_common_substring(max_node, submission1)

    # Calculating similarity_score1 and similarity_score2
    similarity_score1 = round(suffix_tree.maxLength / len(submission1) * 100)
    similarity_score2 = round(suffix_tree.maxLength / len(submission2) * 100)

    return [longest_common_substring, similarity_score1, similarity_score2]


class SuffixTree:
    """
    Class that represents a suffix tree.
    """
    def __init__(self, string1, string2):
        """
        Constructor for SuffixTree class.
        :Input:
            string1: First string to add to the suffix tree
            string2: Second string to add to the suffix tree
        """
        self.string1 = string1
        self.string2 = string2
        self.children = []
        self.commonLength = 0
        self.maxLength = 0
        self.counter = 0

    def addSuffix(self, startNode, startIndex, length, isPartOfString1, isPartOfString2):
        """
        Method to add a suffix into the suffix tree.
        :Input:
            startNode: suffix tree node we are adding the suffix to
            startIndex: starting index of the suffix to add
            length: length of the suffix to add
            isString1: boolean that represents whether the suffix is from string1
            isPartOfString1: boolean that represents whether the suffix is part of string1
            isPartOfString2: boolean that represents whether the suffix is part of string2
        :Post-condition: the suffix is added into the structure of the suffix tree

        :Approach:
        We iterate through each child in the startNode's children and check if the first letter is the same as the first
        letter of the suffix to add. If so, we break down the node by length of 1, i.e. we add the remainder of the
        node's substring as a child of that node and set its length to 1, and also add the remainder of the suffix to
        add as a child as well. This will be done recursively. We also update the isPartOfString2 when adding and
        keep track of the common length. We also update the maxLength of the whole tree if a node's commonLength is
        larger than the current one. We will use this maxLength value when finding the node with the maximum common
        length using the DFS.
        """
        self.counter += 1
        matching_letter_found = False
        if isPartOfString1:
            string = self.string1
        else:
            string = self.string2

        # Iterate through the nodes connected to the starting node
        for node in startNode.children:
            if node.isPartOfString1:
                string_node = self.string1
            else:
                string_node = self.string2
            # TODO: modify here so that it skips over if the letters are the same, don't break down immediately
            # If the first character in the suffix to add matches the start index of the connected node
            if string[startIndex] == string_node[node.startIndex]:
                # If exists, make the remainder of the length into the matched node's child
                if node.length > 1:
                    self.addSuffix(node, node.startIndex + 1, node.length - 1, node.isPartOfString1,
                                   node.isPartOfString2)
                    # And un-mark the node as an end
                    node.isEnd = False
                # Make the matched node into a node of length 1
                node.length = 1
                # Update isPartOfString2 of the matched node
                if isPartOfString2:
                    node.isPartOfString2 = True
                # If the matched node is part of both string1 and string2
                # Update the common length
                if node.isPartOfString1 and node.isPartOfString2:
                    node.commonLength = startNode.commonLength + 1
                    # If the node's commonLength is greater than the current maxLength of the tree
                    if node.commonLength > self.maxLength:
                        self.maxLength = node.commonLength
                # If the suffix to add is only one letter, then just mark the matched node as an end
                if length == 1:
                    node.isEnd = True
                # Else, make the remainder of the suffix to add a child of the matched node
                else:
                    self.addSuffix(node, startIndex + 1, length - 1, isPartOfString1, isPartOfString2)

                # Mark matched_letter_found as True and break out of the for loop
                matching_letter_found = True
                break

        # Else if we have iterated through all the startNode's children and did not find a matching letter
        if not matching_letter_found:
            # Append a new child to the startNode
            startNode.children.append(SuffixTreeNode(startIndex, length, True, isPartOfString1, isPartOfString2))

    def build(self):
        """
        Method to build (i.e. fill up) the suffix tree

        :Time Complexity: O((N+M)^2), where N is the length of the first string and M is the length of the second string
        :Aux Space Complexity: O((N+M)^2), where N is the length of the first string and M is the length of the second
                               string
        """
        # Inserting suffixes of submission1 into the suffix tree
        for startIndex in range(len(self.string1)):
            self.addSuffix(self, startIndex, len(self.string1) - startIndex, True, False)

        # Inserting suffixes of submission2 into the suffix tree
        for startIndex in range(len(self.string2)):
            self.addSuffix(self, startIndex, len(self.string2) - startIndex, False, True)

        # Compress to a suffix tree
        self.compress()

    def compress(self):
        """
        Method to compress the suffix trie built into a suffix tree with O(n) space.

        :Input: none
        :Post-condition: the suffix trie is compressed into a suffix tree with O(N +M) space

        :Time Complexity: O(N + M), where N is the length of the first string and M is the length of the second string
        :Aux Space Complexity: O(1)

        :Approach:
        Using DFS approach, we check whether the current node only has one child, if so, also check if they are both
        part of both string1 and string2. If so, compress them into one node by incrementing the length of the current
        node and setting the children of the child as the current node's children.
        """
        # Iterate through each node in the suffix tree using DFS
        # Stack to keep track of which nodes to visit first
        # Visit the root's children first
        stack = [node for node in self.children]

        # Depth First Search
        while stack:
            # Pop the node to visit
            curr = stack.pop()

            # If the curr only has one child, and they are part of the same string
            if len(curr.children) == 1 and (curr.isPartOfString1 and curr.children[0].isPartOfString1) and \
                    (curr.isPartOfString2 and curr.children[0].isPartOfString2):
                # Do node compression
                # Increment curr's length
                curr.length += 1
                # Update isEnd
                curr.isEnd = curr.children[0].isEnd
                # Update commonLength
                curr.commonLength = curr.children[0].commonLength
                # Set the child's children as curr's children
                curr.children = curr.children[0].children
                # Re-insert the node to the stack
                stack.append(curr)
                continue

            # Append curr's children to the stack
            for child in curr.children:
                stack.append(child)


class SuffixTreeNode:
    """
    Class that represents a suffix tree node.
    """
    def __init__(self, startIndex, length, isEnd, isPartOfString1, isPartOfString2):
        """
        Constructor for SuffixTreeNode class.
        :Input:
            startIndex: starting index of the substring in the node
            length: length of the substring in the node
            isEnd: boolean that indicates whether the node is
            isPartOfString1: boolean that indicates if the suffix to add is part of string1
            isPartOfString2: boolean that indicates if the suffix to add is part of string2
        """
        self.startIndex = startIndex
        self.length = length
        self.children = []
        self.isEnd = isEnd
        self.isPartOfString1 = isPartOfString1
        self.isPartOfString2 = isPartOfString2
        self.commonLength = 0
        self.parent = None

    def __str__(self):
        return "Start Index: " + str(self.startIndex) + ", length: " + str(self.length) + ", isEnd: " + \
               str(self.isEnd) + ", string1: " + str(self.isPartOfString1) + ", string2: " + str(self.isPartOfString2) \
               + ", commonLength: " + str(self.commonLength)


def findMaxNode(suffixTree):
    """
    Function that returns the node with commonLength equal to the maxLength (i.e. the last node of the
    longest common string)

    :Input: SuffixTree object that represents the suffix tree to traverse
    :Output: SuffixTreeNode object that represents the node in the suffix tree that has a commonLength attribute
             equal to the maxLength in the tree

    :Time Complexity: O(N + M)
    :Aux Space Complexity: O(1)

    :Approach:
    Uses DFS concept to traverse the tree, only appending to the stack if the node is part of both strings.
    If we have reached a node that does not have any children that belongs to both nodes, it means we have
    reached the end of the common sub-string of both strings. Once we have reached this node, check whether the
    commonLength is equal to maxLength of the tree, and if so return the node.
    """
    # Stack to keep track of which nodes to visit first
    # Visit the root first
    stack = [suffixTree]

    # Depth First Search
    while stack:
        # Pop the node to visit
        curr = stack.pop()

        # Initialize common_letter_found
        common_letter_found = False

        # Looping through curr's children
        for node in curr.children:
            # If the node is part of both string1 and string2
            # It means we should visit it
            if node.isPartOfString1 and node.isPartOfString2:
                # Append to the stack
                stack.append(node)
                # Mark curr as parent of the node
                node.parent = curr
                # Mark common_letter_found to True
                common_letter_found = True

        # If we have iterated through all the current node's children and did not find another common letter
        # It means we have reached the end of the common substring
        if not common_letter_found:
            # Check if it is the maxLength substring
            if curr.commonLength == suffixTree.maxLength:
                # If yes, we have found the node with max commonLength, and return it
                return curr


def build_longest_common_substring(max_node, string1):
    """
    Function to construct the longest common sub-string from the maxNode.

    :Input:
        maxNode: SuffixTreeNode object with commonLength equal to the maxLength of the suffix tree (i.e. the node
                 that gets returned by the findMaxNode function)
        string1: the first string of the suffix tree
    :Output:
        longest_common_substring: the longest common substring in the suffix tree

    :Time Complexity: O(N + M)
    :Aux Space Complexity: O((N + M))

    :Approach:
    While we have not reached the root node, append the current node's substring to the result. We set the parent as
    the current node and repeat.
    """
    longest_common_substring = []
    curr_node = max_node
    while curr_node.commonLength != 0:
        longest_common_substring.append(string1[curr_node.startIndex:curr_node.startIndex + curr_node.length])
        curr_node = curr_node.parent
    longest_common_substring = "".join(longest_common_substring[::-1])
    return longest_common_substring
