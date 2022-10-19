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

    :Time Complexity: Needs to be O(n^2), where n is the number of days
    :Aux Space Complexity: Needs to be O(n^2), where n is the number of days

    :Approach:
    This function uses the concept of circulation with demands and lower bounds problem. To solve it, the network flow
    is broken down into two graphs, graph_min_flow and graph_adjusted. graph_min_flow has capacity equal to the
    lower bound, while graph_adjusted has capacity equal to the remainder of flow available from the main network flow.
    We use Ford-Fulkerson's algorithm for finding maximum flow to

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
    Function that returns a network in adjacency matrix format.

    :Input:
        availability: list of lists that contains data about the time availability of each person.

    :Output:
        graph: an adjacency matrix that represents the network flow for calculating the meal allocation,
               this network flow has capacities equal to the lower bound needed in the allocation

    :Time Complexity: O(n^2)
    :Aux Space Complexity: O(n^2), where n is the number of days

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

    # Generate empty graph
    graph = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]           # O(n^2)

    # Fill graph
    # Edge from source node to meals node
    graph[source_node][meals_node] = demand

    # Edge from meals to each person (with capacity equal to the lower bound)
    for i in range(num_persons):
        graph[meals_node][i + start_person_nodes] = min_meals

    # No edge from meals to order option (Not required)

    # No edge from order option to every meal (Not required)

    # Edge from each person to each day they can prepare a meal
    # and edge from each day to each meal
    for day in range(n):                        # O(n)
        for person in range(num_persons):
            # If the person can prepare breakfast for that day
            if availability[day][person] == 1:
                # Add edge from person to the day
                # Capacity is one since each person can only prepare one meal for each day
                graph[person + start_person_nodes][day + start_day_nodes] = 1
                # Add edge from the day to the corresponding meal of the day (i.e. breakfast)
                graph[day + start_day_nodes][(day * 2) + start_meal_nodes] = 1

            # If the person can prepare dinner for that day
            elif availability[day][person] == 2:
                # Add edge from person to the day
                graph[person + start_person_nodes][day + start_day_nodes] = 1
                # Add edge from the day to the corresponding meal of the day (i.e. dinner)
                graph[day + start_day_nodes][(day * 2) + start_meal_nodes + 1] = 1

            # If the person can prepare breakfast and dinner for that day
            elif availability[day][person] == 3:
                # Add edge from person to the day
                graph[person + start_person_nodes][day + start_day_nodes] = 1
                # Add edge from the day to the corresponding meal of the day (i.e
                # Add edge to both breakfast and dinner for the day
                graph[day + start_day_nodes][(day * 2) + start_meal_nodes] = 1
                graph[day + start_day_nodes][(day * 2) + start_meal_nodes + 1] = 1

    # Edge from each meal to super sink node
    for i in range(n * 2):                      # O(n^2)
        graph[i + start_meal_nodes][num_nodes - 1] = 1

    return graph


def generate_network_adjusted(availability, breakfast_min_flow, dinner_min_flow):
    """
    Function that returns a network in adjacency matrix format.

    :Input:
        availability: list of lists that contains data about the time availability of each person.

    :Output:
        graph: an adjacency matrix that represents the network flow for calculating the meal allocation,
               this network flow has capacities equal to the remainder flow available after allocation
               using the min flow network

    :Time Complexity: O(n^2)
    :Aux Space Complexity: O(n^2), where n is the number of days

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
    graph = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

    # Fill graph
    # Edge from source node to meals node
    graph[source_node][meals_node] = demand

    # Edge from meals to each person (with capacity equal to the remainder of meals possible for the person)
    for i in range(num_persons):
        graph[meals_node][i + start_person_nodes] = max_meals - min_meals

    # Edge from meals to order option
    graph[meals_node][order_option_node] = max_order

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

        graph[order_option_node][meal + start_meal_nodes] = 1

        # Edge from the meal to sink node
        graph[meal + start_meal_nodes][num_nodes - 1] = 1

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
                graph[person + start_person_nodes][day + start_day_nodes] = 1
                # Add edge from the day to the corresponding meal of the day (i.e. breakfast)
                graph[day + start_day_nodes][(day * 2) + start_meal_nodes] = 1
                # Edge from the meal to super sink node
                graph[(day * 2) + start_meal_nodes][num_nodes - 1] = 1

            # If the person can prepare dinner for that day and dinner for that day is not allocated yet
            elif availability[day][person] == 2 and dinner_min_flow[day] == -1:
                # Add edge from person to the day
                graph[person + start_person_nodes][day + start_day_nodes] = 1
                # Add edge from the day to the corresponding meal of the day (i.e. dinner)
                graph[day + start_day_nodes][(day * 2) + start_meal_nodes + 1] = 1
                # Edge from the meal to super sink node
                graph[(day * 2) + start_meal_nodes + 1][num_nodes - 1] = 1

            # If the person can prepare breakfast and dinner for that day
            # and only dinner has been allocated (breakfast is not allocated yet)
            elif availability[day][person] == 3 and dinner_min_flow[day] > -1 and breakfast_min_flow[day] == -1\
                    and dinner_min_flow[day] != person:
                # Add edge from person to the day
                graph[person + start_person_nodes][day + start_day_nodes] = 1
                # Add edge from the day to the corresponding meal of the day (i.e. breakfast)
                graph[day + start_day_nodes][(day * 2) + start_meal_nodes] = 1
                # Edge from the meal to super sink node
                graph[(day * 2) + start_meal_nodes][num_nodes - 1] = 1

            # If the person can prepare breakfast and dinner for that day
            # and only breakfast has been allocated (dinner is not allocated yet)
            elif availability[day][person] == 3 and breakfast_min_flow[day] > -1 and dinner_min_flow[day] == -1\
                    and breakfast_min_flow[day] != person:
                # Add edge from person to the day
                graph[person + start_person_nodes][day + start_day_nodes] = 1
                # Add edge from the day to the corresponding meal of the day (i.e. dinner)
                graph[day + start_day_nodes][(day * 2) + start_meal_nodes + 1] = 1
                # Edge from the meal to super sink node
                graph[(day * 2) + start_meal_nodes + 1][num_nodes - 1] = 1

            # If the person can prepare breakfast and dinner for that day
            # and both breakfast and dinner is not allocated yet
            elif availability[day][person] == 3 and breakfast_min_flow[day] == -1 and dinner_min_flow[day] == -1:
                # Add edge from person to the day
                graph[person + start_person_nodes][day + start_day_nodes] = 1
                # Add edge from the day to the corresponding meal of the day (i.e. breakfast and dinner)
                # Add edge to both breakfast and dinner for the day
                graph[day + start_day_nodes][(day * 2) + start_meal_nodes] = 1
                graph[day + start_day_nodes][(day * 2) + start_meal_nodes + 1] = 1
                # Edge from the meals to super sink node
                graph[(day * 2) + start_meal_nodes][num_nodes - 1] = 1          # breakfast to sink
                graph[(day * 2) + start_meal_nodes + 1][num_nodes - 1] = 1      # dinner to sink

    return graph


def bfs(availability, network, parent):
    """
    Function to perform breadth-first-search on a network.

    :Input:
        availability: list of lists that contains data about the time availability of each person.
        network: adjacency matrix representation of network to apply BFS to
        parent: list that stores the immediate parent of a node in the BFS path
    :Output:
        Returns True if there exists a simple path from the source node to the sink node.
        Returns False otherwise.

    :Time Complexity: O(n)
    :Aux Space Complexity: O(n)
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

    :Time Complexity:
    :Aux Space Complexity:
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
    parent = [-1] * len(network)

    # Initialize maximum flow to 0
    max_flow = 0

    # Initialize breakfast and dinner arrays
    breakfast = [-1] * n
    dinner = [-1] * n

    # While there is an augmenting path in the network
    while bfs(availability, network, parent):
        # Starting from the sink node and going backwards until source
        curr = sink
        while curr != source:
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

        # Updating the residual network by augmenting with the found residual capacity
        curr = sink
        while curr != source:
            par = parent[curr]
            network[par][curr] -= 1      # Reducing the residual edge
            network[curr][par] += 1      # Incrementing the backward edge

            # Move backwards
            curr = parent[curr]

    print(breakfast, dinner)

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


# Testing task 1
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

# print(allocate(availability1))


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

    :Time Complexity: Needs to be O((N + M)^2)
    :Aux Space Complexity: Needs to be O(N + M)

    :Approach:
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

    def addSuffix(self, startNode, startIndex, length, isString1, isPartOfString1, isPartOfString2):
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

        Approach:
        TODO: Complete add_suffix approach
        """
        matching_letter_found = False
        if isString1:
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
            # While the character in the suffix to add matches the index of the connected node
            suffix_index = startIndex
            node_index = node.startIndex
            matched_length = 0
            while suffix_index < startIndex + length and node_index < node.startIndex + node.length and \
                    string[suffix_index] == string_node[node_index]:
                # Increment the index to check
                suffix_index += 1
                node_index += 1
                # Increment matched_length
                matched_length += 1
                # Mark matched_letter_found as True
                matching_letter_found = True

            if matching_letter_found:
                # If exists, make the remainder of the length into the matched node's child
                if node.length - matched_length > 0:
                    self.addSuffix(node, node.startIndex + matched_length, node.length - matched_length, node.isPartOfString1,
                                   node.isPartOfString1, node.isPartOfString2)
                    # And un-mark the node as an end
                    node.isEnd = False
                # Adjust the matched node's length
                node.length = matched_length
                # Update isPartOfString1 and isPartOfString2 of the matched node
                if isPartOfString1:
                    node.isPartOfString1 = True
                if isPartOfString2:
                    node.isPartOfString2 = True
                # If the matched node is part of both string1 and string2
                # Update the common length
                if node.isPartOfString1 and node.isPartOfString2:
                    node.commonLength = startNode.commonLength + matched_length
                    # If the node's commonLength is greater than the current maxLength of the tree
                    if node.commonLength > self.maxLength:
                        self.maxLength = node.commonLength
                # If the suffix to add fully matches the matched substring, mark the matched node as an end
                if length - matched_length == 0:
                    node.isEnd = True
                # Else, make the remainder of the suffix a child of the matched node
                else:
                    self.addSuffix(node, startIndex + matched_length, length - matched_length, isString1,
                                   isPartOfString1, isPartOfString2)

                break

        # Else if we have iterated through all the startNode's children and did not find a matching letter
        if not matching_letter_found:
            # Append a new child to the startNode
            startNode.children.append(SuffixTreeNode(startIndex, length, True, isPartOfString1, isPartOfString2))

    def build(self):
            """
            Method to build (i.e. fill up) the suffix tree

            :Time Complexity: TODO: Complete time complexity build method
            :Aux Space Complexity: TODO: Complete space complexity build method
            """
            # Inserting suffixes of submission1 into the suffix tree
            for startIndex in range(len(self.string1)):
                self.addSuffix(self, startIndex, len(self.string1) - startIndex, True, True, False)

            # Inserting suffixes of submission2 into the suffix tree
            for startIndex in range(len(self.string2)):
                self.addSuffix(self, startIndex, len(self.string2) - startIndex, False, False, True)


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
            isPartOfString1:
            isPartOfString2:
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


def build_longest_common_substring(max_node, submission1):
    longest_common_substring = []
    curr_node = max_node
    while curr_node.commonLength != 0:
        longest_common_substring.append(submission1[curr_node.startIndex:curr_node.startIndex + curr_node.length])
        curr_node = curr_node.parent
    longest_common_substring = "".join(longest_common_substring[::-1])
    return longest_common_substring


# TESTING TASK 2
# string1 = "referrer"
# string2 = "referee"

# string1 = "the lazy brown dog jumped over the lazy dog"
# string2 = "my lazy dog has eaten my dog"

string1 = "radix sort n counting sort ar sorting algos"
string2 = "counting sort n radix sort ar sorting algos"

# print(compare_subs(string1, string2))

suffix_tree = SuffixTree(string1, string2)
for startIndex in range(len(string1)):
    suffix_tree.addSuffix(suffix_tree, startIndex, len(string1) - startIndex, True, True, False)

# for startIndex in range(len(string2)):
#     suffix_tree.addSuffix(suffix_tree, startIndex, len(string2) - startIndex, False, False, True)

for i in range(len(suffix_tree.children)):
    print(suffix_tree.children[i])

print()

for i in range(len(suffix_tree.children[6].children)):
    print(suffix_tree.children[6].children[i])

# print(suffix_tree.maxLength)
