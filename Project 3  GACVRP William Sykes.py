import numpy as np
import random
import time
'''
William Sykes
2/28/2024
Implements a genetic algorithm (GA) for solving the Capacitated Vehicle Routing Problem (CVRP), 
incorporating several initial population generation strategies and fitness calculation adjustments. 

'''
def read_cvrp(file_path):
    """
    Reads a CVRP data file and extracts node coordinates, demands, and vehicle capacity.

    The function expects a file in a specific format (these files have been provided in the zip) 
    with sections for node coordinates, demands, and vehicle capacity. 
    It parses the file to extract this information, which
    is used in solving the Capacitated Vehicle Routing Problem.

    Parameters:
    - file_path (str): The path to the file containing CVRP data.

    Returns:
    - node_coords (dict): A dictionary where keys are node IDs and values are tuples
      representing the (x, y) coordinates of that node.
    - demands (dict): A dictionary where keys are node IDs and values are the demand
      at that node.
    - capacity (int): The capacity of the vehicle used in the CVRP.
    
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    reading_section = None
    node_coords = {}
    demands = {}
    capacity = 0
    for line in lines:
        line = line.strip()
        if line.startswith("NODE_COORD_SECTION"):
            reading_section = "NODE_COORDS"
            continue
        elif line.startswith("DEMAND_SECTION"):
            reading_section = "DEMANDS"
            continue
        elif line.startswith("DEPOT_SECTION"):
            reading_section = "DEPOT"
            continue
        elif "CAPACITY" in line:
            capacity = int(line.split()[2])
            continue
        elif line.startswith("EOF"):
            break

        if reading_section == "NODE_COORDS":
            parts = line.split()
            node_coords[int(parts[0])] = (int(parts[1]), int(parts[2]))
        elif reading_section == "DEMANDS":
            parts = line.split()
            demands[int(parts[0])] = int(parts[1])
        elif reading_section == "DEPOT":
            
            continue

    return node_coords, demands, capacity


def calculate_euclidean_distance(coord1, coord2):
    """
    Calculates the Euclidean distance between two points.

    Parameters:
    - coord1 (tuple): The (x, y) coordinates of the first point.
    - coord2 (tuple): The (x, y) coordinates of the second point.

    Returns:
    - distance (float): The Euclidean distance between the two points.
    """
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def calculate_route_distance(route, node_coords):
    """
    Calculates the total distance of a given route based on node coordinates.

    Parameters:
    - route (list): A list of node IDs representing the sequence of nodes in the route.
    - node_coords (dict): A dictionary of node coordinates where keys are node IDs and
      values are tuples representing the (x, y) coordinates of that node.

    Returns:
    - distance (float): The total distance of the route.
    """
    distance = 0
    for i in range(len(route) - 1):
        distance += calculate_euclidean_distance(node_coords[route[i]], node_coords[route[i + 1]])
    return distance

def fitness_function(solution, node_coords, demands, vehicle_capacity, min_trucks):
    """
    Calculates the fitness of a CVRP solution, considering the total distance of all routes,
    adherence to vehicle capacity constraints, and the minimum number of trucks required.

    Parameters:
    - solution (list of lists): A list where each element is a route (list of node IDs),
      representing a complete solution to the CVRP.
    - node_coords (dict): A dictionary of node coordinates.
    - demands (dict): A dictionary of demands for each node.
    - vehicle_capacity (int): The maximum capacity of the vehicle.
    - min_trucks (int): The minimum number of trucks required for the solution.

    Returns:
    - fitness (float): The calculated fitness of the solution, where lower values are better.
    """
    total_distance = 0
    capacity_violation_penalty = 0

    for route in solution:
        route_demand = sum(demands[node] for node in route if node != 1)  # Assuming 1 is the depot and has no demand
        total_distance += calculate_route_distance(route, node_coords)
        
        # Penalize capacity violations within each route
        if route_demand > vehicle_capacity:
            capacity_violation_penalty += (route_demand - vehicle_capacity) * 100  # can adjust

    # Penalize solutions that don't meet the minimum number of trucks requirement
    trucks_penalty = 0
    if len(solution) < min_trucks:
        trucks_penalty = (min_trucks - len(solution)) * 10000  # can adjust

    fitness = total_distance + capacity_violation_penalty + trucks_penalty
    
    return fitness

# random shuffle
def generate_solution(nodes, demands, vehicle_capacity):
    """
    Generates a single feasible solution for the CVRP.

    The solution consists of multiple routes that collectively serve all customers
    without exceeding the vehicle's capacity.

    Parameters:
    - nodes (list): A list of all node IDs including the depot.
    - demands (dict): The demand at each node.
    - vehicle_capacity (int): The capacity of the vehicle.

    Returns:
    - routes (list of lists): A list of routes that make up the solution. Each route
      is a list of node IDs, starting and ending at the depot.
      
    """
    routes = []
    remaining_nodes = [node for node in nodes if node != 1]  # Exclude the depot
    random.shuffle(remaining_nodes)  
    
    while remaining_nodes:
        route_capacity = vehicle_capacity
        route = [1]  # Start at the depot
        for node in remaining_nodes[:]:
            if demands[node] <= route_capacity:
                route.append(node)
                route_capacity -= demands[node]
                remaining_nodes.remove(node)
        route.append(1)  # Return to the depot
        routes.append(route)
    return routes

def generate_initial_population(population_size, nodes, demands, capacity):
    """
    Generates an initial population for the genetic algorithm.

    Each individual in the population represents a potential solution to the CVRP,
    consisting of multiple routes that satisfy the capacity constraints.

    Parameters:
    - population_size (int): The size of the population to generate.
    - nodes (list): A list of node IDs including the depot.
    - demands (dict): The demand at each node, as described in read_cvrp_file_fixed.
    - capacity (int): The capacity of the vehicle, as described in read_cvrp_file_fixed.

    Returns:
    - population (list of lists): The generated initial population.
    """
    population = []
    for _ in range(population_size):
        routes = generate_solution(nodes, demands, capacity)
        population.append(routes)
    return population


# Deterministic, will likely generate the same route over and over
def nearest_neighbor_route(start_node, nodes, node_coords, demands, vehicle_capacity):
    """
    Generates a route using the Nearest Neighbor heuristic.

    Parameters:
    - start_node (int): The starting node ID for the route (usually the depot).
    - nodes (list): A list of all node IDs.
    - node_coords (dict): Coordinates for each node.
    - demands (dict): Demand for each node.
    - vehicle_capacity (int): Capacity of the vehicle.

    Returns:
    - route (list): The generated route as a list of node IDs.
    """
    route = [start_node]
    current_capacity = vehicle_capacity
    unvisited_nodes = set(nodes) - {start_node}
    
    current_node = start_node
    while unvisited_nodes and current_capacity > 0:
        nearest_node = None
        nearest_distance = float('inf')
        
        for node in unvisited_nodes:
            if demands[node] <= current_capacity:
                distance = calculate_euclidean_distance(node_coords[current_node], node_coords[node])
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_node = node
        
        if nearest_node is not None:
            route.append(nearest_node)
            current_node = nearest_node
            current_capacity -= demands[nearest_node]
            unvisited_nodes.remove(nearest_node)
        else:
            # No more nodes can be added to this route due to capacity constraints
            break
            
    route.append(start_node)  # Return to depot
    return route
# yields to constraints, short calculation time
def nearest_random_route(start_node, nodes, node_coords, demands, vehicle_capacity):
    """
    Generates a route using a modified Nearest Neighbor heuristic to include stochasticity.

    Parameters remain the same.
    """
    route = [start_node]
    current_capacity = vehicle_capacity - demands[start_node]
    unvisited_nodes = set(nodes) - {start_node}

    current_node = start_node
    while unvisited_nodes and current_capacity > 0:
        nearest_nodes = []
        for node in unvisited_nodes:
            if demands[node] <= current_capacity:
                distance = calculate_euclidean_distance(node_coords[current_node], node_coords[node])
                nearest_nodes.append((node, distance))

        # Sort based on distance, then select randomly among the top N
        nearest_nodes.sort(key=lambda x: x[1])
        top_n = nearest_nodes[:3]  # can adjust N 

        if top_n:
            nearest_node, _ = random.choice(top_n)
            route.append(nearest_node)
            current_node = nearest_node
            current_capacity -= demands[nearest_node]
            unvisited_nodes.remove(nearest_node)
        else:
            break

    route.append(1)  # Assuming 1 is always the depot
    return route

# will completely ignore constraints and give unfeasible solutions
def generate_nearest_random_solution(nodes, node_coords, demands, vehicle_capacity):
    """
    Generates an initial CVRP solution using a stochastic approach to the Nearest Neighbor heuristic
    By randomly selecting a start node that has a demand <= vehicle_capacity to ensure feasibility.

    Parameters remain the same.
    """
    solution = []
    unvisited_nodes = set(nodes) - {1}  # Excluding the depot

    while unvisited_nodes:
        # Randomly select a start node that has a demand <= vehicle_capacity to ensure feasibility
        feasible_starts = [node for node in unvisited_nodes if demands[node] <= vehicle_capacity]
        if not feasible_starts:
            break  # If no feasible starts, exit the loop
        start_node = random.choice(feasible_starts)
        route = nearest_random_route(start_node, list(unvisited_nodes) + [1], node_coords, demands, vehicle_capacity)
        solution.append(route)
        
        # Remove visited nodes from the unvisited set
        for node in route:
            unvisited_nodes.discard(node)

    return solution

# when paired with nearest_random_route yields good solutions in a shorter period of time on average then random shuffle
def generate_nearest_neighbor_solution(nodes, node_coords, demands, vehicle_capacity):
    """
    Generates an initial CVRP solution using the Nearest Neighbor heuristic or Random Nearest Neighbor heuristic
    depending on which is specified in while loop, does not include stochastic start node - starts from depot.

    Parameters:
    - nodes (list): A list of all node IDs including the depot.
    - node_coords (dict): Coordinates for each node.
    - demands (dict): Demand for each node.
    - vehicle_capacity (int): Capacity of the vehicle.

    Returns:
    - solution (list of lists): A list of routes that make up the solution.
    """
    solution = []
    unvisited_nodes = set(nodes) - {1}  # Assuming node 1 is the depot

    while unvisited_nodes:
        start_node = 1  # Start from the depot
        route = nearest_random_route(start_node, list(unvisited_nodes) + [start_node], node_coords, demands, vehicle_capacity)
        solution.append(route)
        
        # Remove the visited nodes from the unvisited set
        for node in route:
            unvisited_nodes.discard(node)
    
    return solution

def generate_initial_population_nearest_neighbor(population_size, nodes, node_coords, demands, capacity):
    population = []
    for _ in range(population_size):
        solution = generate_nearest_neighbor_solution(nodes, node_coords, demands, capacity)
        population.append(solution)
    return population

def calculate_angle(depot_coords, customer_coords):
    """
    Calculates the angle between the depot and a customer node. This angle is calculated
    in radians relative to the positive x-axis, facilitating the sorting of nodes based on
    their angular position around the depot.

    Parameters:
    - depot_coords (tuple): The (x, y) coordinates of the depot.
    - customer_coords (tuple): The (x, y) coordinates of the customer node.

    Returns:
    - (float): The angle in radians between the depot and the customer node.
    """
    
    return np.arctan2(customer_coords[1] - depot_coords[1], customer_coords[0] - depot_coords[0])

def sweep_algorithm(nodes, node_coords, demands, vehicle_capacity):
    """
    Implements the sweep algorithm for generating routes for the CVRP. The sweep algorithm
    sorts customers based on their angular position relative to the depot and then groups
    them into routes, ensuring that each route does not exceed the vehicle capacity.

    Parameters:
    - nodes (list): A list of node IDs including the depot.
    - node_coords (dict): A dictionary where keys are node IDs and values are their (x, y) coordinates.
    - demands (dict): A dictionary where keys are node IDs and values are the demands at those nodes.
    - vehicle_capacity (int): The maximum capacity of the vehicle.

    Returns:
    - routes (list of lists): A list where each sublist represents a route starting and ending at the depot.
    """
    depot_coords = node_coords[1]  # Assuming node 1 is the depot
    angles = {node: calculate_angle(depot_coords, node_coords[node]) for node in nodes if node != 1}
    
    # Sort nodes by angle
    sorted_nodes = sorted(nodes, key=lambda node: angles.get(node, 0))
    
    # Group nodes into routes based on capacity
    routes = []
    current_route = [1]  # Start route with depot
    current_capacity = 0
    
    for node in sorted_nodes:
        if current_capacity + demands[node] <= vehicle_capacity:
            current_route.append(node)
            current_capacity += demands[node]
        else:
            current_route.append(1)  # End current route with depot
            routes.append(current_route)
            current_route = [1, node]  # Start new route
            current_capacity = demands[node]
    current_route.append(1)  # End last route with depot
    routes.append(current_route)
    
    return routes

def generate_initial_population_sweep(population_size, nodes, node_coords, demands, capacity, min_trucks):
    """
    Generates an initial population of solutions for the genetic algorithm using the sweep algorithm.
    This method ensures diversity in the initial population by shuffling the nodes before applying
    the sweep algorithm. It also ensures that the generated solutions meet the minimum number of trucks required.

    Parameters:
    - population_size (int): The size of the initial population to generate.
    - nodes (list): A list of node IDs including the depot.
    - node_coords (dict): Coordinates for each node.
    - demands (dict): Demand for each node.
    - capacity (int): Capacity of the vehicle.
    - min_trucks (int): The minimum number of trucks required for the solution.

    Returns:
    - population (list of lists): The generated initial population, where each element is a solution represented by routes.
    """
    population = []
    for _ in range(population_size):
        
        shuffled_nodes = nodes.copy()
        np.random.shuffle(shuffled_nodes)  # Shuffle nodes for diversity
        routes = sweep_algorithm(shuffled_nodes, node_coords, demands, capacity)
        
        # Ensure we meet the minimum number of trucks requirement
        while len(routes) < min_trucks:
            routes.append([1, 1])  # Adding empty routes to meet min_trucks
        
        population.append(routes)
    return population


def roulette_wheel_selection(population, node_coords, num_parents, min_trucks):
    """
    Selects parents using the Roulette Wheel Selection method.

    Parameters:
    - population (list of lists): The current population of solutions.
    - node_coords (dict): Node coordinates needed to calculate fitness of each solution.
    - num_parents (int): The number of parents to select.
    - min_trucks (int): minimum number of trucks for fitness calculation

    Returns:
    - parents (list of lists): A list of selected parent solutions.
    """
    # Calculate fitness for each individual in the population
    fitness_values = [fitness_function(individual, node_coords, demands, capacity, min_trucks) for individual in population]
    # Convert fitness scores to selection probabilities
    # Inverting fitness values because lower fitness (distance) is better
    max_fitness = max(fitness_values)
    selection_probs = [(max_fitness - fitness) + 1 for fitness in fitness_values]
    total_fitness = sum(selection_probs)
    selection_probs = [prob / total_fitness for prob in selection_probs]
    
    # Select parents based on probabilities
    parents = random.choices(population, weights=selection_probs, k=num_parents)
    return parents

def crossover(parent1, parent2, demands, capacity):
    
    """
    Performs a crossover operation between two parent solutions to produce offspring. 
    The crossover selects one route from each parent randomly to form the basis of each offspring. 
    It then fills the rest of the offspring routes by merging routes from both parents, avoiding duplication.
    After merging, it attempts to repair any potential capacity issues in the offspring.

    Parameters:
    - parent1, parent2 (list of lists): The parent solutions from which offspring are derived. Each parent is a list of routes.
    - demands (dict): A dictionary where keys are node IDs and values are the demands at those nodes.
    - capacity (int): The maximum capacity that any route can handle.

    Returns:
    - offspring1, offspring2 (list of lists): Two new offspring solutions, each a list of routes.
    """
    
    # Select one route from each parent randomly
    route1 = random.choice(parent1)
    route2 = random.choice(parent2)
    
    # Combine these routes to form the base of each offspring
    offspring1 = [list(route1)]
    offspring2 = [list(route2)]
    
    # Fill the rest of the offspring routes by merging routes from both parents, avoiding duplication
    for route in parent1 + parent2:
        if not route in offspring1:
            offspring1.append(list(route))
        if not route in offspring2:
            offspring2.append(list(route))
    
    # Attempt to repair any potential capacity issues in offspring
    repair_offspring(offspring1, demands, capacity)
    repair_offspring(offspring2, demands, capacity)
    
    return offspring1, offspring2

def repair_offspring(offspring, demands, capacity):
    """
    Iterates through each route in an offspring solution to check for and repair any capacity violations.
    If a route exceeds the vehicle's capacity, it is split into two or more routes that comply with the capacity constraint.
    This process modifies the offspring in place.

    Parameters:
    - offspring (list of lists): The offspring solution to be repaired. It is a list of routes.
    - demands (dict): A dictionary where keys are node IDs and values are the demands at those nodes.
    - capacity (int): The maximum capacity that any route can handle.

    Note:
    This method may call `split_route` internally to perform the actual splitting of an overloaded route.
    """
    # Iterate through each route in the offspring
    for i, route in enumerate(offspring):
        total_demand = sum(demands[node] for node in route)
        if total_demand > capacity:
            # The route exceeds capacity, needs splitting
            split_route(offspring, i, demands, capacity)

def split_route(offspring, route_index, demands, capacity):
    """
    Splits an overloaded route into two separate routes to ensure adherence to vehicle capacity constraints.
    This method modifies the offspring in-place by adding a new route that takes some of the nodes from the
    overloaded route, ensuring both resulting routes do not exceed the vehicle's capacity.

    Parameters:
    - offspring (list of lists): The offspring solution containing routes to potentially split.
    - route_index (int): The index of the route within the offspring that needs to be split.
    - demands (dict): Demand at each node, used to calculate capacity usage.
    - capacity (int): The maximum capacity of the vehicle, which routes must not exceed.

    """
    route_to_split = offspring[route_index]
    
    # Start with an empty route for the split
    new_route = [1]  # Initialize with depot as the start point
    
    # Iterate over the route to find the best split point based on capacity
    current_demand = 0
    for node in route_to_split[1:-1]:  # Exclude the depot from the split calculation
        if current_demand + demands[node] > capacity:
            # Found the split point; break the loop
            break
        current_demand += demands[node]
        new_route.append(node)
    
    # Finish the new route by returning to the depot
    new_route.append(1)
    
    # Remove the nodes moved to the new route from the original route
    for node in new_route[1:-1]:  # Exclude the depot
        route_to_split.remove(node)
    
    # Check if new route itself violates the capacity constraint and further splitting is required
    if sum(demands[node] for node in new_route) > capacity:
        print("Further splitting required")
        # could be implemented here for further splitting if required
    
    # Add the new route to the offspring
    offspring.append(new_route)
    
def swap_mutation(offspring, demands, capacity):
    """
    Applies a swap mutation on a given offspring solution for CVRP. Two nodes are selected
    randomly and swapped, either within the same route or between two routes, ensuring that
    the vehicle capacity constraints are not violated.

    Parameters:
    - offspring (list of lists): The offspring solution to mutate. Each inner list represents a route.
    - demands (dict): The demand at each node.
    - capacity (int): The capacity of the vehicle.

    Returns:
    - offspring (list of lists): The mutated offspring solution.
    """
    # Choose two routes randomly; they can be the same route
    route_indices = np.random.choice(len(offspring), 2, replace=True)
    route1, route2 = offspring[route_indices[0]], offspring[route_indices[1]]

    # Choose one node from each selected route, excluding the depot (start and end of the route)
    node_index1 = np.random.randint(1, len(route1) - 1)
    node_index2 = np.random.randint(1, len(route2) - 1)
    node1, node2 = route1[node_index1], route2[node_index2]

    # Check capacity constraint before performing the swap
    if is_swap_feasible(route1, route2, node_index1, node_index2, demands, capacity):
        # Perform the swap
        route1[node_index1], route2[node_index2] = node2, node1

    return offspring

def is_swap_feasible(route1, route2, node_index1, node_index2, demands, capacity):
    """
    Checks if swapping two nodes between routes is feasible without violating vehicle capacity constraints.

    Parameters:
    - route1, route2 (list): The routes involved in the swap.
    - node_index1, node_index2 (int): The indices of the nodes to be swapped.
    - demands (dict): The demand at each node.
    - capacity (int): The vehicle capacity.

    Returns:
    - (bool): True if the swap does not violate capacity constraints, False otherwise.
    """
    # Calculate the new demand for each route after the swap
    new_demand_route1 = sum(demands[route1[i]] for i in range(len(route1))) - demands[route1[node_index1]] + demands[route2[node_index2]]
    new_demand_route2 = sum(demands[route2[i]] for i in range(len(route2))) - demands[route2[node_index2]] + demands[route1[node_index1]]

    # Check if the new demands exceed the capacity for either route
    if new_demand_route1 <= capacity and new_demand_route2 <= capacity:
        return True
    return False



    
def create_next_generation(current_population, node_coords, demands, capacity, min_trucks, mutation_rate=0.8):
    """
    Creates the next generation for the GA.

    Parameters:
    - current_population (list of lists): The current generation of solutions.
    - node_coords (dict): Node coordinates.
    - demands (dict): Demand at each node.
    - capacity (int): Vehicle capacity.
    - mutation_rate (float): Probability of mutating a given offspring.

    Returns:
    - next_generation (list of lists): The next generation of solutions.
    """
    next_generation = []
    num_parents = len(current_population) // 2
    parents = roulette_wheel_selection(current_population, node_coords, num_parents, min_trucks)
    
    # Generate offspring via crossover
    for i in range(0, len(parents)-1, 2):
        offspring1, offspring2 = crossover(parents[i], parents[i+1], demands, capacity)
        # Apply mutation with a given probability
        if random.random() < mutation_rate:
            offspring1 = swap_mutation(offspring1, demands, capacity)
        if random.random() < mutation_rate:
            offspring2 = swap_mutation(offspring2, demands, capacity)
        next_generation.extend([offspring1, offspring2])
    
    # Fill the rest of the next generation by selecting the best individuals from the current generation
    while len(next_generation) < len(current_population):
        sorted_current_population = sorted(current_population, key=lambda x: fitness_function(x, node_coords, demands, capacity, min_trucks))
        next_generation.append(sorted_current_population.pop(0))

    return next_generation

def run_genetic_algorithm(node_coords, demands, capacity, min_trucks, population_size, num_generations, mutation_rate):
    
    """
    Runs the genetic algorithm for the CVRP.

    Parameters:
    - node_coords (dict): Node coordinates.
    - demands (dict): Demand at each node.
    - capacity (int): Vehicle capacity.
    - population_size (int): Size of the population.
    - num_generations (int): Number of generations to evolve.
    - mutation_rate (float): Mutation rate.

    Returns:
    - best_solution (list of lists): The best solution found by the GA.
    """
    # Generate initial population
    nodes = list(node_coords.keys())
    #current_population = generate_initial_population(population_size, nodes, demands, capacity)
    #current_population = generate_initial_population_sweep(population_size, nodes, node_coords, demands, capacity, min_trucks) 
    current_population = generate_initial_population_nearest_neighbor(population_size, nodes, node_coords, demands, capacity)
    
    # Evolve generations
    for generation in range(num_generations):
        current_population = create_next_generation(current_population, node_coords, demands, capacity, mutation_rate)
        
        
    # Identify the best solution at the end
    best_solution = min(current_population, key=lambda x: fitness_function(x, node_coords, demands, capacity, min_trucks))
    
    return best_solution


'''
file_path = 'CVRP.txt'  # Corrected path for this environment
node_coords, demands, capacity = read_cvrp_file_fixed(file_path)

# Debugging print
print("Node Coordinates:", node_coords)
print("Demands:", demands)
print("Vehicle Capacity:", capacity)

# Generate the initial population
population_size = 50
initial_population = generate_initial_population(population_size, list(node_coords.keys()), demands, capacity)

# Calculate the fitness of the first solution in the initial population 
fitness_of_first_solution = fitness_function(initial_population[0], node_coords)
print("Initial Population's first solution:", initial_population[0])
print("Fitness of first solution:", fitness_of_first_solution)

'''
if __name__ == "__main__":
    
    # Parameters and statistics
    file_path = 'CVRP51.txt'  
    node_coords, demands, capacity = read_cvrp(file_path)
    min_trucks = 5;
    num_runs = 1
    total_runtime = 0
    best_solutions_fitness = []

    worst_solution = float('-inf')  # Since lower fitness is better, we start with -inf
    best_solution_fitness = float('inf')  # Since lower fitness is better, we start with inf

for _ in range(num_runs):
    start_time = time.time()

    # driver                                         adjust parameters here, adjust mutation_rate in creat_next_generation aswell
    solution = run_genetic_algorithm(node_coords, demands, capacity, min_trucks, population_size=50, num_generations=100, mutation_rate=0.8)

    end_time = time.time()
    runtime = end_time - start_time
    total_runtime += runtime

    # Fitness of the best solution
    fitness = fitness_function(solution, node_coords, demands, capacity, min_trucks)
    best_solutions_fitness.append(fitness)

    # Update worst and best solution fitness
    worst_solution = max(worst_solution, fitness)
    best_solution_fitness = min(best_solution_fitness, fitness)
    
average_runtime = total_runtime / num_runs
average_best_solution_fitness = np.mean(best_solutions_fitness)

print(f"Average runtime: {average_runtime} seconds.")
print(f"Average best solution's fitness: {average_best_solution_fitness}")
print(f"Worst solution's fitness: {worst_solution}")
print(f"Best solution's fitness: {best_solution_fitness}")
    