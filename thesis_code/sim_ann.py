import numpy as np
import random
import math

def calculate_route_makespan(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i], route[i + 1]]
    return total_distance

# current_route -> itemX....itemZ
def simulated_annealing_order_items(current_route, distance_matrix, initial_temperature, cooling_rate, L, K2, epsilon2):
    n = len(current_route)
    if n < 3:
      return current_route
    # current_route = list(range(1, n))
    current_distance = calculate_route_makespan(current_route, distance_matrix)
    best_route = current_route.copy()
    best_distance = current_distance
    temperature = initial_temperature
    k = 1  # Plateau counter
    terminate = False
    no_improvement_count = 0
    accepted_count = 0

    while not terminate:
        new_route = current_route.copy()
        i, j = random.sample(range(n), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        new_distance = calculate_route_makespan(new_route, distance_matrix)

        if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temperature):
            current_route = new_route
            current_distance = new_distance
            accepted_count += 1

        if new_distance < best_distance:
            best_route = new_route
            best_distance = new_distance
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Evaluate stopping conditions
        if k < L:
            k += 1
            terminate = False
        elif no_improvement_count >= K2 and (accepted_count / L) < epsilon2:
            terminate = True
        else:
            temperature *= cooling_rate  # Decrease temperature
            k += 1
            terminate = False

    return best_route
