import numpy as np
import matplotlib.pyplot as plt
import random
class Vehicle:
    def __init__(self, id, x=0.0, y=0.0, vx=10.0, vy=0.0):
        self.id = id  # Vehicle ID
        self.x = x  # X position
        self.y = y  # Y position
        self.vx = vx  # Longitudinal velocity
        self.vy = vy  # Lateral velocity
        self.velocity_history = []  # Store velocity history for plotting

    def update_dynamics(self, neighbors_info):
        """
        Update the vehicle dynamics using the control input based on neighbors' information.
        """
        # Compute control input based on neighbors' information
        control_input = self.compute_control_input(neighbors_info)

        # Apply control input to adjust the vehicle's velocity
        self.vx += control_input['vx_adjust']  # Adjust longitudinal velocity
        self.velocity_history.append(self.vx)  # Store the current velocity

        # Update position based on current velocity
        self.x += self.vx * 0.01  # Assuming a time step of 0.01 seconds
        self.y += self.vy * 0.01  # Assuming a time step of 0.01 seconds

    def compute_control_input(self, neighbors_info):
        """
        Compute the control input (velocity adjustment) based on neighbors' information.
        Now using the Average Consensus Algorithm.
        """
        # Initialize the control input
        control_input = {'vx_adjust': 0}

        # Average Consensus control parameters
        consensus_gain = 0.1  # Adjust this gain as needed based on system dynamics
        num_neighbors = len(neighbors_info)

        if num_neighbors > 0:
            avg_vx = sum(neighbor['vx'] for neighbor in neighbors_info) / num_neighbors

            # Update vehicle's velocity based on average consensus algorithm
            control_input['vx_adjust'] = consensus_gain * (avg_vx - self.vx)  # Velocity alignment

        return control_input

def get_neighbors_info(vehicles, vehicle_id):
    """
    Retrieve neighbors' information from the list of vehicles.
    """
    # Extract all vehicles' states except for the current vehicle's state
    neighbors_info = [vars(vehicle) for vehicle in vehicles if vehicle.id != vehicle_id]
    return neighbors_info

def calculate_minimum_distance(vehicles):
    """
    Calculate the minimum distance between all pairs of vehicles.
    """
    min_distance = float('inf')
    num_vehicles = len(vehicles)

    for i in range(num_vehicles):
        for j in range(i + 1, num_vehicles):
            distance = np.sqrt((vehicles[i].x - vehicles[j].x) ** 2 + (vehicles[i].y - vehicles[j].y) ** 2)
            if distance < min_distance:
                min_distance = distance

    return min_distance

# Simulation setup
if __name__ == "__main__":
    # Initialize a few vehicles with different IDs
    vehicles = [
        Vehicle(id=i+1, x=i*10, vx=2*(i+1)) for i in range(10)
    ]

    # List to store minimum distances over time
    min_distances = []
    attacked_Agents = [2,6]
    attack_steps = 50
    # Run the simulation for a few steps
    for step in range(100):
        print(f"Step {step}:\n{'-'*20}")

        for vehicle in vehicles:
            # Retrieve neighbors' information
            neighbors_info = get_neighbors_info(vehicles, vehicle.id)
            

            # Introduce the attack
            if attack_steps <step< attack_steps+10 and vehicle.id in attacked_Agents:
                # Modify the velocity drastically to simulate an attack
                vehicle.vx += random.uniform(10, 80)  # Add an unrealistic jump in velocity
                print(f"Cyber attack introduced to vehicle {vehicle.id} at step {step}")


            vehicle.update_dynamics(neighbors_info)
            print(f'Vehicle_{vehicle.id} at ({vehicle.x:.2f}, {vehicle.y:.2f}), velocity: {vehicle.vx:.2f}')

        # Calculate and store the minimum distance between vehicles
        min_distance = calculate_minimum_distance(vehicles)
        min_distances.append(min_distance)

    # Plot the velocity consensus over time
    plt.figure(figsize=(10, 6))
    for vehicle in vehicles:
        plt.plot(vehicle.velocity_history, label=f'Vehicle_{vehicle.id}')

    plt.title("Velocity Consensus Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity (vx)")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot the minimum distance between vehicles over time
    plt.figure(figsize=(10, 6))
    plt.plot(min_distances, label='Minimum Distance', color='red')
    plt.title("Minimum Distance Between Vehicles Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Minimum Distance")
    plt.legend()
    plt.grid()
    plt.show()
