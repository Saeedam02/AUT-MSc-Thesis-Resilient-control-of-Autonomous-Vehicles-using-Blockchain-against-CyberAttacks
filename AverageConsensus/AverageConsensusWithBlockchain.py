import hashlib
import json
from time import time
import numpy as np
import matplotlib.pyplot as plt
import requests
import random
class Blockchain:
    def __init__(self,num_agents):
        self.current_transactions = []
        self.chain = []
        self.nodes = set()
        self.sings = [self.get_sign(i) for i in range(num_agents+1)]
        self.node_states = {}  # Store the state history of each node

        # Create the genesis block
        self.new_block(previous_hash='1', proof=100)

    def register_node(self, vehicle_id,state):
        """
        Add a new node (vehicle) to the list of nodes
        :param vehicle_id: ID of the vehicle
        :param state: Current state of the vehicle
        :param sign: Signature of the vehicle
        """
         # Generate a valid signature for the vehicle ID
        sign = self.get_sign(vehicle_id)
        if vehicle_id in self.nodes:
            print("The same ID is registered in the blockchain, try new one")
        else:
            if sign in self.sings:
                self.nodes.add(vehicle_id)
                self.node_states[vehicle_id] = [state]  # Initialize state history for the node
                #print(self.node_states)
            else:
                print(f"Registration failed for vehicle {vehicle_id}: Invalid signature.")
                return  # Exit the function if the signature is invalid
    def register_additional_nodes(self, vehicle_id, state):
        """
        Manually register a new node (vehicle) to the blockchain.
        :param vehicle_id: ID of the vehicle
        :param state: Current state of the vehicle
        """
        # Generate a valid signature for the vehicle ID
        sign = self.get_sign(vehicle_id)
        self.sings.append(sign)
        if vehicle_id in self.nodes:
            print("The same ID is already registered in the blockchain, try a new one.")
            return
        else:
            # Allow registration without checking against self.sings
            self.nodes.add(vehicle_id)
            self.node_states[vehicle_id] = [state]  # Initialize state history for the node
            print(f"Vehicle {vehicle_id} registered successfully.")

    def get_sign(self, vehicle_id):
        """
        Calculate a sign for each vehicle to use in Blockchain while registering as a Node.
        """
        return hashlib.sha256(str(vehicle_id).encode()).hexdigest()[:8]
    
    def valid_chain(self, chain):
        """
        Determine if a given blockchain is valid
        :param chain: A blockchain
        :return: True if valid, False if not
        """       
        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]
            last_block_hash = self.hash(last_block)
            if block['previous_hash'] != last_block_hash:
                return False
            if not self.valid_proof(last_block['proof'], block['proof'], last_block_hash):
                return False

            last_block = block
            current_index += 1

        return True


    def new_block(self, proof, previous_hash=None):
        """
        Create a new Block in the Blockchain
        :param proof: The proof given by the Proof of Work algorithm
        :param previous_hash: Hash of previous Block
        :return: New Block
        """
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }

        # Reset the current list of transactions
        self.current_transactions = []
        self.chain.append(block)
        return block

    def new_transaction(self, vehicle_id, status):
        """
        Creates a new transaction to go into the next mined Block
        :param vehicle_id: ID of the vehicle
        :param status: Current state of the vehicle
        :return: The index of the Block that will hold this transaction
        """
        self.current_transactions.append({
            'vehicle_id': vehicle_id,
            'status': status,
        })
        # Update the state history for the vehicle
        if vehicle_id in self.node_states:
            self.node_states[vehicle_id].append(status)
        else:
            self.node_states[vehicle_id] = [status]

        return self.last_block['index'] + 1

    def check_for_attacks(self,current_step):
        """
        Apply WMSR to each node's latest state to identify and filter out potential attacks.
        Focus only on velocity components (vx and vy).
        """
        if current_step < 48:
        # Skip filtering during the transitory state
            pass
        else:
            for vehicle_id in self.nodes:

                # Collect the latest velocities of all neighbors
                neighbors_velocities = [{'vx': self.node_states[neighbor][-1]['vx']}
                                        for neighbor in self.nodes if neighbor != vehicle_id and len(self.node_states[neighbor]) > 0]    
                if neighbors_velocities:
                    consensus_velocity = self.wmsr(neighbors_velocities, f=1)  # Assume up to 1 malicious node

                    # You can then compare the filtered state with the current state to detect discrepancies
                    if len(self.node_states[vehicle_id]) > 0:
                        current_velocity = {'vx': self.node_states[vehicle_id][-1]['vx']}
                        if not self.is_consistent(current_velocity, consensus_velocity):
                            print(f"Potential attack detected for vehicle {vehicle_id}.")
                            self.calc_new_velocity(vehicle_id,consensus_velocity)
                            print('done')
    
    def calc_new_velocity(self, vehicle_id, consensus_velocity):
        """ 
        Calculate the new velocity for the Attaced vehicle.
        """
        # Iterate through the current transactions to find the vehicle with the matching vehicle_id
        for vehicle in self.current_transactions:
            if vehicle['vehicle_id'] == vehicle_id:
                # Update the vx value with the desired velocity consensus_velocity
                value = vehicle['status']['vx']
                print(vehicle['status']['vx'])
                vehicle['status']['vx'] = consensus_velocity
                print('Velocity of vehicle',vehicle_id,'with value of:',value,'updated to:', consensus_velocity) 
            else:
                # If vehicle_id is not found, return a message indicating the vehicle is not in the list
                return f"Vehicle with id {vehicle_id} not found."

    def wmsr(self, neighbors_velocities, f):
        """
        WMSR algorithm to filter out up to f highest and f lowest values.
        :param neighbors_states: List of states from neighboring nodes
        :param f: Number of extreme values to filter
        :return: Filtered state (consensus value)
        """
        if len(neighbors_velocities) == 0:
            return None

        else:


            values = [d['vx'] for d in neighbors_velocities]
            values.sort()
            lenlist = len(neighbors_velocities)

            # Remove f highest and f lowest values
            if len(values) > 2 * f:
                values = values[f: -f]

                # Calculate the mean of the remaining values
                consensus_velocity = round(sum(values) / lenlist, 2) if len(values) > 0 else 0


        return consensus_velocity

    def is_consistent(self, current_velocity, consensus_velocity):
        """
        Check if the current state is consistent with the filtered state.
        :param current_state: The current state of the node
        :param filtered_state: The consensus value obtained from WMSR
        :return: True if consistent, False otherwise
        """

        current = current_velocity['vx']
        print('current:',current)
        print('consensus:',consensus_velocity)
        tolerance = 5  # Define a tolerance level for consistency check

        if abs(current - consensus_velocity) > tolerance:
            return False
        else:
            return True
    

    @property
    def last_block(self):
        return self.chain[-1]

    @staticmethod
    def hash(block):
        """
        Creates a SHA-256 hash of a Block
        :param block: Block
        """
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def proof_of_work(self, last_block):
        """
        Simple Proof of Work Algorithm
        :param last_block: <dict> last Block
        :return: <int> Proof of work
        """
        last_proof = last_block['proof']
        last_hash = self.hash(last_block)

        proof = 0
        while self.valid_proof(last_proof, proof, last_hash) is False:
            proof += 1

        return proof

    @staticmethod
    def valid_proof(last_proof, proof, last_hash):
        """
        Validates the Proof
        :param last_proof: <int> Previous Proof
        :param proof: <int> Current Proof
        :param last_hash: <str> The hash of the Previous Block
        :return: <bool> True if correct, False if not
        """
        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

class Vehicle:
    def __init__(self, id, blockchain, x=0.0, y=0.0, yaw=0.0, vx=10.0, vy=0.0, yaw_rate=0.0, delta=0.0):
        self.id = id  # Vehicle ID
        self.x = x  # X position
        self.y = y  # Y position
        self.yaw = yaw  # Yaw angle
        self.vx = vx  # Longitudinal velocity
        self.vy = vy  # Lateral velocity
        self.yaw_rate = yaw_rate  # Yaw rate
        self.delta = delta  # Steering angle (control input)
        self.blockchain = blockchain  # Reference to the shared blockchain
        self.velocity_history = []  # Store velocity history for plotting

        # Register vehicle as a node in the blockchain network
        blockchain.register_node(self.id,self.get_state())

        # Vehicle-specific parameters
        self.mass = 1500  # kg
        self.lf = 1.2  # Distance from CG to front axle (m)
        self.lr = 1.6  # Distance from CG to rear axle (m)
        self.Iz = 2250  # Yaw moment of inertia (kg*m^2)
        self.Cf = 19000  # Cornering stiffness front (N/rad)
        self.Cr = 33000  # Cornering stiffness rear (N/rad)
        self.dt = 0.01  # Time step (s)
    

    def get_state(self):
        """
        Get the current state of the vehicle
        :return: A dictionary containing the vehicle's state
        """
        return {
            'x': round(float(self.x), 2),
            'vx': round(float(self.vx), 2),
   
        }
    
    def update_dynamics(self, neighbors_info):
        """
        Update the vehicle dynamics using the control input based on neighbors' information.
        """
        # Compute control input based on neighbors' information
        control_input = self.compute_control_input(neighbors_info)

        # Apply control input to adjust the vehicle's steering angle and velocity
        self.delta = control_input['delta']
        self.vx += control_input['vx_adjust']  # Adjust longitudinal velocity

        # Calculate slip angles
        vx_safe = max(self.vx, 0.1) # Prevent vx from being too small
        alpha_f = self.delta - (self.vy + self.lf * self.yaw_rate) / vx_safe
        alpha_r = - (self.vy - self.lr * self.yaw_rate) / vx_safe
        Fyf = -self.Cf * alpha_f  # Lateral force at the front tire
        Fyr = -self.Cr * alpha_r  # Lateral force at the rear tire

        # Calculate state derivatives
        vy_dot = (Fyf + Fyr) / self.mass - self.vx * self.yaw_rate
        yaw_rate_dot = (self.lf * Fyf - self.lr * Fyr) / self.Iz
        x_dot = self.vx * np.cos(self.yaw) - self.vy * np.sin(self.yaw)
        y_dot = self.vx * np.sin(self.yaw) + self.vy * np.cos(self.yaw)
        yaw_dot = self.yaw_rate

        # Update the state
        self.vy += vy_dot * self.dt
        self.yaw_rate += yaw_rate_dot * self.dt
        self.yaw += yaw_dot * self.dt
        self.x += x_dot * self.dt
        self.y += y_dot * self.dt

        # Broadcast the updated state to the blockchain
        self.broadcast_state()

    def compute_control_input(self, neighbors_info):
        """
        Compute the control input (steering angle, velocity adjustment) based on neighbors' information from the blockchain.
        Now using the Average Consensus Algorithm instead of flocking.
        """
        # Initialize the control input
        control_input = {'delta': 0, 'vx_adjust': 0}

        # Average Consensus control parameters
        consensus_gain = 0.1  # Adjust this gain as needed based on system dynamics and topology
        tau = 0.05  # Time delay, can be adjusted

        # Variables to store sums of neighbors' states for averaging
        avg_x = 0
        avg_y = 0
        avg_vx = 0
        avg_vy = 0
        num_neighbors = len(neighbors_info)

        if num_neighbors > 0:
            for neighbor in neighbors_info:
                # Position and velocity consensus
                avg_x += neighbor['x']
                avg_y += neighbor['y']
                avg_vx += neighbor['vx']
                avg_vy += neighbor['vy']

            # Compute average position and velocity of neighbors
            avg_x /= num_neighbors
            avg_y /= num_neighbors
            avg_vx /= num_neighbors
            avg_vy /= num_neighbors

            # Update vehicle's position and velocity based on average consensus algorithm
            control_input['delta'] = consensus_gain * (avg_x - self.x)  # Position alignment
            control_input['vx_adjust'] = consensus_gain * (avg_vx - self.vx)  # Velocity alignment

        return control_input

    def broadcast_state(self):
        # Create a transaction that represents the vehicle's state
        state = self.get_state()
        blockchain.new_transaction(vehicle_id=self.id, status=state)

    def get_state(self):
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'yaw': self.yaw,
            'vx': self.vx,
            'vy': self.vy
        }
    def check_and_update_agents_from_blockchain(self,agents):
        """
        After each block, retrieve the last block from the blockchain and update each agent's state
        """
        # Get the last block from the blockchain
        last_block = blockchain.chain[-1]

        # Loop through all transactions in the last block
        for transaction in last_block['transactions']:
            vehicle_id = transaction['vehicle_id']
            status = transaction['status']

            # Find the corresponding agent and update its state
            for agent in agents:
                if agent.id == vehicle_id:
                    agent.x = status['x']
                    agent.vx = status['vx']
                    #print(f"Agent {agent.id} updated from blockchain state.")
                    self.velocity_history.append(agent.vx)  # Store the current velocity


def get_neighbors_info(block, vehicle_id):
    """
    Retrieve neighbors' information from the last mined block.
    """
    # Extract all transactions (vehicle states) except for the current vehicle's state
    neighbors_info = [tx['status'] for tx in block['transactions'] if tx['vehicle_id'] != vehicle_id]
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
    num_agents = 10
    # Create a single blockchain instance
    blockchain = Blockchain(num_agents)

    # Initialize a few vehicles with different IDs
    vehicles = [
        Vehicle(id=i+1, blockchain=blockchain, x=4*(i+1), y=i*2, vx=2*(i+1)) for i in range(num_agents)
    ]

    # List to store minimum distances over time
    min_distances = []

    attacked_Agents = [2,6]
    attack_steps = 50
    # Run the simulation for a few steps
    for step in range(100):
        print(f"Step {step}:\n{'-'*20}")

        for vehicle in vehicles:
            # Retrieve the last mined block to get neighbors' information
            if len(blockchain.chain) > 1:
                neighbors_info = get_neighbors_info(blockchain.last_block, vehicle.id)
            else:
                neighbors_info = []

            # Introduce the attack
            if attack_steps <step< attack_steps+10 and vehicle.id in attacked_Agents:
                # Modify the velocity drastically to simulate an attack
                vehicle.vx += random.uniform(10, 80)  # Add an unrealistic jump in velocity
                print(f"Cyber attack introduced to vehicle {vehicle.id} at step {step}")

            vehicle.update_dynamics(neighbors_info)

            #print(f'Vehicle_{vehicle.id} at ({vehicle.x:.2f}, {vehicle.y:.2f}), velocity: {vehicle.vx:.2f}')

            # Pass the current step to `check_for_attacks()`
            blockchain.check_for_attacks(current_step=step )

            # Mine a new block after all vehicles have updated their states
            proof = blockchain.proof_of_work(blockchain.last_block)
            blockchain.new_block(proof, blockchain.hash(blockchain.last_block))
            vehicle.check_and_update_agents_from_blockchain(vehicles)
        
        # Calculate and store the minimum distance between vehicles
        min_distance = calculate_minimum_distance(vehicles)
        min_distances.append(min_distance)


    # Step 3: Plot the velocity consensus over time
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
