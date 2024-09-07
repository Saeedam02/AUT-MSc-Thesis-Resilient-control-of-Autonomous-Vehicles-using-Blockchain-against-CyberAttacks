import hashlib
import json
from time import time
import numpy as np
import matplotlib.pyplot as plt

import requests

class Blockchain:
    def __init__(self):
        self.current_transactions = []
        self.chain = []
        self.nodes = set()

        # Create the genesis block
        self.new_block(previous_hash='1', proof=100)

    def register_node(self, vehicle_id):
        """
        Add a new node (vehicle) to the list of nodes

        :param vehicle_id: ID of the vehicle
        """
        self.nodes.add(vehicle_id)

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

    def resolve_conflicts(self):
        """
        This is our consensus algorithm, it resolves conflicts
        by replacing our chain with the longest one in the network.

        :return: True if our chain was replaced, False if not
        """

        neighbours = self.nodes
        new_chain = None

        # We're only looking for chains longer than ours
        max_length = len(self.chain)

        # Grab and verify the chains from all the nodes in our network
        for node in neighbours:
            response = requests.get(f'http://{node}/chain')

            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']

                # Check if the length is longer and the chain is valid
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain

        # Replace our chain if we discovered a new, valid chain longer than ours
        if new_chain:
            self.chain = new_chain
            return True

        return False

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
        """
        self.current_transactions.append({
            'vehicle_id': vehicle_id,
            'status': status,
        })

        return self.last_block['index'] + 1

    @property
    def last_block(self):
        return self.chain[-1]

    @staticmethod
    def hash(block):
        """
        Creates a SHA-256 hash of a Block
        """
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def proof_of_work(self, last_block):
        """
        Simple Proof of Work Algorithm
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
        """
        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

class Vehicle:
    def __init__(self, id, blockchain, x=0.0, y=0.0, yaw=0.0, vx=10.0, vy=0.0, r=0.0, delta=0.0, ax=0.0):
        self.id = id  # Vehicle ID
        self.x = x  # X position
        self.y = y  # Y position
        self.yaw = yaw  # Yaw angle
        self.vx = vx  # Longitudinal velocity
        self.vy = vy  # Lateral velocity
        self.r = r  # Yaw rate
        self.delta = delta  # Steering angle (control input)
        self.blockchain = blockchain  # Reference to the shared blockchain
        self.ax = ax # Longitudinal acceleration (driven by throttle or braking).

        # Register vehicle as a node in the blockchain network
        self.blockchain.register_node(self.id)

        # Vehicle-specific parameters
        self.mass = 1500  # kg
        self.lf = 1.2  # Distance from CG to front axle (m)
        self.lr = 1.6  # Distance from CG to rear axle (m)
        self.Iz = 2250  # Yaw moment of inertia (kg*m^2)
        self.Cf = 19000  # Cornering stiffness front (N/rad)
        self.Cr = 33000  # Cornering stiffness rear (N/rad)
        self.dt = 0.01  # Time step (s)
    
    def get_id(self):
        print(self.id)

    def update_dynamics(self, neighbors_info):
        """
        Update the vehicle dynamics using the control input based on neighbors' information.
        """
        # Neighbors info contains the list of state dictionaries for each neighboring vehicle
        control_input = self.compute_control_input(neighbors_info)

        # Apply control input (e.g., steering angle, etc.) to update the dynamics

        # Calculate slip angles
        alpha_f = control_input['delta'] - (self.vy + self.lf * self.r) / self.vx
        alpha_r = - (self.vy - self.lr * self.r) / self.vx
        
        Fyf = self.Cf * alpha_f  # Lateral force at the front tire
        Fyr = self.Cr * alpha_r  # Lateral force at the rear tire

        # Calculate state derivatives

        #In reality, lateral velocity is limited by tire friction and road conditions. 
        #damping_factor = 0.7  # Adjust this value for appropriate damping
        vy_dot = ((Fyf + Fyr) / self.mass - self.vx * self.r) #- damping_factor * self.vy
        vx_dot = self.ax - (self.r * self.vy)
        r_dot = (self.lf * Fyf - self.lr * Fyr) / self.Iz
        x_dot = self.vx * np.cos(self.yaw) - self.vy * np.sin(self.yaw)
        y_dot = self.vx * np.sin(self.yaw) + self.vy * np.cos(self.yaw)
        yaw_dot = (self.vx / (self.lf + self.lr)) * np.tan(self.delta)

        # Longitudinal velocity affected by drag to keep it realistic
        #drag_force = 0.01 * self.vx ** 2  # Drag proportional to the square of vx
        self.vx += vx_dot * self.dt

        # Update the state
        self.vy += vy_dot * self.dt
        self.r += r_dot * self.dt
        self.yaw += yaw_dot * self.dt
        self.x += x_dot * self.dt
        self.y += y_dot * self.dt

        # Broadcast the updated state to the blockchain
        self.broadcast_state()

    def compute_control_input(self, neighbors_info):
        """
        Compute the control input based on neighbors' information from the blockchain.
        """
        # Initialize the control input
        control_input = {'delta': 0}

        # Consensus-based control law to influence delta (steering angle) based on neighbors' positions
        k_p = 0.1  # Proportional gain for position alignment
        for neighbor in neighbors_info:
            control_input['delta'] += k_p * ((neighbor['x'] - self.x) + (neighbor['y'] - self.y))

        return control_input

    def broadcast_state(self):
        # Create a transaction that represents the vehicle's state
        state = self.get_state()
        self.blockchain.new_transaction(vehicle_id=self.id, status=state)

    def get_state(self):
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'yaw': self.yaw,
            'vx': self.vx,
            'vy': self.vy
        }

def get_neighbors_info(block, vehicle_id):
    """
    Retrieve neighbors' information from the last mined block.
    """
    # Extract all transactions (vehicle states) except for the current vehicle's state
    neighbors_info = [tx['status'] for tx in block['transactions'] if tx['vehicle_id'] != vehicle_id]
    return neighbors_info

# Simulation setup
if __name__ == "__main__":
    # Create a single blockchain instance
    blockchain = Blockchain()

    # Initialize a few vehicles with different IDs
    vehicles = [
        Vehicle(id=f'Vehicle_{i+1}', blockchain=blockchain, x=i*10, y=i*2) for i in range(10)
    ]

    # Run the simulation for a few steps
    for step in range(30):
        print(f"Step {step}:\n{'-'*20}")

        for vehicle in vehicles:
            # Retrieve the last mined block to get neighbors' information
            if len(blockchain.chain) > 1:
                neighbors_info = get_neighbors_info(blockchain.last_block, vehicle.id)
            else:
                neighbors_info = []

            vehicle.update_dynamics(neighbors_info)
            #print(f'{vehicle.id} at ({vehicle.x:.2f}, {vehicle.y:.2f})')

        # Mine a new block after all vehicles have updated their states
        proof = blockchain.proof_of_work(blockchain.last_block)
        blockchain.new_block(proof, blockchain.hash(blockchain.last_block))

        #print(f"Blockchain at step {step}:")
        #print(json.dumps(blockchain.chain, indent=4))


# Simulating outputs for positions and velocities to visualize consensus (since the code can't run here directly)
# Example data for positions and velocities of 10 vehicles over 10 time steps
time_steps = list(range(100))
vehicles_count = 10

# Simulated positions and velocities for each vehicle (these should come from the simulation outputs)
simulated_positions = [
    [[i * 10 + t for t in time_steps] for i in range(vehicles_count)],  # X positions
    [[i * 2 + t for t in time_steps] for i in range(vehicles_count)]  # Y positions
]

# Plot positions (X and Y) over time for all vehicles
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

for i in range(vehicles_count):
    ax1.plot(time_steps, simulated_positions[0][i], label=f'Vehicle {i+1}')
    ax2.plot(time_steps, simulated_positions[1][i], label=f'Vehicle {i+1}')

ax1.set_title('X Positions of Vehicles over Time')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('X Position')
ax1.legend()

ax2.set_title('Y Positions of Vehicles over Time')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Y Position')
ax2.legend()

plt.tight_layout()
plt.show()