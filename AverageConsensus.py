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
        :param vehicle_id: ID of the vehicle
        :param status: Current state of the vehicle
        :return: The index of the Block that will hold this transaction
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
        Vehicle(id=f'Vehicle_{i+1}', blockchain=blockchain, x=i*10, y=i*2) for i in range(5)
    ]

    # Run the simulation for a few steps
    for step in range(30):
        print(f"Step {step}:\n{'-'*20}")

        for vehicle in vehicles:
            # Retrieve the last mined block to get neighbors' information
            if len(blockchain.chain) > 1:
                neighbors_info = get_neighbors_info(blockchain.last_block, vehicle.id)
                #print('neighbors_info = ', neighbors_info)
            else:
                neighbors_info = []

            vehicle.update_dynamics(neighbors_info)
            print(f'{vehicle.id} at ({vehicle.x:.2f}, {vehicle.y:.2f}), velocity: {vehicle.vx:.2f}')

        # Mine a new block after all vehicles have updated their states
        proof = blockchain.proof_of_work(blockchain.last_block)
        blockchain.new_block(proof, blockchain.hash(blockchain.last_block))

        #print(f"Blockchain at step {step}:")
        #print(json.dumps(blockchain.chain, indent=4))

