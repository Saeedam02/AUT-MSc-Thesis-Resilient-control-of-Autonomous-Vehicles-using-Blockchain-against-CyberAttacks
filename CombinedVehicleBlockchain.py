import hashlib
import json
from time import time
import numpy as np

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

        :param sender: Address of the Sender
        :param recipient: Address of the Recipient
        :param amount: Amount
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

        # We must make sure that the Dictionary is Ordered, or we'll have inconsistent hashes
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def proof_of_work(self, last_block):
        """
        Simple Proof of Work Algorithm:

         - Find a number p' such that hash(pp') contains leading 4 zeroes
         - Where p is the previous proof, and p' is the new proof
         
        :param last_block: <dict> last Block
        :return: <int>
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
        :return: <bool> True if correct, False if not.

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

    def update_dynamics(self):
        # Calculate slip angles
        alpha_f = self.delta - (self.vy + self.lf * self.yaw_rate) / self.vx
        alpha_r = - (self.vy - self.lr * self.yaw_rate) / self.vx
        Fyf = -self.Cf * alpha_f  # Lateral force at the front tire
        Fyr = -self.Cr * alpha_r  # Lateral force at the rear tire

        # Calculate state derivatives
        vy_dot = (Fyf + Fyr) / self.mass - self.vx * self.yaw_rate
        yaw_rate_dot = (self.lf * Fyf - self.lr * Fyr) / self.Iz
        x_dot = self.vx * np.cos(self.yaw) - self.vy * self.dt
        y_dot = self.vx * np.sin(self.yaw) + self.vy * self.dt
        yaw_dot = self.yaw_rate

        # Update the state
        self.vy += vy_dot * self.dt
        self.yaw_rate += yaw_rate_dot * self.dt
        self.yaw += yaw_dot * self.dt
        self.x += x_dot * self.dt
        self.y += y_dot * self.dt

        # Broadcast the updated state to the blockchain
        self.broadcast_state()

    def broadcast_state(self):
        # Create a transaction that represents the vehicle's state
        state = self.get_state()
        self.blockchain.new_transaction(vehicle_id=self.id, status=self.get_state())

    def get_state(self):
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'yaw': self.yaw,
            'vx' : self.vx,
            'vy' : self.vy
        }

# Simulation setup
if __name__ == "__main__":
    # Create a single blockchain instance
    blockchain = Blockchain()

    # Initialize a few vehicles with different IDs
    vehicles = [
        Vehicle(id=f'Vehicle_{i+1}', blockchain=blockchain, x=i*10 , y=i*2) for i in range(10)
    ]

    # Run the simulation for a few steps
    for step in range(10):
        for vehicle in vehicles:
            
            vehicle.update_dynamics()
            print(f'{vehicle.id} at ({vehicle.x:.2f}, {vehicle.y})')
            #status = vehicle.get_state()
            #print(status)
            #vehicle.blockchain.new_transaction(vehicle.id ,status)


        proof = blockchain.proof_of_work(blockchain.last_block)
        vehicle.blockchain.new_block(proof,blockchain.hash(blockchain.last_block))
        # You can print the blockchain state periodically to observe the transactions
        #if step % 10 == 0:
        print(f"Blockchain at step {step}:")
        print(json.dumps(blockchain.chain, indent=4))

   
def full_chain():
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain),
    }
    return response

def consensus():
    replaced = blockchain.resolve_conflicts()

    if replaced:
        response = {
            'message': 'Our chain was replaced',
            'new_chain': blockchain.chain
        }
    else:
        response = {
            'message': 'Our chain is authoritative',
            'chain': blockchain.chain
        }

    return response
