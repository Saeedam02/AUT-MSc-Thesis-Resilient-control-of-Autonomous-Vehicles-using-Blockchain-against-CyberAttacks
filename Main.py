import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from time import time
from uuid import uuid4

# Blockchain class definition

class Blockchain:
    def __init__(self):
        self.current_transactions = []  # List of transactions for the current block
        self.chain = []  # The blockchain itself, a list of blocks
        self.nodes = set()  # Set of nodes in the network

        # Create the genesis block (the first block in the chain)
        self.new_block(previous_hash='1', proof=100)

    def new_block(self, proof, previous_hash):
        """
        Create a new block and add it to the blockchain
        :param proof: The proof returned by the proof of work algorithm
        :param previous_hash: Hash of the previous block
        :return: New block
        """
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }

        # Reset the current list of transactions after adding them to the block
        self.current_transactions = []

        # Append the new block to the blockchain
        self.chain.append(block)
        return block

    def new_transaction(self, sender, recipient, amount):
        """
        Create a new transaction (state information) to go into the next mined block
        :param sender: Identifier for the sender (vehicle ID)
        :param recipient: Identifier for the recipient (usually "network" in this context)
        :param amount: The state data of the vehicle
        :return: The index of the block that will hold this transaction
        """
        self.current_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
        })
        return self.last_block['index'] + 1

    @property
    def last_block(self):
        """
        Returns the last block in the chain
        """
        return self.chain[-1]

    @staticmethod
    def hash(block):
        """
        Create a SHA-256 hash of a block
        :param block: Block
        :return: SHA-256 hash of the block
        """
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def proof_of_work(self, last_block):
        """
        Simple proof of work algorithm:
        Find a number p such that hash(pp') contains leading 4 zeros, where p is the previous proof and p' is the new proof
        :param last_block: Last block in the chain
        :return: New proof
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
        Validates the proof: Does hash(last_proof, proof, last_hash) contain 4 leading zeros?
        :param last_proof: Previous proof
        :param proof: Current proof
        :param last_hash: Hash of the last block
        :return: True if correct, False otherwise
        """
        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"


#Vehicle dynamic model (bicycle model)
class Vehicle:
    def __init__(self, id, x=0.0, y=0.0, yaw=0.0, vx=10.0, vy=0.0, r=0.0, delta=0.0, ax=0.0):
        self.id = id  # Vehicle ID
        self.x = x  # X position
        self.y = y  # Y position
        self.yaw = yaw  # Yaw angle
        self.vx = vx  # Longitudinal velocity
        self.vy = vy  # Lateral velocity
        self.r = r  # Yaw rate
        self.delta = delta  # Steering angle (control input)
        self.ax = ax # Longitudinal acceleration (driven by throttle or braking).

        # Vehicle-specific parameters
        self.mass = 1500  # kg
        self.lf = 1.2  # Distance from CG to front axle (m)
        self.lr = 1.6  # Distance from CG to rear axle (m)
        self.Iz = 2250  # Yaw moment of inertia (kg*m^2)
        self.Cf = 19000  # Cornering stiffness front (N/rad)
        self.Cr = 20000  # Cornering stiffness rear (N/rad)
        self.dt = 0.01  # Time step (s)

    def update_dynamics(self):
        """
        Update the vehicle's state based on the bicycle model dynamics
        """
        # Calculate slip angles (clamped to avoid excessive forces)
        #alpha_f = np.clip(self.delta - (self.vy + self.lf * self.r) / self.vx, -0.1, 0.1)
        #alpha_r = np.clip(-(self.vy - self.lr * self.r) / self.vx, -0.1, 0.1)

        alpha_f = self.delta - (self.vy + self.lf * self.r) / self.vx
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

    def get_state(self):
        """
        Get the current state of the vehicle
        :return: A dictionary containing the vehicle's state
        """
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'yaw': self.yaw
        }

    def get_velocity(self):
        """
        Calculate and return the total velocity of the vehicle.
        :return: The magnitude of the total velocity.
        """
        return np.sqrt(self.vx**2 + self.vy**2)
# Flocking algorithm

def flocking_vehicles(vehicles):
    """
    Implement the flocking behavior for the group of vehicles
    :param vehicles: List of Vehicle objects
    """
    separation_weight = 1.5  # Weight for separation force
    alignment_weight = 1.0  # Weight for alignment force
    cohesion_weight = 1.0  # Weight for cohesion force

    for vehicle in vehicles:
        separation_force = np.array([0.0, 0.0])
        alignment_force = np.array([0.0, 0.0])
        cohesion_force = np.array([0.0, 0.0])
        num_neighbors = 0

        for neighbor in vehicles:
            if neighbor.id != vehicle.id:
                distance = np.linalg.norm([vehicle.x - neighbor.x, vehicle.y - neighbor.y])
                if distance < 10:  # Consider as a neighbor if within 10 meters
                    num_neighbors += 1

                    # Separation
                    separation_force += (np.array([vehicle.x, vehicle.y]) - np.array([neighbor.x, neighbor.y])) / distance

                    # Alignment
                    alignment_force += np.array([neighbor.vx, neighbor.vy])

                    # Cohesion
                    cohesion_force += np.array([neighbor.x, neighbor.y])

        if num_neighbors > 0:
            separation_force /= num_neighbors
            alignment_force /= num_neighbors
            alignment_force -= np.array([vehicle.vx, vehicle.vy])
            cohesion_force /= num_neighbors
            cohesion_force = cohesion_force - np.array([vehicle.x, vehicle.y])

            # Combine the forces
            force = (separation_weight * separation_force +
                     alignment_weight * alignment_force +
                     cohesion_weight * cohesion_force)

            # Apply the force to steering (delta)
            vehicle.delta = np.arctan2(force[1], force[0])

# Communication between vehicles using blockchain

def communicate_states(vehicles, blockchain):
    """
    Simulate communication between vehicles using blockchain to share state information
    :param vehicles: List of Vehicle objects
    :param blockchain: Blockchain object
    """
    for vehicle in vehicles:
        state = vehicle.get_state()  # Get the current state of the vehicle
        blockchain.new_transaction(
            sender=str(vehicle.id),
            recipient="network",
            amount=state
        )

    # After transactions are recorded, each vehicle can "mine" a new block
    for vehicle in vehicles:
        proof = blockchain.proof_of_work(blockchain.last_block)
        blockchain.new_block(proof, blockchain.hash(blockchain.last_block))




# Initialize vehicles and blockchain

vehicles = [Vehicle(id=i) for i in range(5)]  # Create 5 vehicles
blockchain = Blockchain()  # Create a blockchain

# Store history for plotting

x_histories = [[] for _ in range(len(vehicles))]
y_histories = [[] for _ in range(len(vehicles))]

# Simulation loop

for t in range(100):
    flocking_vehicles(vehicles)
    communicate_states(vehicles, blockchain)

    for vehicle in vehicles:
        vehicle.update_dynamics()
        x_histories[vehicle.id].append(vehicle.x)
        y_histories[vehicle.id].append(vehicle.y)

# Plot the trajectories of the vehicles

plt.figure(figsize=(10, 8))
for i in range(len(vehicles)):
    plt.plot(x_histories[i], y_histories[i], label=f'Vehicle {i}')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Vehicle Trajectories in Flocking Simulation')
plt.legend()
plt.grid(True)
plt.show()

# Print the final blockchain state for inspection

for block in blockchain.chain:
    print(f"Block {block['index']}:\n", block)