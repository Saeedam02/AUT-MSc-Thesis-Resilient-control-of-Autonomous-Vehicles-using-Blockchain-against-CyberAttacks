import hashlib
import json
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import requests

class Blockchain:
    def __init__(self):
        self.current_transactions = []
        self.chain = []
        self.nodes = set()
        self.new_block(previous_hash='1', proof=100)

    def register_node(self, vehicle_id):
        self.nodes.add(vehicle_id)

    def valid_chain(self, chain):
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
        neighbours = self.nodes
        new_chain = None
        max_length = len(self.chain)

        for node in neighbours:
            response = requests.get(f'http://{node}/chain')

            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']

                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain

        if new_chain:
            self.chain = new_chain
            return True

        return False

    def new_block(self, proof, previous_hash=None):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }

        self.current_transactions = []
        self.chain.append(block)
        return block

    def new_transaction(self, vehicle_id, status):
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
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def proof_of_work(self, last_block):
        last_proof = last_block['proof']
        last_hash = self.hash(last_block)
        proof = 0
        while self.valid_proof(last_proof, proof, last_hash) is False:
            proof += 1
        return proof

    @staticmethod
    def valid_proof(last_proof, proof, last_hash):
        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"


class Vehicle:
    def __init__(self, id, blockchain, x=0.0, y=0.0, yaw=0.0, vx=10.0, vy=0.0, yaw_rate=0.0, delta=0.0):
        self.id = id
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.yaw_rate = yaw_rate
        self.delta = delta
        self.blockchain = blockchain
        self.blockchain.register_node(self.id)
        self.mass = 1500
        self.lf = 1.2
        self.lr = 1.6
        self.Iz = 2250
        self.Cf = 19000
        self.Cr = 33000
        self.dt = 0.01
    
    def update_dynamics(self, neighbors_info):
        control_input = self.compute_control_input(neighbors_info)
        self.delta = control_input['delta']
        self.vx += control_input['vx_adjust']
        alpha_f = self.delta - (self.vy + self.lf * self.yaw_rate) / self.vx
        alpha_r = - (self.vy - self.lr * self.yaw_rate) / self.vx
        Fyf = -self.Cf * alpha_f
        Fyr = -self.Cr * alpha_r
        vy_dot = (Fyf + Fyr) / self.mass - self.vx * self.yaw_rate
        yaw_rate_dot = (self.lf * Fyf - self.lr * Fyr) / self.Iz
        x_dot = self.vx * np.cos(self.yaw) - self.vy * self.dt
        y_dot = self.vx * np.sin(self.yaw) + self.vy * self.dt
        yaw_dot = self.yaw_rate
        self.vy += vy_dot * self.dt
        self.yaw_rate += yaw_rate_dot * self.dt
        self.yaw += yaw_dot * self.dt
        self.x += x_dot * self.dt
        self.y += y_dot * self.dt
        self.broadcast_state()

    def compute_control_input(self, neighbors_info):
        control_input = {'delta': 0, 'vx_adjust': 0}
        k_p = 0.1
        k_v = 0.05
        k_rep = 0.2
        avg_x = 0
        avg_y = 0
        avg_vx = 0
        avg_vy = 0
        num_neighbors = len(neighbors_info)

        if num_neighbors > 0:
            for neighbor in neighbors_info:
                avg_x += neighbor['x']
                avg_y += neighbor['y']
                avg_vx += neighbor['vx']
                avg_vy += neighbor['vy']
                distance = np.sqrt((neighbor['x'] - self.x) ** 2 + (neighbor['y'] - self.y) ** 2)
                if distance < 5:
                    control_input['delta'] -= k_rep / (distance ** 2) * (self.x - neighbor['x'])
                    control_input['delta'] -= k_rep / (distance ** 2) * (self.y - neighbor['y'])

            avg_x /= num_neighbors
            avg_y /= num_neighbors
            avg_vx /= num_neighbors
            avg_vy /= num_neighbors
            control_input['delta'] += k_p * ((avg_x - self.x) + (avg_y - self.y))
            control_input['vx_adjust'] += k_v * ((avg_vx - self.vx) + (avg_vy - self.vy))

        return control_input

    def broadcast_state(self):
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
    neighbors_info = [tx['status'] for tx in block['transactions'] if tx['vehicle_id'] != vehicle_id]
    return neighbors_info


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

# Setup for position plot (ax1)
ax1.set_xlim(-100, 100)
ax1.set_ylim(-100, 100)
ax1.set_title("Vehicle Positions")

# Setup for velocity plot (ax2)
ax2.set_xlim(0, 100)  # Time will be plotted on the x-axis
ax2.set_ylim(0, 20)  # Velocity magnitude on the y-axis
ax2.set_title("Vehicle Velocities (vx)")
ax2.set_xlabel("Time (frames)")
ax2.set_ylabel("Velocity (vx)")

# Create scatter plot for each vehicle's position
vehicles_plots = []

# Generate unique colors for each vehicle
colors = ['red', 'blue', 'green', 'orange', 'purple']

for color in colors:
    plot, = ax1.plot([], [], 'o', color=color, markersize=8)
    vehicles_plots.append(plot)

# Create lines for each vehicle's velocity
vel_lines = []
for color in colors:
    line, = ax2.plot([], [], color=color, lw=2)
    vel_lines.append(line)

# To store velocity history
time_data = []
vel_data = {i: [] for i in range(len(colors))}

# Initialize the plots
def init():
    for plot in vehicles_plots:
        plot.set_data([], [])
    for line in vel_lines:
        line.set_data([], [])
    return vehicles_plots + vel_lines

# Update function for animation
def update(frame):
    time_data.append(frame)
    
    for i, vehicle in enumerate(vehicles):
        if len(blockchain.chain) > 1:
            neighbors_info = get_neighbors_info(blockchain.last_block, vehicle.id)
        else:
            neighbors_info = []
        vehicle.update_dynamics(neighbors_info)

        # Update position on the position plot
        vehicles_plots[i].set_data(vehicle.x, vehicle.y)

        # Track velocity over time
        vel_data[i].append(vehicle.vx)

        # Update velocity plot
        vel_lines[i].set_data(time_data, vel_data[i])

    return vehicles_plots + vel_lines


# Create the blockchain and vehicles with random positions
blockchain = Blockchain()
vehicles = [Vehicle(id=f'Vehicle_{i+1}', blockchain=blockchain, x=np.random.uniform(-10, 15), y=np.random.uniform(-10, 10),vx=np.random.uniform(0, 10)) for i in range(5)]

# Animation setup
ani = FuncAnimation(fig, update, frames=np.arange(0, 100, 1), init_func=init, blit=True, interval=100)

plt.tight_layout()
plt.show()
# List to store velocities over time for all vehicles
vehicles_velocities = {i: [] for i in range(len(vehicles))}

# Simulation and block mining
for step in range(20):  # Run for 100 steps
    print(f"Step {step}:\n{'-'*20}")

    for i, vehicle in enumerate(vehicles):
        if len(blockchain.chain) > 1:
            neighbors_info = get_neighbors_info(blockchain.last_block, vehicle.id)
        else:
            neighbors_info = []

        # Update vehicle dynamics with neighbors' information
        vehicle.update_dynamics(neighbors_info)

        # Store the current velocity for each vehicle
        vehicles_velocities[i].append(vehicle.vx)

        print(f'{vehicle.id} at ({vehicle.x:.2f}, {vehicle.y:.2f}), velocity: {vehicle.vx:.2f}')

    # Mine a new block after all vehicles have updated their states
    proof = blockchain.proof_of_work(blockchain.last_block)
    blockchain.new_block(proof, blockchain.hash(blockchain.last_block))

    #print(f"Blockchain at step {step}:")
    #print(json.dumps(blockchain.chain, indent=4))

# After the simulation, the velocities over time are stored in `vehicles_velocities`
# Now you can inspect or process the velocity data.

# Example of how to print the velocities for each vehicle
for i in range(len(vehicles)):
    print(f"Vehicle {i+1} velocities over time: {vehicles_velocities[i]}")
