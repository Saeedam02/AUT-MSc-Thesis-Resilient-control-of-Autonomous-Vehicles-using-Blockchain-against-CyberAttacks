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
    def __init__(self, id, blockchain, is_malicious=False, x=0.0, y=0.0, yaw=0.0, vx=10.0, vy=0.0, r=0.0, delta=0.0, ax=0.0):
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

        self.is_malicious = is_malicious  # Flag for malicious agents

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
    

    def update_dynamics(self, neighbors_info):
        """
        Update the vehicle's state based on the bicycle model dynamics
        """
        control_input = self.compute_control_input(neighbors_info)
        self.delta = control_input['delta']
        self.vx += control_input['vx_adjust']

        # Safety for low velocities
        vx_safe = max(self.vx, 0.1)

        # Calculate slip angles
        alpha_f = self.delta - (self.vy + self.lf * self.r) / vx_safe
        alpha_r = - (self.vy - self.lr * self.r) / vx_safe

        # Lateral forces
        Fyf = self.Cf * alpha_f
        Fyr = self.Cr * alpha_r

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

        # Check if agent is malicious
        if self.is_malicious:
            # Malicious behavior: Provide wrong state
            return {'delta': np.random.uniform(-1, 1), 'vx_adjust': np.random.uniform(-5, 5)}

        if num_neighbors > 0:
            # Apply W-MSR filtering to remove extreme neighbor states
            neighbors_x = sorted([n['x'] for n in neighbors_info])
            neighbors_y = sorted([n['y'] for n in neighbors_info])
            neighbors_vx = sorted([n['vx'] for n in neighbors_info])
            neighbors_vy = sorted([n['vy'] for n in neighbors_info])

            f = 1  # Assume at most 1 malicious neighbor

            # Remove the highest and lowest f states (W-MSR algorithm)
            filtered_x = neighbors_x[f:-f] if len(neighbors_x) > 2 * f else neighbors_x
            filtered_y = neighbors_y[f:-f] if len(neighbors_y) > 2 * f else neighbors_y
            filtered_vx = neighbors_vx[f:-f] if len(neighbors_vx) > 2 * f else neighbors_vx
            filtered_vy = neighbors_vy[f:-f] if len(neighbors_vy) > 2 * f else neighbors_vy

            # Compute average of the filtered values
            avg_x = np.mean(filtered_x) if filtered_x else 0
            avg_y = np.mean(filtered_y) if filtered_y else 0
            avg_vx = np.mean(filtered_vx) if filtered_vx else 0
            avg_vy = np.mean(filtered_vy) if filtered_vy else 0

            # Compute repulsion if the agent is too close to neighbors
            for neighbor in neighbors_info:
                distance = np.sqrt((neighbor['x'] - self.x) ** 2 + (neighbor['y'] - self.y) ** 2)
                if distance < 5:
                    repulsion_x = k_rep / (distance ** 2) * (self.x - neighbor['x'])
                    repulsion_y = k_rep / (distance ** 2) * (self.y - neighbor['y'])
                    control_input['delta'] -= np.arctan2(repulsion_y, repulsion_x)

            # Compute control input using W-MSR filtered values
            angle_to_avg = np.arctan2(avg_y - self.y, avg_x - self.x)
            control_input['delta'] += k_p * (angle_to_avg - self.yaw)
            control_input['vx_adjust'] += k_v * (avg_vx - self.vx)

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


# Create the blockchain and vehicles, two of them are malicious
blockchain = Blockchain()
vehicles = [
    Vehicle(id=f'Vehicle_{i+1}', blockchain=blockchain, x=np.random.uniform(-10, 15), y=np.random.uniform(-10, 10),
            vx=np.random.uniform(0, 10), is_malicious=(i in [3, 4])) for i in range(5)
]

# Animation setup
ani = FuncAnimation(fig, update, frames=np.arange(0, 100, 1), init_func=init, blit=True, interval=100)

plt.tight_layout()
plt.show()

# Simulation and block mining
for step in range(3):
    print(f"Step {step}:\n{'-'*20}")

    for vehicle in vehicles:
        if len(blockchain.chain) > 1:
            neighbors_info = get_neighbors_info(blockchain.last_block, vehicle.id)
        else:
            neighbors_info = []

        vehicle.update_dynamics(neighbors_info)
        print(f'{vehicle.id} at ({vehicle.x:.2f}, {vehicle.y:.2f})')

    proof = blockchain.proof_of_work(blockchain.last_block)
    blockchain.new_block(proof, blockchain.hash(blockchain.last_block))

    print(f"Blockchain at step {step}:")
    print(json.dumps(blockchain.chain, indent=4))
