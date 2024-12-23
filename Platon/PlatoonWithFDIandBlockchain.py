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
        self.chain.append(block)
        self.current_transactions = []

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

    def check_for_attacks(self,current_step, v_l):
        """
        Apply WMSR to each node's latest state to identify and filter out potential attacks.
        Focus only on velocity components (vx and vy).
        """
        if current_step < 10:
        # Skip filtering during the transitory state
            pass
        else:
            for vehicle_id in self.nodes:
                consensus_velocity = v_l
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
        # print('//////')
        # print(self.current_transactions)
        # print('//////')

        # Iterate through the current transactions to find the vehicle with the matching vehicle_id
        for vehicle in self.current_transactions:
            if vehicle['vehicle_id'] == vehicle_id:
                # Update the vx value with the desired velocity consensus_velocity
                value = vehicle['status']['vx']
                # print(vehicle['status']['vx'])
                vehicle['status']['vx'] = consensus_velocity
                print('Velocity of vehicle',vehicle_id,'with value of:',value,'updated to:', consensus_velocity) 
            


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
        # print('current:',current)
        # print('consensus:',consensus_velocity)
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
        return guess_hash[:2] == "00"

class DynamicBicycleModel:
    def __init__(self, blockchain, id, x=0.0, y=0.0, psi=0.0, vx=10.0, vy=0.0, r=0.0):
        self.id = id  # Vehicle ID
        self.x = x  # X position
        self.y = y  # Y position
        self.psi = psi  # Yaw angle
        self.v_x = vx  # Longitudinal velocity
        self.v_y = vy  # Lateral velocity
        self.r = r  # Yaw rate
        self.blockchain = blockchain  # Reference to the shared blockchain

        # Register vehicle as a node in the blockchain network
        self.blockchain.register_node(self.id, self.get_state())

        self.velocity_history = []  # Store velocity history for plotting

        # Vehicle-specific parameters
        self.m = 1500  # Vehicle mass kg
        self.L_f = 1.2  # Distance from CG to front axle (m)
        self.L_r = 1.6  # Distance from CG to rear axle (m)
        self.I_z = 2250  # Yaw moment of inertia (kg*m^2)
        self.C_f = 19000  # Cornering stiffness front (N/rad)
        self.C_r = 20000  # Cornering stiffness rear (N/rad)
        self.dt = 0.01  # Time step (s)

    def update(self, a, delta):
        """
        Update the vehicle's state based on the bicycle model dynamics
        """
        xx = np.array([self.x, self.y, self.psi, self.v_x, self.v_y, self.r])

        k1 = self.f(xx, a, delta)
        k2 = self.f(xx + self.dt / 2 * k1, a, delta)
        k3 = self.f(xx + self.dt / 2 * k2, a, delta)
        k4 = self.f(xx + self.dt * k3, a, delta)

        xx = xx + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        self.x = xx[0]  # Global x position
        self.y = xx[1]  # Global y position
        self.psi = xx[2]  # Yaw angle
        self.v_x = xx[3]  # Longitudinal velocity
        self.v_y = xx[4]  # Lateral velocity
        self.r = xx[5]  # Yaw rate

        # Broadcast the updated state to the blockchain
        self.broadcast_state()

    def f(self, xx, a, delta):
        """
        calculating the states' derivatives using Runge-Kutta Algorithm
        """
        x = xx[0]  # Global x position
        y = xx[1]  # Global y position
        psi = xx[2]  # Yaw angle
        v_x = xx[3]  # Longitudinal velocity
        v_y = xx[4]  # Lateral velocity
        r = xx[5]  # Yaw rate

        # Calculate slip angles
        alpha_f = delta - np.arctan2((v_y + self.L_f * r), v_x)
        alpha_r = -np.arctan2((v_y - self.L_r * r), v_x)

        # Calculate lateral forces
        F_yf = self.C_f * alpha_f
        F_yr = self.C_r * alpha_r

        # State variables
        x_dot = v_x * np.cos(psi) - v_y * np.sin(psi)  # Global x position
        y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)  # Global y position
        psi_dot = r  # Yaw angle
        v_x_dot = a - (F_yf * np.sin(delta)) / self.m + v_y * r  # Longitudinal velocity
        v_y_dot = (F_yf * np.cos(delta) + F_yr) / self.m - v_x * r  # Lateral velocity
        r_dot = (self.L_f * F_yf * np.cos(delta) - self.L_r * F_yr) / self.I_z  # Yaw rate

        return np.array([x_dot, y_dot, psi_dot, v_x_dot, v_y_dot, r_dot])

    def broadcast_state(self):
        # Create a transaction that represents the vehicle's state
        state = self.get_state()
        self.blockchain.new_transaction(vehicle_id=self.id, status=state)

    def get_state(self):
        """
        Get the current state of the vehicle
        :return: A dictionary containing the vehicle's state
        """
        return {
            'x': round(float(self.x), 2),
            'y': round(float(self.y), 2),
            'yaw': round(float(self.psi), 2),
            'vx': round(float(self.v_x), 2),
            'vy': round(float(self.v_y), 2)
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
                # print(agent.id)
                if agent.id == vehicle_id:
                    agent.x = status['x']
                    agent.v_x = status['vx']
                    #print(f"Agent {agent.id} updated from blockchain state.")
                    self.velocity_history.append(agent.v_x)  # Store the current velocity


class LeaderFollowerSimulation:
    def __init__(self, num_followers, blockchain):
        self.leader = DynamicBicycleModel(blockchain, id=0, x=100, y=10, vx=6.0)
        # Initialize followers with slightly different initial velocities
        self.followers = [
            DynamicBicycleModel(
                blockchain, 
                id=i + 1, 
                x=100 - 10 * (i + 1), 
                y=10, 
                vx=6.0   # Add random initial velocity offset
            ) for i in range(num_followers)
        ]
        self.blockchain = blockchain
        self.num_followers = num_followers
        self.desired_gap = 10  # Desired gap between vehicles (m)
        self.dt = 0.05
        self.time_steps = int(50 / self.dt)
        self.road_width = 20  # Width of the road (meters)

    def run_simulation(self):
        x_history = [[] for _ in range(self.num_followers + 1)]
        y_history = [[] for _ in range(self.num_followers + 1)]
        v_history = [[] for _ in range(self.num_followers + 1)]
        min_distances = []
        time_points = np.arange(0, self.time_steps) * self.dt

        for t in range(self.time_steps):
            # Update leader's state
            v_target = 6.0
            k_p = 1
            a_l = k_p * (v_target - self.leader.v_x)
            self.leader.update(0, 0)

            # Save leader's position and velocity
            x_history[0].append(self.leader.x)
            y_history[0].append(self.leader.y)
            v_history[0].append(self.leader.v_x)

            min_dist_timestep = float('inf')

            # Update each follower with modified control gains
            for i, follower in enumerate(self.followers):
                # Modify the control gain for each follower to show different convergence rates
                k_p_follower = 1 
                distance_to_leader = self.leader.x - follower.x - self.desired_gap * (i + 1)
                a_f = k_p_follower * distance_to_leader
                if 30 <t< 35 and i == 8:
                    # Modify the velocity drastically to simulate an attack
                    follower.v_x += random.uniform(5, 10)  # Add an unrealistic jump in velocity
                    print(f"Cyber attack introduced to vehicle {follower.id} at step {t}")

                follower.update(0, 0)

                x_history[i + 1].append(follower.x)
                y_history[i + 1].append(follower.y)
                # v_history[i + 1].append(follower.v_x)

                if i == 0:
                    dist = np.sqrt((follower.x - self.leader.x)**2 + (follower.y - self.leader.y)**2)
                else:
                    dist = np.sqrt((follower.x - self.followers[i-1].x)**2 + 
                                 (follower.y - self.followers[i-1].y)**2)
                min_dist_timestep = min(min_dist_timestep, dist)

            min_distances.append(min_dist_timestep)

            # Pass the current step to `check_for_attacks()`
            blockchain.check_for_attacks(current_step=t ,v_l=self.leader.v_x)
            # Mine a new block after all vehicles have updated their states
            proof = blockchain.proof_of_work(blockchain.last_block)
            blockchain.new_block(proof, blockchain.hash(blockchain.last_block))
            for i, follower in enumerate(self.followers):
                follower.check_and_update_agents_from_blockchain(self.followers)
                v_history[i + 1].append(follower.v_x)

        # Plot 1: Trajectory snapshots [same as before]
        plt.figure(figsize=(15, 8))
        plt.suptitle('Vehicle Platooning Trajectory Snapshots', fontsize=14)
        t_samples = [int(self.time_steps * 0.01), int(self.time_steps * 0.2), int(self.time_steps * 0.4),
                     int(self.time_steps * 0.6), int(self.time_steps * 0.8), int(self.time_steps * 0.99)]
        
        for idx, t in enumerate(t_samples):
            plt.subplot(3, 2, idx + 1)
            for i in range(self.num_followers + 1):
                plt.plot(x_history[i][t], y_history[i][t], '.k' if i > 0 else '.r', markersize=10 if i == 0 else 5)

            plt.plot([0, self.leader.x + 20], [0, 0], 'b-', linewidth=2)
            plt.plot([0, self.leader.x + 20], [self.road_width, self.road_width], 'b-', linewidth=2)

            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title(f't={t * self.dt:.2f} sec')
            plt.ylim(-5, self.road_width + 5)
        plt.tight_layout()
        plt.show()

        # Plot 2: Velocity consensus with zoomed y-axis
        plt.figure(figsize=(10, 6))
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_followers + 1))
        for i in range(self.num_followers + 1):
            label = 'Leader' if i == 0 else f'Follower {i}'
            plt.plot(time_points, v_history[i], label=label, color=colors[i], linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Velocity Consensus Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        # Set y-axis limits to better show velocity differences
        plt.ylim(0, 20)
        plt.tight_layout()
        plt.show()

        # Plot 3: Minimum distances [same as before]
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, min_distances, 'b-', label='Minimum Distance')
        plt.axhline(y=self.desired_gap, color='r', linestyle='--', label='Desired Gap')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.title('Minimum Inter-Vehicle Distance Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Run the simulation
num_followers = 10
blockchain = Blockchain(num_followers)
simulation = LeaderFollowerSimulation(num_followers, blockchain)
simulation.run_simulation()
print(blockchain.nodes)