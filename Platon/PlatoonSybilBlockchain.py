import hashlib
import json
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from time import time
import rsa 

###############################################################################
# RSA and Blockchain Utilities
###############################################################################
def generate_keys():
    """
    Generate RSA (public_key, private_key) for demonstration.
    Using a 512-bit key for simplicity; in production, use >= 2048 bits.
    """
    (public_key, private_key) = rsa.newkeys(512)
    return public_key, private_key

def sign_data(data, private_key):
    """
    Sign a dictionary (data) using the private key.
    We convert the data (JSON) to bytes, then call rsa.sign(...).
    """
    message_bytes = json.dumps(data, sort_keys=True).encode()
    signature = rsa.sign(message_bytes, private_key, 'SHA-256')
    return signature

def verify_signature(data, signature, public_key):
    """
    Verify the signature of data using the given public_key.
    Raises rsa.VerificationError if invalid.
    """
    message_bytes = json.dumps(data, sort_keys=True).encode()
    rsa.verify(message_bytes, signature, public_key)
    return True


###############################################################################
# Blockchain Class
###############################################################################
class Blockchain:
    def __init__(self, num_agents):
        self.current_transactions = []
        self.chain = []
        self.nodes = set()

        # For backward compatibility in your code logic
        self.sings = [self.get_sign(i) for i in range(num_agents + 1)]
        self.node_states = {}  # Store the state history of each node

        # RSA public keys stored here
        self.public_keys = {}   # vehicle_id -> RSA public key

        # Create the genesis block
        self.new_block(previous_hash='1', proof=100)

    def register_node(self, vehicle_id, state, public_key):
        """
        Add a new node (vehicle) to the list of nodes, associating it with its RSA public_key.
        
        Return True if registration succeeded, False otherwise.
        """
        sign = self.get_sign(vehicle_id)
        # For this demo, let's say only IDs from the initial set {0..num_agents} 
        # are 'valid' to register:
        if vehicle_id in self.nodes:
            print(f"ERROR: The same ID {vehicle_id} is already registered in the blockchain.")
            return False
        else:
            # If this sign is not in self.sings => fail registration
            if sign in self.sings:
                self.nodes.add(vehicle_id)
                self.node_states[vehicle_id] = [state]
                self.public_keys[vehicle_id] = public_key
                print(f"Vehicle {vehicle_id} registered successfully with blockchain.")
                return True
            else:
                print(f"Registration failed for vehicle {vehicle_id}: Invalid signature.")
                return False

    def get_sign(self, vehicle_id):
        """
        Original simplistic sign from your code (not cryptographically secure).
        We still keep it for backward compatibility in the code logic.
        """
        return hashlib.sha256(str(vehicle_id).encode()).hexdigest()[:8]

    def valid_chain(self, chain):
        """
        Basic chain validation logic
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
        Create a new block in the blockchain.
        """
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        self.chain.append(block)
        self.current_transactions = []
        return block

    def new_transaction(self, vehicle_id, status, signature=None):
        """
        Creates a new transaction to go into the next mined Block,
        verifying RSA signature if provided.
        """
        if vehicle_id not in self.nodes:
            print(f"Unregistered vehicle {vehicle_id} attempted transaction. Rejected.")
            return self.last_block['index'] + 1

        # If RSA signature is provided, verify it:
        if signature is not None:
            pub_key = self.public_keys.get(vehicle_id, None)
            if pub_key:
                try:
                    verify_signature(status, signature, pub_key)
                except rsa.VerificationError:
                    print(f"[Sybil/Spoofing Attack] Signature invalid for vehicle {vehicle_id}. Transaction rejected.")
                    return self.last_block['index'] + 1
            else:
                print(f"No public key found for vehicle {vehicle_id}. Transaction rejected.")
                return self.last_block['index'] + 1

        # If all checks pass, add transaction
        self.current_transactions.append({
            'vehicle_id': vehicle_id,
            'status': status,
        })

        # Also update stored state
        if vehicle_id in self.node_states:
            self.node_states[vehicle_id].append(status)
        else:
            self.node_states[vehicle_id] = [status]
        return self.last_block['index'] + 1

    def check_for_attacks(self, current_step, leader_vx):
        """
        Stub for checking attacks or abnormal velocities. 
        Could use WMSR or thresholds to detect anomalies.
        """
        if current_step < 10:
            return
        # If a node's velocity is too far from leader's velocity => suspicious
        for vehicle_id in self.nodes:
            if len(self.node_states[vehicle_id]) == 0:
                continue
            cur_vx = self.node_states[vehicle_id][-1]['vx']
            # Example threshold
            if abs(cur_vx - leader_vx) > 15:
                print(f"Potential attack detected for vehicle {vehicle_id} (vx={cur_vx}).")

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
        return guess_hash[:3] == "000"


###############################################################################
# DynamicBicycleModel Class
###############################################################################
class DynamicBicycleModel:
    def __init__(self, blockchain, id, x=0.0, y=0.0, psi=0.0, vx=10.0, vy=0.0, r=0.0):
        """
        Each vehicle has:
         - self.blockchain: reference to the global Blockchain
         - self.id
         - self.(x, y, psi, v_x, v_y, r)
         - RSA key pair (public_key, private_key)
         - Registration attempt: store public_key in blockchain
        """
        self.blockchain = blockchain
        self.id = id
        self.x = x
        self.y = y
        self.psi = psi
        self.v_x = vx
        self.v_y = vy
        self.r = r

        # Generate RSA keys
        self.public_key, self.private_key = generate_keys()

        # Attempt to register node on the blockchain using initial state + public key
        init_state = self.get_state()
        self.registered = self.blockchain.register_node(self.id, init_state, self.public_key)

        # Physical params
        self.m = 1500
        self.L_f = 1.2
        self.L_r = 1.6
        self.I_z = 2250
        self.C_f = 19000
        self.C_r = 20000
        self.dt = 0.01

    def update(self, a, delta):
        """
        Update the vehicle's state using a bicycle model
        and broadcast the new transaction to the blockchain with RSA signature
        *only* if registered.
        """
        xx = np.array([self.x, self.y, self.psi, self.v_x, self.v_y, self.r])
        k1 = self.f(xx, a, delta)
        k2 = self.f(xx + self.dt/2*k1, a, delta)
        k3 = self.f(xx + self.dt/2*k2, a, delta)
        k4 = self.f(xx + self.dt*k3, a, delta)
        xx = xx + self.dt*(k1 + 2*k2 + 2*k3 + k4)/6

        self.x, self.y, self.psi, self.v_x, self.v_y, self.r = xx

        # Then broadcast updated state only if truly registered
        if self.registered:
            self.broadcast_state()

    def f(self, xx, a, delta):
        x, y, psi, v_x, v_y, r = xx
        alpha_f = delta - np.arctan2((v_y + self.L_f*r), v_x)
        alpha_r = -np.arctan2((v_y - self.L_r*r), v_x)
        F_yf = self.C_f*alpha_f
        F_yr = self.C_r*alpha_r

        x_dot = v_x*np.cos(psi) - v_y*np.sin(psi)
        y_dot = v_x*np.sin(psi) + v_y*np.cos(psi)
        psi_dot = r
        v_x_dot = a - (F_yf*np.sin(delta))/self.m + v_y*r
        v_y_dot = (F_yf*np.cos(delta) + F_yr)/self.m - v_x*r
        r_dot = (self.L_f*F_yf*np.cos(delta) - self.L_r*F_yr)/self.I_z
        return np.array([x_dot, y_dot, psi_dot, v_x_dot, v_y_dot, r_dot])

    def broadcast_state(self):
        """
        Sign the current state with the private key, then add a transaction
        to the blockchain. This ensures authenticity (Sybil-proof).
        """
        state = self.get_state()
        signature = sign_data(state, self.private_key)
        self.blockchain.new_transaction(vehicle_id=self.id, status=state, signature=signature)

    def get_state(self):
        return {
            'x': round(float(self.x), 2),
            'y': round(float(self.y), 2),
            'yaw': round(float(self.psi), 2),
            'vx': round(float(self.v_x), 2),
            'vy': round(float(self.v_y), 2)
        }


###############################################################################
# LeaderFollowerSimulation: Malicious Vehicles Cannot Register
###############################################################################
class LeaderFollowerSimulation:
    def __init__(self, num_followers):
        # 1) Create the Blockchain
        self.blockchain = Blockchain(num_followers)
        
        # 2) Create Leader (registered ID=0)
        self.leader = DynamicBicycleModel(
            blockchain=self.blockchain, 
            id=0, 
            x=100, 
            y=10, 
            vx=6.0
        )

        # 3) Create Followers (registered IDs=1..num_followers)
        self.followers = [
            DynamicBicycleModel(
                blockchain=self.blockchain,
                id=i + 1,
                x=100 - 10 * (i + 1),
                y=10,
                vx=6.0
            ) for i in range(num_followers)
        ]
        self.num_followers = num_followers
        self.desired_gap = 10
        self.dt = 0.05
        self.time_steps = int(50 / self.dt)
        self.road_width = 20

        # 4) Sybil attack (malicious vehicles)
        self.malicious_vehicles = []
        self.num_malicious = 3
        self.malicious_added = False

        # Data storage
        self.ttc_history = []
        self.min_distances = []

    def calculate_ttc(self):
        """
        Calculate Time to Collision (TTC) for all vehicles.
        """
        ttc_values = []
        
        # Leader-to-first-follower TTC
        leader = self.leader
        first_follower = self.followers[0]
        relative_velocity = first_follower.v_x - leader.v_x
        relative_position = leader.x - first_follower.x - self.desired_gap

        if relative_position > 0 and relative_velocity > 0:
            ttc = relative_position / relative_velocity
        else:
            ttc = float('inf')  # No collision risk or invalid scenario

        ttc_values.append(ttc)

        # Follower-to-follower TTC
        for i in range(1, len(self.followers)):
            leader = self.followers[i - 1]
            follower = self.followers[i]
            relative_velocity = follower.v_x - leader.v_x
            relative_position = leader.x - follower.x - self.desired_gap

            if relative_position > 0 and relative_velocity > 0:
                ttc = relative_position / relative_velocity
            else:
                ttc = float('inf')  # No collision risk or invalid scenario

            ttc_values.append(ttc)

        return min(ttc_values)  # Return the minimum TTC across all pairs


    def run_simulation(self):
        # Histories for recognized vehicles (leader + followers only)
        x_history = [[] for _ in range(self.num_followers + 1)]
        y_history = [[] for _ in range(self.num_followers + 1)]
        v_history = [[] for _ in range(self.num_followers + 1)]

        time_points = np.arange(0, self.time_steps) * self.dt

        for t in range(self.time_steps):
            # 1) Possibly create malicious vehicles after t=300
            #    but they fail to register => not recognized
            if t == 300 and not self.malicious_added:
                for i_m in range(self.num_malicious):
                    # Malicious IDs outside [0..num_followers], e.g. 100,101,102
                    m_id = 100 + i_m
                    mv = DynamicBicycleModel(
                        blockchain=self.blockchain,
                        id=m_id,
                        x=60 - 10*(i_m + 1),
                        y=10,
                        vx=20.0
                    )
                    # Because the sign is invalid for these IDs, 
                    # mv.registered is False => they do not join the blockchain.
                    self.malicious_vehicles.append(mv)

                self.malicious_added = False

            # 2) Update Leader
            v_target = 6.0
            a_l = 1.0*(v_target - self.leader.v_x)
            self.leader.update(0, 0.0)
            x_history[0].append(self.leader.x)
            y_history[0].append(self.leader.y)
            v_history[0].append(self.leader.v_x)

            # 3) Update Followers
            min_dist_timestep = float('inf')
            for i, follower in enumerate(self.followers):
                distance_to_leader = self.leader.x - follower.x - self.desired_gap*(i + 1)
                a_f = 1.0 * distance_to_leader
                follower.update(0, 0.0)

                x_history[i+1].append(follower.x)
                y_history[i+1].append(follower.y)
                v_history[i+1].append(follower.v_x)

                # Distances among recognized followers only
                # Dist from follower[i] to the leader or the previous follower
                if i == 0:
                    dist = math.sqrt((follower.x - self.leader.x)**2 + 
                                     (follower.y - self.leader.y)**2)
                else:
                    dist = math.sqrt((follower.x - self.followers[i-1].x)**2 + 
                                     (follower.y - self.followers[i-1].y)**2)
                min_dist_timestep = min(min_dist_timestep, dist)

            # 4) We do NOT update malicious vehicles in the network calculations
            #    Even if they physically do something, we do not store or plot them 
            #    because they're not in the blockchain.

            self.min_distances.append(min_dist_timestep)
            
            # 5) Calculate and store TTC (registered vehicles only)
            min_ttc = self.calculate_ttc()
            self.ttc_history.append(min_ttc)

            # 6) Blockchain checks (consensus, PoW, etc.)
            self.blockchain.check_for_attacks(t, self.leader.v_x)
            proof = self.blockchain.proof_of_work(self.blockchain.last_block)
            self.blockchain.new_block(proof, self.blockchain.hash(self.blockchain.last_block))

        # Once simulation completes, plot recognized vehicles only
        self.plot_simulation(time_points, x_history, y_history, v_history)

    def plot_simulation(self, time_points, x_history, y_history, v_history):
        # --- 1) Positions at snapshots ---
        plt.figure(figsize=(15, 8))
        t_samples = [
            int(self.time_steps * 0.01),
            int(self.time_steps * 0.2),
            int(self.time_steps * 0.4),
            int(self.time_steps * 0.6),
            int(self.time_steps * 0.8),
            int(self.time_steps * 0.99)
        ]
        for idx, sample_idx in enumerate(t_samples):
            plt.subplot(3, 2, idx + 1)
            # Leader + followers
            for i in range(self.num_followers + 1):
                color = '.r' if i == 0 else '.k'
                size = 10 if i == 0 else 5
                plt.plot(x_history[i][sample_idx], y_history[i][sample_idx], color, markersize=size)

            # Road boundaries
            leader_x = max(x_history[0]) if x_history[0] else 100
            plt.plot([0, leader_x + 20], [0, 0], 'b-', linewidth=2)
            plt.plot([0, leader_x + 20], [20, 20], 'b-', linewidth=2)
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title(f't={sample_idx*self.dt:.2f} s')
            plt.ylim(-5, 25)
        plt.tight_layout()
        plt.show()

        # --- 2) Velocities & Min Distance ---
        plt.figure(figsize=(10, 12))
        # Subplot for velocity
        plt.subplot(2,1,1)
        colors = plt.cm.viridis(np.linspace(0,1,self.num_followers+1))
        for i in range(self.num_followers + 1):
            label = 'Leader' if i == 0 else f'Follower {i}'
            plt.plot(time_points, v_history[i], label=label, color=colors[i], linewidth=2.5)

        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        plt.grid(True)
        plt.ylim(0, 15)

        # Subplot for min distance
        plt.subplot(2,1,2)
        plt.plot(time_points, self.min_distances, 'b-', label='Min Distance')
        plt.axhline(y=self.desired_gap, color='r', linestyle='--', label='Desired Gap')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 25)
        plt.tight_layout()
        plt.show()

        # --- 3) TTC Over Time ---
        self.plot_ttc_over_time()
    def plot_ttc_over_time(self):
        """
        Plot the Time to Collision (TTC) over time.
        """
        time_points = np.arange(len(self.ttc_history)) * self.dt
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, self.ttc_history, 'g-', label='Minimum TTC')
        plt.axhline(y=1, color='r', linestyle='--', label='TTC Threshold (1s)')
        plt.axhline(y=2, color='orange', linestyle='--', label='TTC Threshold (2s )')
        plt.xlabel('Time [s]')
        plt.ylabel('TTC [s]')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, time_points[-1])
        plt.tight_layout()
        plt.show()


# -------------------------------------------------------------------
# MAIN RUN
# -------------------------------------------------------------------
if __name__ == "__main__":
    num_followers = 10
    simulation = LeaderFollowerSimulation(num_followers)
    simulation.run_simulation()

    print("\n--- Registered Nodes in Blockchain ---")
    print(simulation.blockchain.nodes)
    print("\n--- Final Blockchain Length ---", len(simulation.blockchain.chain))
