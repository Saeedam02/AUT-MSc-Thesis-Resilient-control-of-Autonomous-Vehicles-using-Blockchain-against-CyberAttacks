import hashlib
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from time import time
import rsa 
###############################################################################
# Generate RSA Key Pairs for Each Vehicle
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

        # We still keep 'sings' from the original code, but we won't rely on it for security
        # Instead, we store actual RSA public keys in:
        self.public_keys = {}   # vehicle_id -> RSA public key

        # This part is from your FDI code
        self.sings = [self.get_sign(i) for i in range(num_agents + 1)]
        self.node_states = {}  # Store the state history of each node

        # Create the genesis block
        self.new_block(previous_hash='1', proof=100)

    def register_node(self, vehicle_id, state, public_key):
        """
        Add a new node (vehicle) to the list of nodes, associating it with its RSA public_key.
        """
        # We do the old signature check for consistency with your code
        sign = self.get_sign(vehicle_id)
        if vehicle_id in self.nodes:
            print(f"The same ID {vehicle_id} is registered in the blockchain, try new one")
        else:
            if sign in self.sings:
                self.nodes.add(vehicle_id)
                self.node_states[vehicle_id] = [state]
                self.public_keys[vehicle_id] = public_key  # store the RSA public key
                print(f"Vehicle {vehicle_id} registered successfully with a public key.")
            else:
                print(f"Registration failed for vehicle {vehicle_id}: Invalid signature.")
                return

    def register_additional_nodes(self, vehicle_id, state, public_key):
        """
        Manually register a new node (vehicle) with its public key.
        """
        sign = self.get_sign(vehicle_id)
        self.sings.append(sign)
        if vehicle_id in self.nodes:
            print(f"The same ID {vehicle_id} is already registered in the blockchain, try a new one.")
        else:
            self.nodes.add(vehicle_id)
            self.node_states[vehicle_id] = [state]
            self.public_keys[vehicle_id] = public_key
            print(f"Vehicle {vehicle_id} registered successfully (additional).")

    def get_sign(self, vehicle_id):
        """
        Original simplistic sign from your code (not cryptographically secure).
        We still keep it for backward compatibility in your code logic.
        """
        return hashlib.sha256(str(vehicle_id).encode()).hexdigest()[:8]

    def valid_chain(self, chain):
        """
        Original validity check from your code
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

    def new_transaction(self, vehicle_id, status, signature=None):
        """
        Creates a new transaction to go into the next mined Block,
        verifying RSA signature if provided.
        """
        # Ensure node is recognized
        if vehicle_id not in self.nodes:
            print(f"Unregistered vehicle {vehicle_id} tried a transaction.")
            return self.last_block['index'] + 1

        # If an RSA signature is provided, verify it
        if signature is not None:
            # Retrieve the stored public key
            pub_key = self.public_keys.get(vehicle_id, None)
            if pub_key:
                try:
                    verify_signature(status, signature, pub_key)
                except rsa.VerificationError:
                    print(f"[Spoofing] Signature invalid for vehicle {vehicle_id}. Transaction rejected.")
                    return self.last_block['index'] + 1
            else:
                print(f"No public key found for vehicle {vehicle_id}.")
                return self.last_block['index'] + 1

        # If verification is ok or no signature provided, add the transaction
        self.current_transactions.append({
            'vehicle_id': vehicle_id,
            'status': status,
        })

        # Update the state history
        if vehicle_id in self.node_states:
            self.node_states[vehicle_id].append(status)
        else:
            self.node_states[vehicle_id] = [status]

        return self.last_block['index'] + 1

    def check_for_attacks(self, current_step, v_l):
        """
        Same WMSR-based or velocity-based check for FDI
        """
        if current_step < 10:
            pass
        else:
            for vehicle_id in self.nodes:
                consensus_velocity = v_l
                # Compare with the current state
                if len(self.node_states[vehicle_id]) > 0:
                    current_velocity = {'vx': self.node_states[vehicle_id][-1]['vx']}
                    if not self.is_consistent(current_velocity, consensus_velocity):
                        print(f"Potential attack detected for vehicle {vehicle_id}.")
                        self.calc_new_velocity(vehicle_id, consensus_velocity)
                        print('done')

    def calc_new_velocity(self, vehicle_id, consensus_velocity):
        """
        Overwrite the velocity in the current transactions if it deviates too much.
        """
        for vehicle in self.current_transactions:
            if vehicle['vehicle_id'] == vehicle_id:
                value = vehicle['status']['vx']
                vehicle['status']['vx'] = consensus_velocity
                print('Velocity of vehicle', vehicle_id, 'with value of:', value, 'updated to:', consensus_velocity)

    def wmsr(self, neighbors_velocities, f):
        """
        WMSR algorithm (placeholder) from your code
        """
        if len(neighbors_velocities) == 0:
            return None

        values = [d['vx'] for d in neighbors_velocities]
        values.sort()
        lenlist = len(values)
        if len(values) > 2 * f:
            values = values[f:-f]
            # Could compute mean, etc.

        consensus_velocity = round(sum(values) / lenlist, 2) if len(values) > 0 else 0
        return consensus_velocity

    def is_consistent(self, current_velocity, consensus_velocity):
        """
        Checks if the node's velocity is within a certain tolerance.
        """
        current = current_velocity['vx']
        tolerance = 5
        return abs(current - consensus_velocity) <= tolerance

    @property
    def last_block(self):
        return self.chain[-1]

    @staticmethod
    def hash(block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def proof_of_work(self, last_block):
        """
        Same PoW logic from your code
        """
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
        return guess_hash[:2] == "00"


###############################################################################
# DynamicBicycleModel Class
###############################################################################
class DynamicBicycleModel:
    def __init__(self, blockchain, id, x=0.0, y=0.0, psi=0.0, vx=10.0, vy=0.0, r=0.0):
        self.blockchain = blockchain
        self.id = id
        self.x = x
        self.y = y
        self.psi = psi
        self.v_x = vx
        self.v_y = vy
        self.r = r

        # Generate and store RSA keys for this vehicle
        self.public_key, self.private_key = generate_keys()

        # Register node with the blockchain using initial state + public key
        self.blockchain.register_node(
            vehicle_id=self.id,
            state=self.get_state(),
            public_key=self.public_key
        )

        self.velocity_history = []
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
        Update the vehicle's state using the bicycle model
        and broadcast the new transaction with RSA signature.
        """
        xx = np.array([self.x, self.y, self.psi, self.v_x, self.v_y, self.r])
        k1 = self.f(xx, a, delta)
        k2 = self.f(xx + self.dt/2*k1, a, delta)
        k3 = self.f(xx + self.dt/2*k2, a, delta)
        k4 = self.f(xx + self.dt*k3, a, delta)
        xx = xx + self.dt*(k1 + 2*k2 + 2*k3 + k4)/6

        self.x, self.y, self.psi, self.v_x, self.v_y, self.r = xx
        # Then broadcast the updated state
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
        v_y_dot = (F_yf*np.cos(delta)+ F_yr)/self.m - v_x*r
        r_dot = (self.L_f*F_yf*np.cos(delta) - self.L_r*F_yr)/self.I_z
        return np.array([x_dot, y_dot, psi_dot, v_x_dot, v_y_dot, r_dot])

    def broadcast_state(self):
        """
        Sign the current state with the private key, then call new_transaction
        so the blockchain can verify the signature with the stored public key.
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

    def check_and_update_agents_from_blockchain(self, agents):
        """
        Similar to your code: read the last block and reconcile states
        """
        last_block = self.blockchain.chain[-1]
        for transaction in last_block['transactions']:
            v_id = transaction['vehicle_id']
            status = transaction['status']
            for agent in agents:
                if agent.id == v_id:
                    agent.x = status['x']
                    agent.v_x = status['vx']
                    self.velocity_history.append(agent.v_x)


###############################################################################
# Spoofing Attack
###############################################################################
def simulate_spoofing_attack(blockchain, malicious_vehicle, spoofed_id):
    """
    Malicious tries to submit a transaction claiming 'vehicle_id=spoofed_id',
    but signs with malicious_vehicle's private key. The blockchain should detect mismatch.
    """
    fake_state = {'x': 9999, 'y': 9999, 'yaw': 0, 'vx': 50.0, 'vy': 0}
    # Sign the fake state with the malicious vehicle's private key
    # But we pass 'vehicle_id=spoofed_id' to attempt impersonation
    signature = sign_data(fake_state, malicious_vehicle.private_key)

    print(f"\n--- Spoofing Attack! Vehicle {malicious_vehicle.id} tries to impersonate {spoofed_id} ---")
    # Attempt to add the transaction to the blockchain
    blockchain.new_transaction(vehicle_id=spoofed_id, status=fake_state, signature=signature)

 
    # Attempt to add the transaction to the blockchain
    blockchain.new_transaction(vehicle_id=spoofed_id, status=fake_state, signature=signature)

    # Check if the last transaction was accepted or rejected
    if blockchain.current_transactions:
        if blockchain.current_transactions[-1]['vehicle_id'] == spoofed_id:
            print(f"Transaction for vehicle {spoofed_id} accepted (this should not happen).")
        else:
            print(f"Transaction for vehicle {spoofed_id} rejected due to invalid signature.")
    else:
        print(f"No transactions were added to the blockchain due to rejection.")

###############################################################################
# Leader-Follower Simulation
###############################################################################
class LeaderFollowerSimulation:
    def __init__(self, num_followers, blockchain, spoof_params=None):
        self.blockchain = blockchain
        self.num_followers = num_followers
        self.elapsed_times = []  # List to store elapsed times for each step
        self.spoof_params = spoof_params  # Store spoofing parameters

        # Initialize leader
        self.leader = DynamicBicycleModel(self.blockchain, id=0, x=100, y=10, vx=6.0)

        # Initialize followers
        self.followers = [
            DynamicBicycleModel(
                self.blockchain,
                id=i+1,
                x=100 - 10*(i+1),
                y=10,
                vx=6.0
            )
            for i in range(num_followers)
        ]
        self.dt = 0.05
        self.time_steps = int(50 / self.dt)
        self.desired_gap = 10
        self.road_width = 20

    def run_simulation(self):
        x_history = [[] for _ in range(self.num_followers+1)]
        y_history = [[] for _ in range(self.num_followers+1)]
        v_history = [[] for _ in range(self.num_followers+1)]
        min_distances = []
        time_points = np.arange(0, self.time_steps)*self.dt

        for t in range(self.time_steps):
            # Leader update
            v_target = 6.0
            kp = 1.0
            a_l = kp*(v_target - self.leader.v_x)
            self.leader.update(a_l, 0)

            # Save leader state
            x_history[0].append(self.leader.x)
            y_history[0].append(self.leader.y)
            v_history[0].append(self.leader.v_x)

            min_dist_timestep = float('inf')
            # Update followers
            for i, follower in enumerate(self.followers):
                distance_to_leader = self.leader.x - follower.x - self.desired_gap*(i+1)
                a_f = 1.0 * distance_to_leader
                follower.update(a_f, 0)

                x_history[i+1].append(follower.x)
                y_history[i+1].append(follower.y)
                # we push follower vx in run once we've updated from chain
                # but let's keep it simple here

                if i == 0:
                    dist = np.sqrt((follower.x - self.leader.x)**2 + (follower.y - self.leader.y)**2)
                else:
                    dist = np.sqrt((follower.x - self.followers[i-1].x)**2 + 
                                   (follower.y - self.followers[i-1].y)**2)
                min_dist_timestep = min(min_dist_timestep, dist)

            min_distances.append(min_dist_timestep)
            attack_step = self.spoof_params['step']
            # Check for spoofing attack
            if self.spoof_params and attack_step < t < attack_step+10:
                malicious_vehicle = simulation.followers[-1]
                target_id = self.spoof_params['target_id']
                simulate_spoofing_attack(blockchain, malicious_vehicle, target_id)

            # Spoofing checks + block finalization
            start_time = time()

            blockchain.check_for_attacks(t, self.leader.v_x)
            proof = blockchain.proof_of_work(blockchain.last_block)
            blockchain.new_block(proof, blockchain.hash(blockchain.last_block))
            end_time = time()
            # Calculate the time taken for the loop to complete
            elapsed_time = end_time - start_time
            self.elapsed_times.append(elapsed_time)  # Store elapsed time

            # print(f"Total simulation time_delay: {elapsed_time:.6f} seconds")
            # print('************************')
            # Reconcile states from the chain
            for j, f in enumerate(self.followers):
                f.check_and_update_agents_from_blockchain(self.followers)
                v_history[j+1].append(f.v_x)
            self.leader.check_and_update_agents_from_blockchain([self.leader])

        # Plot results
        self.plot_results(x_history, y_history, v_history, min_distances, time_points)
        self.plot_elapsed_times()

    def plot_elapsed_times(self):
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(self.elapsed_times)), self.elapsed_times, color='blue')
        plt.xlabel('Simulation Step')
        plt.ylabel('Elapsed Time (seconds)')
        plt.title('Elapsed Time per Simulation Step')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_results(self, x_history, y_history, v_history, min_distances, time_points):
        # 1) Trajectory snapshots
        plt.figure(figsize=(15, 8))
        plt.suptitle('Vehicle Platooning Trajectory Snapshots', fontsize=14)
        t_samples = [int(self.time_steps*0.01), int(self.time_steps*0.2), int(self.time_steps*0.4),
                     int(self.time_steps*0.6), int(self.time_steps*0.8), int(self.time_steps*0.99)]
        for idx, t in enumerate(t_samples):
            plt.subplot(3, 2, idx + 1)
            for i in range(self.num_followers+1):
                if i == 0:
                    plt.plot(x_history[i][t], y_history[i][t], '.r', markersize=10)
                else:
                    plt.plot(x_history[i][t], y_history[i][t], '.k', markersize=5)
            leader_x = max(x_history[0]) if x_history[0] else 100
            plt.plot([0, leader_x + 20], [0, 0], 'b-', linewidth=2)
            plt.plot([0, leader_x + 20], [self.road_width, self.road_width], 'b-', linewidth=2)
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title(f't={t*self.dt:.2f}s')
            plt.ylim(-5, self.road_width + 5)
        plt.tight_layout()
        plt.show()

        # 2) We can also plot velocity over time in a simpler manner
        # or we gather them from blockchain states if we want a more advanced approach.
        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set1(np.linspace(0, 1, self.num_followers+1))
        for i in range(self.num_followers+1):
            label = 'Leader' if i == 0 else f'Follower {i}'
            plt.plot(time_points, v_history[i], label=label, color=colors[i], linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Velocity of All Vehicles Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.ylim(0, 15)
        plt.tight_layout()
        plt.show()

        # 3) Min distances
        plt.figure(figsize=(10,6))
        plt.plot(time_points, min_distances, 'b-', label='Min Distance')
        plt.axhline(y=self.desired_gap, color='r', linestyle='--', label='Desired Gap')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.title('Minimum Inter-Vehicle Distance Over Time')
        plt.legend()
        plt.grid(True)
        plt.ylim(5, 15)
        plt.tight_layout()
        plt.show()


###############################################################################
# MAIN EXECUTION
###############################################################################
if __name__ == "__main__":
    num_followers = 10
    blockchain = Blockchain(num_followers)
    spoof_params = {'target_id': 1, 'step': 30}
    simulation = LeaderFollowerSimulation(num_followers=num_followers, blockchain=blockchain, spoof_params=spoof_params)
    simulation.run_simulation()

    # Print the final chain or check logs to see the spoofing rejection
    print("\n--- Registered Nodes ---", blockchain.nodes)
