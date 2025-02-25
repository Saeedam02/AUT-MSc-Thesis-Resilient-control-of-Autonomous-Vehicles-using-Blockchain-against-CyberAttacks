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
        if abs(current - consensus_velocity) > tolerance:
            return False
        else:
            return True
        
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
        return guess_hash[:3] == "000"


###############################################################################
# DynamicBicycleModel Class
###############################################################################
class DynamicBicycleModel:
    def __init__(self, blockchain, 
                 x=0.0, 
                 y=0.0, 
                 psi=0.0, 
                 vx=10.0, 
                 vy=0.0, 
                 r=0.0, 
                 dt=0.01,
                 vehicle_id=0):
        self.blockchain = blockchain
        self.id  = vehicle_id  # For logging or debug
        # States
        self.x   = x
        self.y   = y
        self.psi = psi
        self.vx  = vx
        self.vy  = vy
        self.r   = r
        
        # Vehicle parameters
        self.m   = 1500.0    # mass [kg]
        self.L_f = 1.2       # CG to front axle [m]
        self.L_r = 1.6       # CG to rear axle [m]
        self.I_z = 2250.0    # Yaw moment of inertia [kg*m^2]
        self.C_f = 19000.0   # Front cornering stiffness [N/rad]
        self.C_r = 20000.0   # Rear cornering stiffness [N/rad]

        self.dt  = dt  # integration step
        # Generate and store RSA keys for this vehicle
        self.public_key, self.private_key = generate_keys()
        # Register node with the blockchain using initial state + public key
        self.blockchain.register_node(
            vehicle_id=self.id,
            state=self.get_state(),
            public_key=self.public_key
        )
        self.velocity_history = []

    def _state_derivs(self, s, a, delta):
        """
        s     = [x, y, psi, vx, vy, r]
        a     = acceleration [m/s^2]
        delta = steering angle [rad]
        """
        x, y, psi, vx, vy, r = s
        vx_safe = max(vx, 0.1)  # clamp to avoid division by zero

        # Slip angles
        alpha_f = delta - np.arctan2((vy + self.L_f*r), vx_safe)
        alpha_r = - np.arctan2((vy - self.L_r*r), vx_safe)

        # Lateral forces (linear tire model)
        F_yf = self.C_f * alpha_f
        F_yr = self.C_r * alpha_r

        # Global kinematics
        x_dot   = vx*np.cos(psi) - vy*np.sin(psi)
        y_dot   = vx*np.sin(psi) + vy*np.cos(psi)
        psi_dot = r

        # Body-frame acceleration
        vx_dot  = a - (F_yf*np.sin(delta))/self.m + vy*r
        vy_dot  = (F_yf*np.cos(delta) + F_yr)/self.m - vx*r
        r_dot   = (self.L_f*F_yf*np.cos(delta) - self.L_r*F_yr)/self.I_z

        return np.array([x_dot, y_dot, psi_dot, vx_dot, vy_dot, r_dot])

    def update(self, a, delta):
        """
        4th-order Runge-Kutta integration with inputs (a, delta).
        """
        s = np.array([self.x, self.y, self.psi, self.vx, self.vy, self.r])

        k1 = self._state_derivs(s,               a, delta)
        k2 = self._state_derivs(s + 0.5*self.dt*k1, a, delta)
        k3 = self._state_derivs(s + 0.5*self.dt*k2, a, delta)
        k4 = self._state_derivs(s +     self.dt*k3, a, delta)

        s_next = s + (self.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # Unpack
        self.x   = s_next[0]
        self.y   = s_next[1]
        self.psi = s_next[2]
        self.vx  = s_next[3]
        self.vy  = s_next[4]
        self.r   = s_next[5]

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
            'vx': round(float(self.vx), 2),
            'vy': round(float(self.vy), 2),
            'r' : round(float(self.r),2)
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
                    agent.vx = status['vx']
                    self.velocity_history.append(agent.vx)

class InnerLoopController:
    def __init__(self, dt=0.01):
        self.dt = dt

        # Longitudinal Gains (PI)
        self.kp_speed = 1.2
        self.ki_speed = 0.1
        self.speed_error_int = 0.0

        # Lateral Gains: "2-state" feedback on (v_y, r)
        # delta = -K_vy*v_y - K_r*(r - r_ref)  (plus feedforward, if needed)
        self.K_vy   = 0.15
        self.K_r    = 0.08
        self.K_r_ff = 1.0  # feedforward for r_ref

        # Physical saturations
        self.max_steer = 0.5     # rad
        self.min_steer = -0.5
        self.max_accel = 3.0     # m/s^2
        self.min_accel = -5.0

    def control(self, vx, vy, r, vx_ref, r_ref):
        """
        vx, vy, r   : actual states
        vx_ref, r_ref : references
        returns a, delta
        """
        # --- 1) Longitudinal speed control (PI) ---
        speed_err = vx_ref - vx
        self.speed_error_int += speed_err * self.dt

        # Acceleration command
        a_unclamped = (self.kp_speed * speed_err
                       + self.ki_speed * self.speed_error_int)
        a = np.clip(a_unclamped, self.min_accel, self.max_accel)

        # --- 2) Lateral / yaw-rate control ---
        # Simple feedback: delta = -K_vy*v_y - K_r*(r - r_ref)
        delta_unclamped = (
            - self.K_vy * vy
            - self.K_r  * (r - r_ref)
            + self.K_r_ff * r_ref  # optional feedforward
        )
        delta = np.clip(delta_unclamped, self.min_steer, self.max_steer)

        return a, delta

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

# ------------------------------------------------------------------------
# 3) LEADER-FOLLOWER SIMULATION WITH OUTER + INNER LOOP
# ------------------------------------------------------------------------
class LeaderFollowerSimulation:
    def __init__(self, num_followers,  blockchain, dt=0.01, spoof_params=None):
        self.dt = dt
        self.blockchain = blockchain
        self.time_steps = int(100.0 / dt)  # 50s sim, for example
        self.num_followers = num_followers
        self.elapsed_times = []  # List to store elapsed times for each step
        self.spoof_params = spoof_params  # Store spoofing parameters

        # Leader (we can choose to use or skip an inner loop here)
        self.leader = DynamicBicycleModel(self.blockchain, x=100, y=10, vx= 10, dt=dt, vehicle_id=0)
        
        # For simplicity, let's do direct control for leader:
        self.leader_speed_ref = 6.0  # We'll keep it constant
        self.kp_leader = 1.0
        self.road_width = 20
        self.ttc_history = []  # Track minimum TTC over time
        # Follower vehicles, each with its own inner controller
        self.followers = []
        self.controllers = []
        for i in range(num_followers):
            veh = DynamicBicycleModel(self.blockchain, x=100 - 14*(i+1),
                                      y=10,
                                      vx=3*(i+1),
                                      dt=dt,
                                      vehicle_id=i+1)
            self.followers.append(veh)
            ctrl = InnerLoopController(dt=dt)
            self.controllers.append(ctrl)

        # Outer loop platoon gains
        self.desired_gap = 10.0
        self.k_s   = 0.15   # how strongly we correct spacing error -> v_x_ref
        self.k_v   = 0.2   # optionally might do relative speed
        self.k_ey  = 0.6   # lateral offset -> r_ref
        self.k_epsi= 0.6   # heading error -> r_ref

    def calculate_ttc(self):
        """
        Calculate Time to Collision (TTC) for all vehicles.
        """
        ttc_values = []
        
        # Leader-to-first-follower TTC
        leader = self.leader
        first_follower = self.followers[0]
        relative_velocity = first_follower.vx - leader.vx
        relative_position = leader.x - first_follower.x 

        if relative_position > 0 and relative_velocity > 0:
            ttc = relative_position / relative_velocity
        else:
            ttc = float('inf')  # No collision risk or invalid scenario

        ttc_values.append(ttc)

        # Follower-to-follower TTC
        for i in range(1, len(self.followers)):
            leader = self.followers[i - 1]
            follower = self.followers[i]
            relative_velocity = follower.vx - leader.vx
            relative_position = leader.x - follower.x 

            if relative_position > 0 and relative_velocity > 0:
                ttc = relative_position / relative_velocity
            else:
                ttc = float('inf')  # No collision risk or invalid scenario

            ttc_values.append(ttc)

        return min(ttc_values)  # Return the minimum TTC across all pairs

    def run_simulation(self, spoof_params=None):
        # For logging
        x_history = [[] for _ in range(self.num_followers + 1)]
        y_history = [[] for _ in range(self.num_followers + 1)]
        v_history = [[] for _ in range(self.num_followers + 1)]
        min_distances = []
        time_points = np.arange(0, self.time_steps) * self.dt

        spoof_step = -1
        attacker_id = None
        target_id = None
        if spoof_params:
            spoof_step = spoof_params.get('step', -1)
            attacker_id = spoof_params.get('attacker_id', None)
            target_id = spoof_params.get('target_id', None)

        for step in range(self.time_steps):

            #-----------------------------------------------
            # Leader Control (simple speed P-control)
            #-----------------------------------------------
            a_leader = self.kp_leader * (self.leader_speed_ref - self.leader.vx)
            delta_leader = 0.0  # keep steering = 0 for leader

            self.leader.update(a_leader, delta_leader)

            # Save leader state
            x_history[0].append(self.leader.x)
            y_history[0].append(self.leader.y)
            v_history[0].append(self.leader.vx)

            #-----------------------------------------------
            # Follower Vehicles
            #-----------------------------------------------
            min_dist_timestep = float('inf')

            for i, follower in enumerate(self.followers):
                # Outer loop: measure spacing/lateral error w.r.t. the preceding vehicle
                if i == 0:
                    # preceding = leader
                    state0 = self.leader.get_state()
                else:
                    state0 = self.followers[i - 1].get_state()

                state1 = follower.get_state()

                # Extract numerical values from the dictionaries
                x0 = state0['x']
                y0 = state0['y']
                psi0 = state0['yaw']
                vx0 = state0['vx']
                vy0 = state0['vy']
                r0 = state0['r']

                x1 = state1['x']
                y1 = state1['y']
                psi1 = state1['yaw']
                vx1 = state1['vx']
                vy1 = state1['vy']
                r1 = state1['r']

                dx = x1 - x0
                dy = y1 - y0

                cos_psi0 = np.cos(psi0)
                sin_psi0 = np.sin(psi0)
                e_x = cos_psi0 * dx + sin_psi0 * dy
                e_y = -sin_psi0 * dx + cos_psi0 * dy

                # Spacing error: we want the follower to be 'desired_gap' behind
                e_s = -e_x - self.desired_gap  # i.e., if e_x < -10, we are behind

                # Heading error for outer loop
                e_psi = psi0 - psi1

                # Generate references
                v_x_ref = vx0 + self.k_s * e_s + self.k_v * (vx0 - vx1)
                r_ref = r0 + self.k_ey * e_y + self.k_epsi * e_psi

                # Call the follower's inner loop controller
                a_f, delta_f = self.controllers[i].control(
                    vx1, vy1, r1,
                    v_x_ref, r_ref
                )

                # Update follower dynamics
                follower.update(a_f, delta_f)

                # For logging
                x_history[i + 1].append(follower.x)
                y_history[i + 1].append(follower.y)

                # Calculate distance to preceding vehicle
                if i == 0:
                    dist = np.sqrt((follower.x - self.leader.x) ** 2 + (follower.y - self.leader.y) ** 2)
                else:
                    dist = np.sqrt((follower.x - self.followers[i - 1].x) ** 2 +
                                (follower.y - self.followers[i - 1].y) ** 2)
                min_dist_timestep = min(min_dist_timestep, dist)

            # Keep track of min distance among all follower pairs for plotting
            min_distances.append(min_dist_timestep)

            attack_step = self.spoof_params['step']
            # Check for spoofing attack
            if self.spoof_params and attack_step < step < attack_step+10:
                malicious_vehicle = simulation.followers[-1]
                target_id = self.spoof_params['target_id']
                simulate_spoofing_attack(blockchain, malicious_vehicle, target_id)

            # Calculate and store TTC
            min_ttc = self.calculate_ttc()
            self.ttc_history.append(min_ttc)

            # Spoofing checks + block finalization
            start_time = time()
            self.blockchain.check_for_attacks(step, self.leader.vx)
            proof = self.blockchain.proof_of_work(self.blockchain.last_block)
            self.blockchain.new_block(proof, self.blockchain.hash(self.blockchain.last_block))
            end_time = time()
            # Calculate the time taken for the loop to complete
            elapsed_time = end_time - start_time
            self.elapsed_times.append(elapsed_time)  # Store elapsed time

            # Reconcile states from the chain
            for j, f in enumerate(self.followers):
                f.check_and_update_agents_from_blockchain(self.followers)
                v_history[j + 1].append(f.vx)
            self.leader.check_and_update_agents_from_blockchain([self.leader])

        # Plot results
        self.plot_results(x_history, y_history, v_history, min_distances, time_points)
        self.plot_elapsed_times()

        self.plot_ttc_over_time(time_points)

    def plot_ttc_over_time(self,time_points):
        """
        Plot the Time to Collision (TTC) over time.
        """
        time_points = np.arange(len(self.ttc_history)) * self.dt  # Create time points based on the length of ttc_history
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, self.ttc_history, 'g-', label='Minimum TTC')  # Use time_points for x-axis
        plt.axhline(y=1, color='r', linestyle='--', label='TTC Threshold (1s)')
        plt.axhline(y=2, color='orange', linestyle='--', label='TTC Threshold (2s)')
        plt.xlabel('Time [s]')
        plt.ylabel('TTC [s]')
        plt.ylim(-1,10)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_elapsed_times(self):
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(self.elapsed_times)), self.elapsed_times, color='blue')
        plt.xlabel('Simulation Step')
        plt.ylabel('Elapsed Time (seconds)')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_results(self, x_history, y_history, v_history, min_distances, time_points):
        # 1) Trajectory 
        plt.figure(figsize=(15, 8))
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

        # 2) Combine Velocity and Minimum Distance plots
        plt.figure(figsize=(10, 12))

        # Subplot for Velocity Consensus
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
        colors = plt.cm.viridis(np.linspace(0, 1, self.num_followers + 1))  # Use viridis colormap
        for i in range(self.num_followers + 1):
            label = 'Leader' if i == 0 else f'Follower {i}'
            plt.plot(time_points, v_history[i], label=label, color=colors[i], linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.ylim(0, 15)

        # Subplot for Minimum Distance
        plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
        plt.plot(time_points, min_distances, 'b-', label='Min Distance')
        plt.axhline(y=self.desired_gap, color='r', linestyle='--', label='Desired Gap')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.legend()
        plt.grid(True)
        plt.ylim(5, 15)

        plt.tight_layout()
        plt.show()

###############################################################################
# MAIN EXECUTION
###############################################################################
if __name__ == "__main__":
    num_followers = 5
    blockchain = Blockchain(num_followers)
    spoof_params = {
        'step': 500,
        'attacker_id':2,
        'target_id': 1
    }
    simulation = LeaderFollowerSimulation(num_followers=num_followers, blockchain=blockchain, spoof_params=spoof_params)
    simulation.run_simulation()

    # Print the final chain or check logs to see the spoofing rejection
    print("\n--- Registered Nodes ---", blockchain.nodes)