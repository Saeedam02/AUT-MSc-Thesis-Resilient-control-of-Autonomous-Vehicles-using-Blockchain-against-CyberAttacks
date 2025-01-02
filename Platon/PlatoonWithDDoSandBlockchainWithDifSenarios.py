import numpy as np
import matplotlib.pyplot as plt
import random
from time import time, sleep
from collections import deque
import logging
from datetime import datetime
import hashlib
import json
# Additional import for creating the final table
import pandas as pd

##################################################
# Logging Configuration
##################################################
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File Handler
file_handler = logging.FileHandler(
    f'platooning_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Stream (Console) Handler
stream_handler = logging.StreamHandler()
stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(stream_formatter)

# Add Handlers to the Logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

##################################################
# Security Monitor (for queue-based DDoS detection)
##################################################
class SecurityMonitor:
    def __init__(self):
        self.message_rates = {}
        self.velocity_changes = {}
        self.window_size = 50  # Time window for rate monitoring
        self.global_anomaly_count = 0  # track total anomalies

    def track_message_rate(self, vehicle_id, current_queue_size):
        if vehicle_id not in self.message_rates:
            self.message_rates[vehicle_id] = deque(maxlen=self.window_size)
        self.message_rates[vehicle_id].append(current_queue_size)

    def track_velocity_change(self, vehicle_id, velocity):
        if vehicle_id not in self.velocity_changes:
            self.velocity_changes[vehicle_id] = deque(maxlen=self.window_size)
        self.velocity_changes[vehicle_id].append(velocity)

    def detect_anomalies(self, vehicle_id):
        anomalies = []

        # Check for high message rate anomalies
        if len(self.message_rates.get(vehicle_id, [])) >= 2:
            rate = np.mean(list(self.message_rates[vehicle_id]))
            # Example threshold for high message rate
            if rate > 100:
                anomalies.append(f"High message rate: {rate:.2f}")

        # Check for sudden velocity change anomalies
        if len(self.velocity_changes.get(vehicle_id, [])) >= 2:
            velocities = list(self.velocity_changes[vehicle_id])
            velocity_change = abs(velocities[-1] - velocities[-2])
            if velocity_change > 20:
                anomalies.append(f"Suspicious velocity change: {velocity_change:.2f}")

        # If any anomalies found, increment global counter
        if anomalies:
            self.global_anomaly_count += len(anomalies)

        return anomalies



###############################################################################
# Blockchain Class (with DDoS transaction-rate limiting)
###############################################################################
class Blockchain:
    def __init__(self, num_agents):
        """
        The blockchain tracks transactions (vehicle states).
        It also enforces a transaction rate limit to mitigate DDoS.
        """
        self.current_transactions = []
        self.chain = []
        self.nodes = set()

        # A signature list for valid vehicle IDs (simple demonstration)
        self.sings = [self.get_sign(i) for i in range(num_agents + 1)]
        self.node_states = {}  # Store the on-chain state history of each node

        # Transaction rate-limiting data
        self.transaction_counts = {}
        self.MAX_TX_PER_BLOCK = 5  # Maximum transactions per node per block
        self.suspicious_nodes = set()

        # Create the genesis block
        self.new_block(previous_hash='1', proof=100)

    def register_node(self, vehicle_id, state):
        """
        Add a new node (vehicle) if it has a valid signature.
        """
        sign = self.get_sign(vehicle_id)
        if vehicle_id in self.nodes:
            logging.info(f"[Blockchain] Vehicle {vehicle_id} already registered.")
        else:
            if sign in self.sings:
                self.nodes.add(vehicle_id)
                self.node_states[vehicle_id] = [state]
                logging.info(f"[Blockchain] Vehicle {vehicle_id} registered successfully.")
            else:
                logging.warning(f"[Blockchain] Registration failed: Invalid signature for {vehicle_id}.")

    def flag_node_as_suspicious(self, vehicle_id):
        """
        Mark a node as suspicious; remove it from the blockchain network.
        """
        self.suspicious_nodes.add(vehicle_id)
        if vehicle_id in self.nodes:
            self.nodes.remove(vehicle_id)
        logging.warning(f"[Blockchain] Node {vehicle_id} flagged and removed from the network.")

    def new_transaction(self, vehicle_id, status):
        """
        Creates a new transaction if within rate limit.
        """
        if vehicle_id not in self.nodes:
            logging.warning(f"[Blockchain] Vehicle {vehicle_id} not recognized.")
            return None

        # If node is already suspicious, reject it
        if vehicle_id in self.suspicious_nodes:
            logging.warning(f"[Blockchain] Vehicle {vehicle_id} blocked (suspicious).")
            return None

        # Check or initialize transaction count
        if vehicle_id not in self.transaction_counts:
            self.transaction_counts[vehicle_id] = 0

        # Rate limit check
        if self.transaction_counts[vehicle_id] >= self.MAX_TX_PER_BLOCK:
            logging.warning(
                f"[DDoS] Vehicle {vehicle_id} exceeded TX limit ({self.MAX_TX_PER_BLOCK})."
            )
            self.flag_node_as_suspicious(vehicle_id)
            return None

        # Otherwise accept transaction
        self.current_transactions.append({
            'vehicle_id': vehicle_id,
            'status': status,
        })
        # Update node state
        if vehicle_id not in self.node_states:
            self.node_states[vehicle_id] = []
        self.node_states[vehicle_id].append(status)

        self.transaction_counts[vehicle_id] += 1
        return self.last_block['index'] + 1

    def new_block(self, proof, previous_hash=None):
        """
        Creates a new block, resets transaction counts.
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

        # Reset transaction counts for the next block
        self.transaction_counts = {}

        return block

    def check_for_attacks(self, current_step, v_l):
        """
        FDI-like check: if a node’s velocity is too far from consensus, fix or flag it.
        For brevity, only a simple check is shown here.
        """
        if current_step < 10:
            return
        for vehicle_id in list(self.nodes):
            # Trivial approach: if velocity is too far from leader velocity, we "fix" it
            if len(self.node_states[vehicle_id]) > 0:
                current_vx = self.node_states[vehicle_id][-1]['vx']
                if abs(current_vx - v_l) > 0.01:
                    logging.warning(f"[FDI] Potential anomaly for vehicle {vehicle_id}. Resetting velocity.")
                    self.calc_new_velocity(vehicle_id, v_l)

    def calc_new_velocity(self, vehicle_id, consensus_velocity):
        """
        Overwrite the last transaction for the suspicious vehicle with a corrected velocity
        """
        for txn in reversed(self.current_transactions):
            if txn['vehicle_id'] == vehicle_id:
                old_vx = txn['status']['vx']
                txn['status']['vx'] = consensus_velocity
                logging.info(
                    f"[FDI] Velocity of {vehicle_id} changed from {old_vx} to {consensus_velocity}"
                )
                break

    @staticmethod
    def hash(block):
        """
        SHA-256 hash of a block
        """
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @property
    def last_block(self):
        return self.chain[-1]

    def proof_of_work(self, last_block):
        """
        Simple PoW: find a number 'proof' so that hash(last_proof, proof, last_hash) starts with '00'.
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

    def get_sign(self, vehicle_id):
        return hashlib.sha256(str(vehicle_id).encode()).hexdigest()[:8]
##################################################
# Attacker Class (multiple sources for DDoS)
##################################################
class Attacker:
    """
    An attacker with a specific ID. Each attacker can generate flood messages
    to perform the attack from multiple sources.
    """
    def __init__(self, attacker_id):
        self.id = attacker_id

    def generate_attack_traffic(self, intensity):
        """
        Generate a large number of messages (intensity) for flooding.
        For identification, include attacker_id in the message.
        """
        flood_messages = [
            {'timestamp': time(), 'type': 'status_update', 'attacker_id': self.id}
            for _ in range(intensity)
        ]
        return flood_messages


##################################################
# Dynamic Bicycle Model + Security Additions
##################################################
class DynamicBicycleModel:
    """
    Dynamic bicycle model with a message queue and security monitor
    to simulate DoS/DDoS attacks.
    """
    def __init__(self, blockchain,id, x=0.0, y=0.0, psi=0.0, vx=6.0, vy=0.0, r=0.0):
        self.id = id
        # States
        self.x = x
        self.y = y
        self.psi = psi
        self.v_x = vx
        self.v_y = vy
        self.r = r
        self.blockchain = blockchain

        # Message queue (limited capacity)
        self.message_queue = deque(maxlen=1000)
        self.processing_delay = 0.001
        self.last_update_time = time()

        # Flags for anomaly and recovery
        self.recovery_mode = False

        # Vehicle parameters
        self.m = 1500
        self.L_f = 1.2
        self.L_r = 1.6
        self.I_z = 2250
        self.C_f = 19000
        self.C_r = 20000
        self.dt = 0.01

        # For plotting
        self.velocity_history = []
        # Register the vehicle as a node in the blockchain
        self.blockchain.register_node(self.id, self.get_state())

    def update_dynamics(self, security_monitor, simulation):
        """
        - Process messages in the queue within a 0.1s budget
        - Detect anomalies via SecurityMonitor
        - Possibly switch to recovery mode
        - Also update global counters for processed messages
        """
        current_time = time()
        time_delta = current_time - self.last_update_time
        self.last_update_time = current_time

        # Process messages from the queue
        start_time = time()
        while self.message_queue and (time() - start_time) < 0.1:
            msg = self.message_queue.popleft()
            # Count that we've processed one message
            simulation.total_messages_processed += 1
            sleep(self.processing_delay)

        # Update Security Monitor
        queue_size = len(self.message_queue)
        security_monitor.track_message_rate(self.id, queue_size)
        security_monitor.track_velocity_change(self.id, self.v_x)

        anomalies = security_monitor.detect_anomalies(self.id)
        if anomalies:
            logging.warning(f"Vehicle {self.id} - Anomalies detected: {anomalies}")
            print(
                f"🚨 [Anomaly Detected] Vehicle ID: {self.id} has anomalies: {', '.join(anomalies)}. "
                f"Entering Recovery Mode."
            )
            self.recovery_mode = True

        # If in recovery mode, reduce speed
        if self.recovery_mode:
            self.apply_recovery_control()

        self.velocity_history.append(self.v_x)

    def apply_recovery_control(self):
        """
        Simplest recovery: reduce speed by 5%
        """
        self.v_x *= 0.95

    def update(self, a, delta):
        """
        4th-order Runge-Kutta integration of the bicycle model.
        """
        xx = np.array([self.x, self.y, self.psi, self.v_x, self.v_y, self.r])

        k1 = self.f(xx, a, delta)
        k2 = self.f(xx + self.dt/2 * k1, a, delta)
        k3 = self.f(xx + self.dt/2 * k2, a, delta)
        k4 = self.f(xx + self.dt * k3, a, delta)

        xx = xx + self.dt * (k1 + 2*k2 + 2*k3 + k4)/6

        self.x = xx[0]
        self.y = xx[1]
        self.psi = xx[2]
        self.v_x = xx[3]
        self.v_y = xx[4]
        self.r = xx[5]
        
        # Finally, broadcast new transaction to the blockchain
        self.broadcast_state()

    def broadcast_state(self):
        """
        Publish (vehicle_id, state) to the blockchain as a transaction.
        """
        state = self.get_state()
        blockchain.new_transaction(self.id, state)


    def f(self, xx, a, delta):
        """
        State derivatives (bicycle model).
        """
        x, y, psi, v_x, v_y, r = xx
        alpha_f = delta - np.arctan2((v_y + self.L_f*r), v_x)
        alpha_r = -np.arctan2((v_y - self.L_r*r), v_x)

        F_yf = self.C_f * alpha_f
        F_yr = self.C_r * alpha_r

        x_dot = v_x*np.cos(psi) - v_y*np.sin(psi)
        y_dot = v_x*np.sin(psi) + v_y*np.cos(psi)
        psi_dot = r
        v_x_dot = a - (F_yf*np.sin(delta))/self.m + v_y*r
        v_y_dot = (F_yf*np.cos(delta) + F_yr)/self.m - v_x*r
        r_dot = (self.L_f*F_yf*np.cos(delta) - self.L_r*F_yr)/self.I_z

        return np.array([x_dot, y_dot, psi_dot, v_x_dot, v_y_dot, r_dot])

    def get_state(self):
        return {
            'x': round(float(self.x), 2),
            'y': round(float(self.y), 2),
            'yaw': round(float(self.psi), 2),
            'vx': round(float(self.v_x), 2),
            'vy': round(float(self.v_y), 2)
        }

    def check_and_update_from_blockchain(self, agents):
        """
        After mining, vehicles can reconcile their local states with the final blockchain state
        if desired (optional).
        """
        last_block = self.blockchain.chain[-1]
        for txn in last_block['transactions']:
            veh_id = txn['vehicle_id']
            status = txn['status']
            for ag in agents:
                if ag.id == veh_id:
                    # e.g., unify x, vx from the chain
                    ag.x = status['x']
                    ag.v_x = status['vx']

##################################################
# Distributed Attack Simulation (DDoS)
##################################################
def simulate_distributed_attack(
    attackers,
    vehicles,
    target_ids,
    intensity=1000,
    attack_type="flood",
    simulation=None
):
    """
    For each attacker, generate flood traffic for the target vehicles.
    We also increment the simulation's total_messages_sent counter.
    """
    if attack_type != "flood":
        logging.warning(f"Attack type '{attack_type}' not supported.")
        return

    for attacker in attackers:
        for vehicle in vehicles:
            if vehicle.id in target_ids:
                flood_messages = attacker.generate_attack_traffic(intensity)
                # Increment total messages SENT
                if simulation is not None:
                    simulation.total_messages_sent += len(flood_messages)

                # Put them in the vehicle's queue
                vehicle.message_queue.extend(flood_messages)

                logging.info(
                    f"[DDoS] Attacker {attacker.id} -> Vehicle {vehicle.id}: "
                    f"Flooded {intensity} messages. Queue size={len(vehicle.message_queue)}"
                )
                print(
                    f"⚠️ [DDoS Attack] Attacker ID:{attacker.id} -> Vehicle ID:{vehicle.id}, "
                    f"{intensity} messages. QueueSize={len(vehicle.message_queue)}"
                )


##################################################
# Leader-Follower (Platooning) Simulation
##################################################
class LeaderFollowerSimulation:
    def __init__(self, num_followers,blockchain):
        self.blockchain = blockchain

        # Vehicles
        self.leader = DynamicBicycleModel(blockchain=self.blockchain,id=0, x=100, y=10, vx=6.0)
        self.followers = [
            DynamicBicycleModel(blockchain=self.blockchain,id=i+1, x=100 - 10*(i+1), y=10, vx=6.0)
            for i in range(num_followers)
        ]
        self.vehicles = [self.leader] + self.followers

        # Default attackers
        self.attackers = [Attacker(attacker_id=1001), Attacker(attacker_id=1002)]
        self.max_delay = 0  # Initialize max delay
        self.num_followers = num_followers
        self.desired_gap = 10
        self.dt = 0.05
        self.time_steps = int(50 / self.dt)
        self.road_width = 20

        self.x_history = [[] for _ in range(num_followers + 1)]
        self.y_history = [[] for _ in range(num_followers + 1)]
        self.v_history = [[] for _ in range(num_followers + 1)]
        self.min_distances = []
        self.time_points = np.arange(0, self.time_steps)*self.dt
        self.ttc_history = []
        self.total_attack_delay_times = []
        self.elapsed_times = []
        # == NEW: track message stats to compute packet loss
        self.total_messages_sent = 0
        self.total_messages_processed = 0

        # Shared Security Monitor
        self.security_monitor = SecurityMonitor()

    def calculate_ttc(self):
        ttc_values = []

        # Leader -> first follower
        if self.followers:
            first_follower = self.followers[0]
            rel_vel = first_follower.v_x - self.leader.v_x
            rel_pos = self.leader.x - first_follower.x - self.desired_gap
            if rel_pos > 0 and rel_vel > 0:
                ttc = rel_pos / rel_vel
            else:
                ttc = float('inf')
            ttc_values.append(ttc)

        # Follower -> follower
        for i in range(1, len(self.followers)):
            front = self.followers[i-1]
            rear  = self.followers[i]
            rel_vel = rear.v_x - front.v_x
            rel_pos = front.x - rear.x - self.desired_gap
            if rel_pos > 0 and rel_vel > 0:
                ttc = rel_pos / rel_vel
            else:
                ttc = float('inf')
            ttc_values.append(ttc)

        return min(ttc_values) if ttc_values else float('inf')

    def run_simulation(self, attack_params=None):
        for t_step in range(self.time_steps):
            logging.info(f"\n--- Simulation Step {t_step} ---")
            start_time = time()

            # Possibly launch the DDoS
            if attack_params:
                start_step = attack_params.get('start', 0)
                end_step = attack_params.get('end', 0)
                if start_step <= t_step <= end_step:
                    simulate_distributed_attack(
                        attackers=self.attackers,
                        vehicles=self.vehicles,
                        target_ids=attack_params['targets'],
                        intensity=attack_params.get('intensity', 500),
                        attack_type=attack_params.get('type', 'flood'),
                        simulation=self  # pass reference for stats
                    )

            end_time = time()
            delay = end_time - start_time
            self.total_attack_delay_times.append(delay)
            # Update max delay if the current delay is greater
            if delay > self.max_delay:
                self.max_delay = delay
            # 1) Leader updates
            v_target = 6.0
            k_p = 1.0
            a_l = k_p*(v_target - self.leader.v_x)
            self.leader.update_dynamics(self.security_monitor, simulation=self)
            self.leader.update(0, 0)

            # 2) Record leader state
            self.x_history[0].append(self.leader.x)
            self.y_history[0].append(self.leader.y)

            # 3) Followers
            min_dist_timestep = float('inf')
            for i, follower in enumerate(self.followers):
                dist_to_leader = self.leader.x - follower.x - self.desired_gap*(i+1)
                a_f = 1.0 * dist_to_leader  # naive P-control

                follower.update_dynamics(self.security_monitor, simulation=self)
                follower.update(0, 0)

                self.x_history[i+1].append(follower.x)
                self.y_history[i+1].append(follower.y)


                # Distance check
                if i == 0:
                    dist = np.sqrt((follower.x - self.leader.x)**2 + (follower.y - self.leader.y)**2)
                else:
                    dist = np.sqrt(
                        (follower.x - self.followers[i-1].x)**2 +
                        (follower.y - self.followers[i-1].y)**2
                    )
                min_dist_timestep = min(min_dist_timestep, dist)

            self.min_distances.append(min_dist_timestep)

            # 4) TTC
            min_ttc = self.calculate_ttc()
            self.ttc_history.append(min_ttc)

            # 3) Blockchain checks & mining each step
            # - check for FDI vs. leader velocity, etc.
            start_time = time()
            blockchain.check_for_attacks(t_step, self.leader.v_x)

            proof = blockchain.proof_of_work(blockchain.last_block)
            blockchain.new_block(proof, blockchain.hash(blockchain.last_block))
            end_time = time()
            # Calculate the time taken for the loop to complete
            elapsed_time = end_time - start_time
            self.elapsed_times.append(elapsed_time)  # Store elapsed time
            # 4) Reconcile states from the chain (optional)

            for i, follower in enumerate(self.followers):
                follower.check_and_update_from_blockchain(self.followers)
                self.v_history[i+1].append(follower.v_x)


            # Leader could also do so if you want fully consistent states:
            self.leader.check_and_update_from_blockchain([self.leader])
            self.v_history[0].append(self.leader.v_x)

            # Log
            self.log_system_status(t_step)

        self.plot_trajectory_snapshots()
        self.plot_velocity_and_min_distance()
        self.plot_elapsed_times()
        self.plot_ttc_over_time()

    def log_system_status(self, t_step):
        logging.info(f"Step {t_step} - System Status:")
        print(f"📊 [System Status] Step {t_step}:")
        for v in self.vehicles:
            logging.info(
                f"  Vehicle {v.id}: Pos=({v.x:.2f}, {v.y:.2f}), "
                f"Vx={v.v_x:.2f}, Qsize={len(v.message_queue)}, "
                f"Recovery={v.recovery_mode}"
            )
            print(
                f"  Vehicle {v.id}: Pos=({v.x:.2f}, {v.y:.2f}), "
                f"Vx={v.v_x:.2f}, Qsize={len(v.message_queue)}, "
                f"Recovery={v.recovery_mode}"
            )

    ##################################################
    # Plotting (optional)
    ##################################################
    def plot_trajectory_snapshots(self):
        plt.figure(figsize=(15, 8))
        t_samples = [
            int(self.time_steps*0.01), int(self.time_steps*0.2),
            int(self.time_steps*0.4),  int(self.time_steps*0.6),
            int(self.time_steps*0.8),  int(self.time_steps*0.99)
        ]
        for idx, t in enumerate(t_samples):
            plt.subplot(3, 2, idx+1)
            for i in range(self.num_followers+1):
                if i == 0:
                    plt.plot(self.x_history[i][t], self.y_history[i][t], '.r', markersize=10)
                else:
                    plt.plot(self.x_history[i][t], self.y_history[i][t], '.k', markersize=5)

            leader_x = max(self.x_history[0]) if self.x_history[0] else 100
            plt.plot([0, leader_x+20], [0, 0], 'b-', linewidth=2)
            plt.plot([0, leader_x+20], [self.road_width, self.road_width], 'b-', linewidth=2)

            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title(f't={t*self.dt:.2f} sec')
            plt.ylim(-5, self.road_width+5)
        plt.tight_layout()
        plt.show()


    def plot_velocity_and_min_distance(self):
        # Create a figure with two subplots
        plt.figure(figsize=(10, 12))

        # Subplot for Velocity Consensus
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.vehicles)))
        for i in range(len(self.vehicles)):
            label = 'Leader' if i == 0 else f'Follower {i}'
            plt.plot(self.time_points, self.v_history[i], label=label, color=colors[i], linewidth=2.0)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.ylim(0, 15)

        # Subplot for Minimum Distance
        plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
        plt.plot(self.time_points, self.min_distances, 'b-', label='Min Distance')
        plt.axhline(y=self.desired_gap, color='r', linestyle='--', label='Desired Gap')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 25)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_elapsed_times(self):
        plt.figure(figsize=(10,6))
        plt.bar(range(len(self.total_attack_delay_times)), self.total_attack_delay_times, color='blue')
        plt.xlabel('Simulation Step')
        plt.ylabel('Time delay (s)')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_ttc_over_time(self):
        time_points = np.arange(len(self.ttc_history))*self.dt
        plt.figure(figsize=(10,6))
        plt.plot(time_points, self.ttc_history, 'g-', label='Minimum TTC')
        plt.axhline(y=1, color='r', linestyle='--', label='TTC=1s')
        plt.axhline(y=2, color='orange', linestyle='--', label='TTC=2s')
        plt.xlabel('Time [s]')
        plt.ylabel('TTC [s]')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, time_points[-1])
        plt.tight_layout()
        plt.show()


##################################################
# Main Execution (with Scenario Table)
##################################################
if __name__ == "__main__":
    # Define multiple scenarios (like a Table)
    scenarios = [
        {
            "name": "No Attack",
            "attackers": 0,
            "intensity": 0,
            "start": None,
            "end": None
        },
        {
            "name": "Mild Flood",
            "attackers": 2,
            "intensity": 200,
            "start": 50,
            "end": 150
        },
        {
            "name": "Severe Flood",
            "attackers": 3,
            "intensity": 1000,
            "start": 50,
            "end": 150
        },
        {
            "name": "Very Severe Flood",
            "attackers": 4,
            "intensity": 10000,
            "start": 50,
            "end": 150
        },
    ]

    results = []

    for scn in scenarios:
        print("\n========================================")
        print(f"Running Scenario: {scn['name']}")
        print("========================================")

        # Create a new simulation
        num_followers = 10
        # Instantiate blockchain
        blockchain = Blockchain(num_followers)
        sim = LeaderFollowerSimulation(num_followers,blockchain)

        # Overwrite default attackers
        new_attackers = []
        for i in range(scn["attackers"]):
            new_attackers.append(Attacker(attacker_id=1001 + i))
        sim.attackers = new_attackers

        # Build attack_params if attackers > 0
        if scn["attackers"] > 0:
            attack_params = {
                'start': scn["start"] if scn["start"] else 0,
                'end': scn["end"] if scn["end"] else sim.time_steps - 1,
                'targets': [2, 6],  # arbitrary chosen
                'intensity': scn["intensity"],
                'type': 'flood'
            }
        else:
            attack_params = None

        # Run
        sim.run_simulation(attack_params)

        # Compute metrics
        # Minimum distance
        global_min_dist = min(sim.min_distances) if sim.min_distances else float('inf')
        # Packet loss
        if sim.total_messages_sent > 0:
            loss_percent = 100.0 * (1.0 - sim.total_messages_processed / sim.total_messages_sent)
        else:
            loss_percent = 0.0

        results.append({
            "Scenario": scn["name"],
            "Attackers": scn["attackers"],
            "Intensity": scn["intensity"],
            "Min Dist": round(global_min_dist, 2),
            "Loss (%)": round(loss_percent, 2),
            "Max Delay (s)": round(sim.max_delay, 4)  # Add max delay to results
        })

    # Show table
    df = pd.DataFrame(results)
    print("\n========== Simulation Results Table ==========")
    print(df.to_string(index=False))
    print("=============================================\n")
