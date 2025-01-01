import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time, sleep
from collections import deque
from datetime import datetime
import logging

###############################################################################
# Logging Configuration
###############################################################################
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

###############################################################################
# Security Monitor (for local queue-based DDoS detection)
###############################################################################
class SecurityMonitor:
    def __init__(self):
        self.message_rates = {}
        self.velocity_changes = {}
        self.window_size = 50  # Time window for rate monitoring

    def track_message_rate(self, vehicle_id, current_queue_size):
        if vehicle_id not in self.message_rates:
            self.message_rates[vehicle_id] = deque(maxlen=self.window_size)
        self.message_rates[vehicle_id].append(current_queue_size)

    def track_velocity_change(self, vehicle_id, velocity):
        if vehicle_id not in self.velocity_changes:
            self.velocity_changes[vehicle_id] = deque(maxlen=self.window_size)
        self.velocity_changes[vehicle_id].append(velocity)

    def detect_anomalies(self, vehicle_id):
        """
        Simple anomaly detection:
          - High queue size average => potential DDoS
          - Large velocity jump => potential FDI or extreme event
        """
        anomalies = []

        # Check for high message rate anomalies
        if len(self.message_rates.get(vehicle_id, [])) >= 2:
            rate = np.mean(list(self.message_rates[vehicle_id]))
            if rate > 100:  # Example threshold
                anomalies.append(f"High message rate: {rate:.2f}")

        # Check for sudden velocity change anomalies
        if len(self.velocity_changes.get(vehicle_id, [])) >= 2:
            velocities = list(self.velocity_changes[vehicle_id])
            velocity_change = abs(velocities[-1] - velocities[-2])
            if velocity_change > 20:  # Example threshold
                anomalies.append(f"Suspicious velocity change: {velocity_change:.2f}")

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


###############################################################################
# Attacker Class (multiple sources for DDoS)
###############################################################################
class Attacker:
    """
    Attacker class that only has an ID and can generate spam packets
    """
    def __init__(self, attacker_id):
        self.id = attacker_id

    def generate_attack_traffic(self, intensity):
        """
        Generates spam messages based on intensity.
        Each message contains an 'attacker_id' field for origin identification.
        """
        flood_messages = [
            {'timestamp': time(), 'type': 'status_update', 'attacker_id': self.id}
            for _ in range(intensity)
        ]
        return flood_messages


###############################################################################
# Extended Dynamic Bicycle Model (local message queue + blockchain node)
###############################################################################
class DynamicBicycleModel:
    """
    Extended bicycle model with:
      - Local message queue (for DDoS flooding)
      - SecurityMonitor usage
      - Blockchain for global state
    """
    def __init__(self, blockchain, security_monitor, id,
                 x=0.0, y=0.0, psi=0.0, vx=10.0, vy=0.0, r=0.0):
        self.blockchain = blockchain
        self.security_monitor = security_monitor
        self.id = id

        # Vehicle states
        self.x = x
        self.y = y
        self.psi = psi
        self.v_x = vx
        self.v_y = vy
        self.r = r

        # Local DDoS/flood simulation
        self.message_queue = deque(maxlen=1000)
        self.processing_delay = 0.001
        self.last_update_time = time()

        # Recovery mode if local anomalies are detected
        self.recovery_mode = False

        # Physical parameters
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

    def update_dynamics(self):
        """
        Process local message queue -> check anomalies -> possibly recover -> record velocity
        Then publish final state to blockchain (transactions).
        """
        current_time = time()
        time_delta = current_time - self.last_update_time
        self.last_update_time = current_time

        # Process messages up to a small time budget to simulate limited CPU
        start_time = time()
        messages_processed = 0
        while self.message_queue and (time() - start_time) < 0.1:
            msg = self.message_queue.popleft()
            # Optional: You can log or analyze msg['attacker_id'] here
            messages_processed += 1
            sleep(self.processing_delay)

        # Track queue size + velocity in security monitor
        queue_size = len(self.message_queue)
        self.security_monitor.track_message_rate(self.id, queue_size)
        self.security_monitor.track_velocity_change(self.id, self.v_x)

        # Detect anomalies
        anomalies = self.security_monitor.detect_anomalies(self.id)
        if anomalies:
            logging.warning(f"Vehicle {self.id} - Anomalies detected: {anomalies}")
            # **Print Statement for Anomaly Detection**
            print(
                f"🚨 [Anomaly Detected] Vehicle ID: {self.id} has detected anomalies: {', '.join(anomalies)}. "
                f"Entering Recovery Mode."
            )
            self.recovery_mode = True

        # If in recovery, slow down
        if self.recovery_mode:
            self.apply_recovery_control()

        # Keep velocity history
        self.velocity_history.append(self.v_x)

    def apply_recovery_control(self):
        """
        Simple approach: reduce velocity by 5% if anomalies are detected.
        """
        self.v_x *= 0.95

    def update_physics(self, a, delta):
        """
        Standard 4th-order Runge-Kutta to update the vehicle’s physical states
        """
        xx = np.array([self.x, self.y, self.psi, self.v_x, self.v_y, self.r])

        k1 = self.f(xx, a, delta)
        k2 = self.f(xx + self.dt/2*k1, a, delta)
        k3 = self.f(xx + self.dt/2*k2, a, delta)
        k4 = self.f(xx + self.dt*k3, a, delta)

        xx = xx + self.dt*(k1 + 2*k2 + 2*k3 + k4)/6

        self.x, self.y, self.psi, self.v_x, self.v_y, self.r = xx
        
        # Finally, broadcast new transaction to the blockchain
        self.broadcast_state()

    def f(self, xx, a, delta):
        """
        Bicycle model dynamics
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

    def broadcast_state(self):
        """
        Publish (vehicle_id, state) to the blockchain as a transaction.
        """
        state = self.get_state()
        self.blockchain.new_transaction(self.id, state)

    def get_state(self):
        """
        Return current vehicle state
        """
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


###############################################################################
# Distributed Attack Simulation (DDoS)
###############################################################################
def simulate_distributed_attack(
    attackers,      # List of attackers (from Attacker class)
    vehicles,       # List of all vehicles (Leader + Followers)
    target_ids,     # IDs of victim vehicles
    intensity=1000,
    attack_type="flood"
):
    """
    Each attacker sends spam messages to target vehicles to simulate a DDoS attack.
    """
    if attack_type != "flood":
        logging.warning(f"Attack type '{attack_type}' not supported yet.")
        return

    for attacker in attackers:
        for vehicle in vehicles:
            if vehicle.id in target_ids:
                # Current attacker generates flood messages
                flood_messages = attacker.generate_attack_traffic(intensity)
                # Messages are added to the target vehicle's message queue
                vehicle.message_queue.extend(flood_messages)
                logging.info(
                    f"[DDoS] Attacker {attacker.id} -> Vehicle {vehicle.id}: "
                    f"Flooded {intensity} messages. Queue size now {len(vehicle.message_queue)}"
                )
                # **Print Statement for Terminal Output**
                print(
                    f"⚠️  [DDoS Attack] Attacker ID: {attacker.id} is attacking "
                    f"Vehicle ID: {vehicle.id} with {intensity} flood messages. "
                    f"New Queue Size: {len(vehicle.message_queue)}"
                )


###############################################################################
# Leader-Follower Simulation
###############################################################################
class LeaderFollowerSimulation:
    def __init__(self, num_followers, blockchain):
        self.blockchain = blockchain
        self.security_monitor = SecurityMonitor()
        self.elapsed_times = []  # List to store elapsed times for each step

        # Create leader
        self.leader = DynamicBicycleModel(
            blockchain=self.blockchain,
            security_monitor=self.security_monitor,
            id=0, x=100, y=10, vx=6.0
        )

        # Create followers
        self.followers = [
            DynamicBicycleModel(
                blockchain=self.blockchain,
                security_monitor=self.security_monitor,
                id=i+1,
                x=100 - 10*(i+1),
                y=10,
                vx=6.0
            )
            for i in range(num_followers)
        ]
        self.vehicles = [self.leader] + self.followers

        # Define list of attackers (multiple attack sources)
        self.attackers = [
            Attacker(attacker_id=1001),
            Attacker(attacker_id=1002),
        ]

        self.desired_gap = 10
        self.dt = 0.05
        self.time_steps = int(50 / self.dt)
        self.road_width = 20

        # For plotting
        self.x_history = [[] for _ in range(num_followers+1)]
        self.y_history = [[] for _ in range(num_followers+1)]
        self.v_history = [[] for _ in range(num_followers+1)]
        self.min_distances = []
        self.ttc_history = []  # Track minimum TTC over time

        self.time_points = np.arange(0, self.time_steps)*self.dt

    def calculate_ttc(self):
        """
        Calculate Time to Collision (TTC) for all vehicles.
        """
        ttc_values = []
        for i in range(1, len(self.vehicles)):  # Start from the first follower
            leader = self.vehicles[i - 1]
            follower = self.vehicles[i]

            relative_velocity = follower.v_x - leader.v_x
            relative_position = leader.x - follower.x - self.desired_gap

            # Calculate TTC only if conditions are valid
            if relative_position > 0 and relative_velocity > 0:
                ttc = relative_position / relative_velocity
            else:
                ttc = float('inf')  # No collision risk or invalid position

            ttc_values.append(ttc)

        return min(ttc_values)  # Return the minimum TTC across all pairs

    def run_simulation(self, attack_params=None):
        for step in range(self.time_steps):
            logging.info(f"--- Simulation Step {step} ---")

            # If attack parameters are defined, apply the attack
            if attack_params:
                start_step = attack_params.get('start', 0)
                end_step = attack_params.get('end', 0)
                if start_step <= step <= end_step:
                    simulate_distributed_attack(
                        attackers=self.attackers,
                        vehicles=self.vehicles,
                        target_ids=attack_params['targets'],
                        intensity=attack_params.get('intensity', 500),
                        attack_type=attack_params.get('type', 'flood')
                    )

            # 1) Leader update
            v_target = 6.0
            kp = 1.0
            a_leader = kp*(v_target - self.leader.v_x)

            # Security-based update (queue processing, anomaly check, etc.)
            self.leader.update_dynamics()
            # Physical update (Runge-Kutta)
            self.leader.update_physics(a_leader, 0)

            # 2) Followers update
            min_dist_step = float('inf')
            for i, follower in enumerate(self.followers):
                # Simple proportional control for spacing
                distance_to_leader = self.leader.x - follower.x - self.desired_gap*(i+1)
                a_f = 1.0 * distance_to_leader

                follower.update_dynamics()
                follower.update_physics(a_f, 0)

                # Measure distance
                if i == 0:
                    dist = np.sqrt((follower.x - self.leader.x)**2 + (follower.y - self.leader.y)**2)
                else:
                    dist = np.sqrt((follower.x - self.followers[i-1].x)**2 + 
                                   (follower.y - self.followers[i-1].y)**2)
                min_dist_step = min(min_dist_step, dist)

            self.min_distances.append(min_dist_step)

            # Calculate and store TTC
            min_ttc = self.calculate_ttc()
            self.ttc_history.append(min_ttc)

            # 3) Blockchain checks & mining each step
            # - check for FDI vs. leader velocity, etc.
            start_time = time()
            self.blockchain.check_for_attacks(step, self.leader.v_x)

            proof = self.blockchain.proof_of_work(self.blockchain.last_block)
            self.blockchain.new_block(proof, self.blockchain.hash(self.blockchain.last_block))
            end_time = time()
            # Calculate the time taken for the loop to complete
            elapsed_time = end_time - start_time
            self.elapsed_times.append(elapsed_time)  # Store elapsed time

            print(f"⏱️  Total simulation time_delay: {elapsed_time:.6f} seconds")
            print('************************')

            # 4) Reconcile states from the chain (optional)
            for i, follower in enumerate(self.followers):
                follower.check_and_update_from_blockchain(self.followers)
                self.v_history[i+1].append(follower.v_x)
                self.x_history[i+1].append(follower.x)
                self.y_history[i+1].append(follower.y)

            # Leader could also do so if you want fully consistent states:
            self.leader.check_and_update_from_blockchain([self.leader])
            self.x_history[0].append(self.leader.x)
            self.y_history[0].append(self.leader.y)
            self.v_history[0].append(self.leader.v_x)

            # 5) Log system status
            self.log_system_status(step)

        # Plot final results
        self.plot_trajectory_snapshots()
        self.plot_velocity_and_min_distance()
        self.plot_elapsed_times()
        self.plot_ttc_over_time()

    def log_system_status(self, step):
        logging.info(f"Step {step} - System Status:")
        print(f"📊 [System Status] Step {step}:")
        for v in self.vehicles:
            logging.info(
                f"Vehicle {v.id}: Pos=({v.x:.2f},{v.y:.2f}), "
                f"Vx={v.v_x:.2f}, QueueSize={len(v.message_queue)}, "
                f"RecoveryMode={v.recovery_mode}"
            )
            print(f"  Vehicle {v.id}: Pos=({v.x:.2f}, {v.y:.2f}), "
                  f"Vx={v.v_x:.2f}, QueueSize={len(v.message_queue)}, "
                  f"RecoveryMode={v.recovery_mode}")

    def plot_trajectory_snapshots(self):
        plt.figure(figsize=(15, 8))
        plt.suptitle('Vehicle Platooning Trajectory Snapshots', fontsize=14)
        t_samples = [
            int(self.time_steps*0.01), int(self.time_steps*0.2),
            int(self.time_steps*0.4), int(self.time_steps*0.6),
            int(self.time_steps*0.8), int(self.time_steps*0.99)
        ]

        for idx, t in enumerate(t_samples):
            plt.subplot(3, 2, idx + 1)
            for i in range(len(self.vehicles)):
                if i == 0:
                    plt.plot(self.x_history[i][t], self.y_history[i][t],
                             '.r', markersize=10)  # Leader in red
                else:
                    plt.plot(self.x_history[i][t], self.y_history[i][t],
                             '.k', markersize=5)
            leader_x = max(self.x_history[0]) if self.x_history[0] else 100
            plt.plot([0, leader_x + 20], [0, 0], 'b-', linewidth=2)
            plt.plot([0, leader_x + 20], [self.road_width, self.road_width], 'b-', linewidth=2)

            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title(f't={t*self.dt:.2f}s')
            plt.ylim(-5, self.road_width + 5)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_velocity_and_min_distance(self):
        # Create a figure with two subplots
        plt.figure(figsize=(10, 12))
        plt.suptitle('Velocity and Minimum Distance Over Time', fontsize=14)

        # Subplot for Velocity Consensus
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.vehicles)))
        for i in range(len(self.vehicles)):
            label = 'Leader' if i == 0 else f'Follower {i}'
            plt.plot(self.time_points, self.v_history[i], label=label, color=colors[i], linewidth=2.0)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Velocity Consensus Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.ylim(0, 15)

        # Subplot for Minimum Distance
        plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
        plt.plot(self.time_points, self.min_distances, 'b-', label='Min Distance')
        plt.axhline(y=self.desired_gap, color='r', linestyle='--', label='Desired Gap')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.title('Minimum Inter-Vehicle Distance Over Time')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 25)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_elapsed_times(self):
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(self.elapsed_times)), self.elapsed_times, color='blue')
        plt.xlabel('Simulation Step')
        plt.ylabel('Elapsed Time (seconds)')
        plt.title('Elapsed Time per Simulation Step')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_ttc_over_time(self):
        """
        Plot the Time to Collision (TTC) over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_points, self.ttc_history, 'g-', label='Minimum TTC')
        plt.axhline(y=1, color='r', linestyle='--', label='TTC Threshold (1s)')
        plt.axhline(y=2, color='orange', linestyle='--', label='TTC Threshold (2s)')
        plt.xlabel('Time [s]')
        plt.ylabel('TTC [s]')
        plt.title('Time to Collision (TTC) Over Time')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, self.time_points[-1])
        plt.tight_layout()
        plt.show()


###############################################################################
# Main Execution
###############################################################################
if __name__ == "__main__":
    num_followers = 10
    # Instantiate blockchain
    blockchain = Blockchain(num_followers)

    # Create simulation with the shared blockchain
    simulation = LeaderFollowerSimulation(num_followers, blockchain)

    # Optional DDoS attack parameters
    attack_params = {
        'start': 50,      # Simulation step
        'end': 150,       # Simulation step
        'targets': [2, 6],  # Vehicles to attack
        'intensity': 10000,  # Number of flood messages per attacker
        'type': 'flood'
    }

    # Run the simulation (with DDoS attack)
    simulation.run_simulation(attack_params)

    logging.info(f"\n[Blockchain] Active Nodes: {blockchain.nodes}")
    if blockchain.suspicious_nodes:
        logging.info(f"[Blockchain] Suspicious Nodes: {blockchain.suspicious_nodes}")
    else:
        logging.info("[Blockchain] No suspicious nodes flagged.")
