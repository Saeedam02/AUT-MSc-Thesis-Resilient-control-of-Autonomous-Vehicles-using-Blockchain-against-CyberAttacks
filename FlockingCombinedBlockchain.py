import hashlib
import json
from time import time
import numpy as np
import matplotlib.pyplot as plt
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
        :param vehicle_id: ID of the vehicle
        :param status: Current state of the vehicle
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
        return guess_hash[:4] == "0000"

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
        self.blockchain.register_node(self.id)
               
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
        self.v_x = xx[3]  # Longitudinal velocity (initial value)
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
        v_x = xx[3]  # Longitudinal velocity (initial value)
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
        v_x_dot = a - (F_yf * np.sin(delta)) / self.m + v_y * r  # Longitudinal velocity (initial value)
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
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'yaw': self.psi,
            'vx': self.v_x,
            'vy': self.v_y
        }
    
    def get_velocity(self):
        """
        Calculate and return the total velocity of the vehicle.
        :return: The magnitude of the total velocity.
        """
        return np.sqrt(self.v_x**2 + self.v_y**2)

class FlockingSimulation:
    def __init__(self, num_agents,blockchain):
        self.num_agents = num_agents
        self.agents = [DynamicBicycleModel(blockchain,i,  x=np.random.rand() * 50, y=np.random.rand() * 50, vx=np.random.rand() * 12) for i in range(num_agents)]
        self.dt = 0.05
        self.time_steps = int(500 / self.dt)
        self.ca1, self.ca2 = 10, 2 * np.sqrt(10)
        self.cb1, self.cb2 = 5, 0
        self.cg1, self.cg2 = 15, 2 * np.sqrt(15)
        self.obstacles = np.array([[180, 200], [20, 50], [8, 8]])  # Obstacles [x, y, radius]

        # flock Parameters
        self.d = 7
        self.r = 1.2*self.d
        self.d_b = 30
        self.r_b = 1.2*self.d_b

    def flocking_algorithm(self,X, V, agent_idx,x_des,v_des):

        y = (self.f_alpha(X, V, agent_idx) +
             self.f_beta(X, V, agent_idx) +
             self.f_gamma(X, V, agent_idx,x_des,v_des))
        
        return y
    
    def f_alpha(self,X, V, i):
        """
        Flocking Behavior: Ensures agents maintain a certain distance from each other.
        """
        h_a = 0.2
        n = len(X)
        def phi_a(z):
            return self.ph((z / self.norm_a(self.r)), h_a) * self.phi(z - self.norm_a(self.d))
        y_a1 = np.zeros((2, 1))
        y_a2 = np.zeros((2, 1))

        for j in range(n):
            if np.linalg.norm(X[:, j] - X[:, i]) < self.r and i != j:
                y_a1 = y_a1 + phi_a(self.norm_a(X[:, j] - X[:, i])) * self.d_norm_a(X[:, j] - X[:, i])

                A_ij = self.ph(self.norm_a(X[:, j] - X[:, i]) / self.norm_a(self.r), h_a)
                y_a2 = y_a2 + A_ij * (V[:, j] - V[:, i])

        y = self.ca1 * y_a1 + self.ca2 * y_a2
        return y

    def f_beta(self,X, V, i):
        """
        Obstacle Avoidance : Keeps agents from colliding with obstacles.
        """
        h_b = 0.9
        n_o = self.obstacles.shape[1]

        def phi_b(z):
            return self.ph(z / self.norm_a(self.d_b), h_b) * ((z - self.d_b) / np.sqrt(1 + (z - self.d_b) ** 2) - 1)

        y_b1 = np.zeros((2, 1))
        y_b2 = np.zeros((2, 1))

        if n_o > 0:
            for k in range(n_o):
                if np.linalg.norm(self.obstacles[:2, k] - X[:, i]) < self.r_b + self.obstacles[2, k]:
                    mu = self.obstacles[2, k] / np.linalg.norm(X[:, i] - self.obstacles[0:2, k])
                    q_beta = mu * X[:, i] + (1 - mu) * self.obstacles[0:2, k]
                    qik = q_beta - X[:, i]
                    y_b1 = y_b1 + phi_b(self.norm_a(qik)) * self.d_norm_a(qik)

                    ak = (X[:, i] - self.obstacles[0:2, k]) / np.linalg.norm(X[:, i] - self.obstacles[0:2, k])
                    p_beta = mu * (np.eye(2) - np.outer(ak, ak)) * V[:, i]
                    B_ik = self.ph(self.norm_a(qik) / self.norm_a(self.r_b), h_b)
                    y_b2 = y_b2 + B_ik * (p_beta - V[:, i])

        y = self.cb1 * y_b1 + self.cb2 * y_b2
        return y

    def f_gamma(self, X, V, i, x_des, v_des):
        """
        Leader Following: Directs agents toward the desired position (x_des)
        """
        q_bar = x_des - X[:, i]
        v_bar = v_des - V[:, i]

        # Apply stronger forces to move toward the leader
        y = self.cg1 * (q_bar / (np.linalg.norm(q_bar) + 1e-3)) + self.cg2 * v_bar
        return y


    def norm_a(self,z):

        z_norm_sq = np.linalg.norm(z)**2
        return (np.sqrt(1 + 0.1 * z_norm_sq) - 1) / 0.1

    def d_norm_a(self,z):
        return z / (1 + 0.1 * self.norm_a(z))

    def ph(self,z,h):
        if 0 <= z < h:
            y = 1
        elif h <= z < 1:
            y = (1 + np.cos(np.pi * (z - h) / (1 - h))) / 2
        else:
            y = 0
        return y

    def phi(self,z):
        a = 5
        b = 5
        c = abs(a - b) / np.sqrt(4 * a * b)
        y = ((a + b) * ((z + c) / np.sqrt(1 + (z + c) ** 2)) + (a - b)) / 2
        return y

    def run_simulation(self):



        X = np.array([[agent.x, agent.y] for agent in self.agents]).T
        V = np.array([[agent.v_x, agent.v_y] for agent in self.agents]).T

        x_history = [[] for _ in range(self.num_agents)]
        y_history = [[] for _ in range(self.num_agents)]
        vx_history = [[] for _ in range(self.num_agents)]

        x_leader = []
        y_leader = []
        vx_leader = []
        leader = DynamicBicycleModel(blockchain,1 , x=130, y=40, vx=6)
        for t in range(self.time_steps):
            #Proportional Control to Target Speed:
            v_target = 6.0  # Target longitudinal speed (m/s)
            k_p = 0.8  # Proportional gain
            a_l = k_p * (v_target - leader.v_x)
            leader.update(a_l,0.1 * np.sin(0.2 * t * self.dt))
            x_leader.append(leader.x)
            y_leader.append(leader.y)
            vx_leader.append(leader.v_x)
            for i, agent in enumerate(self.agents):

                x_des = leader.x
                v_des = leader.v_x
                a = self.flocking_algorithm(X, V, i, x_des, v_des)

                # Normalize and apply a gain to the acceleration (optional to avoid large changes)
                a_magnitude = np.linalg.norm(a)
                if a_magnitude > 1:
                    a = a / a_magnitude

                agent.update(a=a[0, 0], delta=0.1 * np.sin(0.2 * t * self.dt))

                x_history[i].append(agent.x)
                y_history[i].append(agent.y)
                vx_history[i].append(agent.v_x)

            X = np.array([[agent.x, agent.y] for agent in self.agents]).T
            V = np.array([[agent.v_x, agent.v_y] for agent in self.agents]).T

        # Plot the trajectory of each agent
        fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        t_samples = [int(self.time_steps * 0.01), int(self.time_steps * 0.2), int(self.time_steps * 0.4),
                     int(self.time_steps * 0.6), int(self.time_steps * 0.8), int(self.time_steps * 0.99)]
        for idx, t in enumerate(t_samples):
            ax = axs[idx // 2, idx % 2]
            for i in range(self.num_agents):
                ax.plot(x_history[i][t], y_history[i][t], '.k', markersize=5)
            ax.plot(x_leader[t], y_leader[t], '.r', markersize=15)
            for k in range(self.obstacles.shape[1]):
                circle = plt.Circle(self.obstacles[:2, k], self.obstacles[2, k], color='blue', alpha=0.5)
                ax.add_artist(circle)
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_title(f'Flocking in time: {t * self.dt:.2f} sec')
            ax.axis('equal')

        plt.tight_layout()
        plt.show()

        # Plot velocity consensus (longitudinal velocities over time)
        plt.figure(figsize=(10, 6))
        for i in range(self.num_agents):
            plt.plot(np.linspace(0, self.time_steps * self.dt, self.time_steps), vx_history[i], label=f'Agent {i+1}')
    
        # Plot the leader's velocity
        plt.plot(np.linspace(0, self.time_steps * self.dt, self.time_steps), vx_leader, label='Leader', color='red', linestyle='--')

        plt.xlabel('Time (s)')
        plt.ylabel('Velocity v_x (m/s)')
        plt.title('Velocity Consensus - Longitudinal Velocities Over Time')
        plt.legend(loc='best', fontsize='small', ncol=2)
        plt.grid(True)
        plt.show()


        #Plot minimum distance between agents over time
        min_distances = []
        for t in range(self.time_steps):
            distances = []
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    dist = np.linalg.norm(np.array([x_history[i][t], y_history[i][t]]) - np.array([x_history[j][t], y_history[j][t]]))
                    distances.append(dist)
            min_distances.append(min(distances))

        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, self.time_steps * self.dt, self.time_steps), min_distances)
        plt.xlabel('Time (s)')
        plt.ylabel('Minimum Distance (m)')
        plt.title('Minimum Distance Between Agents Over Time')
        plt.grid(True)
        plt.show()


num_agents = 10
# Create a single blockchain instance
blockchain = Blockchain()

# Run the simulation
simulation = FlockingSimulation(num_agents,blockchain)
simulation.run_simulation()
print(blockchain.nodes)