import random
import numpy as np
import matplotlib.pyplot as plt
from main import Grid, TaxiDomain


class Q_Learning:

    def __init__(self, taxi_domain_problem, learning_rate, discount, exploration_rate, decay_exploration):
        self.tdp = taxi_domain_problem
        self.grid = tdp.grid
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_rate = exploration_rate
        self.decay_exploration = decay_exploration # True or False
        self.q_values = {}
        for state in self.tdp.all_states:
            for action in TaxiDomain.actions:
                self.q_values[(state, action)] = 0
        self.valid_passenger_depots = ["R", "B", "G", "Y"]
        self.valid_passenger_depots.remove(tdp.dest_depot)

    def initialize_episode(self):
        passenger_loc = self.grid.depots[random.sample(self.valid_passenger_depots, 1)[0]]
        taxi_loc = (random.randint(0,4), random.randint(0,4))
        passenger_in_taxi = False
        self.tdp.state = (taxi_loc, passenger_loc, passenger_in_taxi) 

    def choose_action(self, state, num_updates):
        sample = random.uniform(0.0, 1.0)
        exp_rate = self.exploration_rate
        if self.decay_exploration:
            exp_rate /= num_updates
        if sample <= exp_rate: # choose random action (explore)
            return random.choice(["N", "E", "S", "W", "PICKUP", "PUTDOWN"])            
        else: # choose action with best q-value (greedy)
            best_action = None
            best_value = None
            for action in TaxiDomain.actions:
                qv = self.q_values[(state, action)]
                if best_value == None or best_value < qv:
                    best_value = qv
                    best_action = action
            return best_action

    def learn(self, num_episodes):
        tdp = self.tdp
        num_updates = 0
        for episode in range(num_episodes):
            print(f"Episode no.: {episode}")
            self.initialize_episode()
            num_step = 0
            while not tdp.state == tdp.goal_state and num_step < 500:
                num_step += 1
                num_updates += 1
                print(f"\nStep No.: {num_step}")
                a = self.choose_action(tdp.state, num_updates)
                next_state, reward = tdp.take_action(tdp.state, a)
                max_q_next_state = 0
                for action in TaxiDomain.actions:
                    max_q_next_state = max(max_q_next_state, self.q_values[(tdp.state, action)])
                sample = reward + self.discount * max_q_next_state
                old_q_value = self.q_values[(tdp.state, a)]
                self.q_values[(tdp.state, a)] = (1-self.learning_rate)*self.q_values[(tdp.state, a)] + self.learning_rate*sample
                tdp.print_state()
                print(f"Q({tdp.state}, {a}) updated from {old_q_value} to {self.q_values[(tdp.state, a)]}")
                tdp.state = next_state

# class SARSA_Learning:

if __name__ == "__main__":
    grid = Grid(1)
    tdp = TaxiDomain(grid)
    ql = Q_Learning(tdp, 0.25, 0.99, 0.1, False)
    ql.learn(2000)