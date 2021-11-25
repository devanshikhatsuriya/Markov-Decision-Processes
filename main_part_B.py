import random
import numpy as np
import matplotlib.pyplot as plt
from main import Grid, TaxiDomain


class Q_Learning:

    def __init__(self, taxi_domain_problem, learning_rate, discount, exploration_rate, decay_exploration):
        self.tdp = taxi_domain_problem
        self.grid = self.tdp.grid
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_rate = exploration_rate
        self.decay_exploration = decay_exploration # True or False
        self.q_values = {}
        for state in self.tdp.all_states:
            for action in TaxiDomain.actions:
                self.q_values[(state, action)] = 0
        if self.grid.size == (5,5):
            self.valid_passenger_depots = ["R", "B", "G", "Y"]
        elif self.grid.size == (10,10):
            self.valid_passenger_depots = ["R", "B", "G", "Y", "W", "M", "C", "P"]
        self.valid_passenger_depots.remove(self.tdp.dest_depot)

    def initialize_episode(self):
        passenger_loc = self.grid.depots[random.sample(self.valid_passenger_depots, 1)[0]]
        taxi_loc = (random.randint(0,self.grid.size[1]-1), random.randint(0,self.grid.size[0]-1))
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

    def extract_policy(self, q_values):
        p = {}
        for state in self.tdp.all_states:
            best_action = None
            best_q_value = None
            for action in TaxiDomain.actions:
                if best_q_value == None or q_values[(state, action)] > best_q_value:
                    best_q_value = q_values[(state, action)]
                    best_action = action
            p[state] = action
        return p

    def compute_discounted_rewards(self, policy, num_episodes):
        drs = []
        discount = self.discount
        test_tdp = TaxiDomain(self.grid)
        test_tdp.dest_depot = self.tdp.dest_depot
        test_tdp.dest_loc = self.tdp.dest_loc
        test_tdp.goal_state = self.tdp.goal_state

        for trial in range(num_episodes):
            dr = 0
            multiplier = 1
            # initialize passenger and taxi locations
            passenger_loc = self.grid.depots[random.sample(self.valid_passenger_depots, 1)[0]]
            taxi_loc = (random.randint(0,self.grid.size[1]-1), random.randint(0,self.grid.size[0]-1))
            test_tdp.state = (taxi_loc, passenger_loc, False)
            # run this episode
            num_step = 0
            while not test_tdp.state == test_tdp.goal_state and num_step < 500:
                num_step += 1
                print(f"\nStep No.: {num_step} [Evaluation]")
                # test_tdp.print_state()
                action = policy[test_tdp.state]
                next_state, reward = test_tdp.take_action(test_tdp.state, action)
                test_tdp.state = next_state
                dr += multiplier * reward
                multiplier *= discount
            drs.append(dr)
        return drs                

    def learn(self, num_episodes):
        tdp = self.tdp
        num_updates = 0
        discounted_rewards = []
        average_discounted_rewards = []
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
                # tdp.print_state()
                # print(f"Q({tdp.state}, {a}) updated from {old_q_value} to {self.q_values[(tdp.state, a)]}")
                tdp.state = next_state
            policy = self.extract_policy(self.q_values)
            dr = self.compute_discounted_rewards(policy, 20)
            discounted_rewards.append(dr)
            average_discounted_rewards.append(sum(dr)/len(dr))
        return (discounted_rewards, average_discounted_rewards)

# class SARSA_Learning:


def partB_2():
    average_discounted_rewards_list = []
    alpha = 0.25
    gamma = 0.99
    epsilon = 0.1
    grid = Grid(1)
    
    # Q-Learning with epsilon greedy
    tdp1 = TaxiDomain(grid)
    ql1 = Q_Learning(tdp1, alpha, gamma, epsilon, False)
    drs1, adrs1 = ql1.learn(2000)
    
    # Q-Learning with decaying epsilon greedy rate
    tdp2 = TaxiDomain(grid)
    ql2 = Q_Learning(tdp2, alpha, gamma, epsilon, True)
    drs2, adrs2 = ql2.learn(2000)

    # SARSA Learning with epsilon greedy
    # SARSA Learning with decaying epsilon greedy rate

    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.plot(list(range(len(adrs1))), adrs1, label=f"Q-Learning")
    plt.plot(list(range(len(adrs2))), adrs2, label=f"Q-Learning with decaying exploration")
    plt.xlabel("Episode No.")
    plt.ylabel("Average discounted reward (over 20 episodes)")
    plt.title("Average discounted reward vs. Training episodes")
    plt.legend()
    plt.show()

    fig_name = "PartB_2.png"
    fig.savefig(fig_name, dpi=100)

    print(f"Plot '{fig_name}' generated...")

if __name__ == "__main__":
    # grid = Grid(1)
    # tdp = TaxiDomain(grid)
    # ql = Q_Learning(tdp, 0.25, 0.99, 0.1, False)
    # ql.learn(2000)
    partB_2()