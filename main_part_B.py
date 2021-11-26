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
        passenger_loc = self.grid.depots[random.choice(self.valid_passenger_depots)]
        taxi_loc = (random.randint(0,self.grid.size[1]-1), random.randint(0,self.grid.size[0]-1))
        passenger_in_taxi = False
        self.tdp.state = (taxi_loc, passenger_loc, passenger_in_taxi) 

    def choose_action(self, state, num_updates):
        sample = random.uniform(0.0, 1.0)
        exp_rate = self.exploration_rate
        if self.decay_exploration:
            exp_rate /= num_updates
        if sample <= exp_rate: # choose random action (explore)
            a = random.choice(list(TaxiDomain.actions))            
        else: # choose action with best q-value (greedy)
            best_action = None
            best_value = None
            for action in TaxiDomain.actions:
                qv = self.q_values[(state, action)]
                if best_value == None or best_value < qv:
                    best_value = qv
                    best_action = action
            a = best_action
        # print(a)
        return a

    def extract_policy(self, q_values):
        p = {}
        for state in self.tdp.all_states:
            best_action = None
            best_q_value = None
            for action in TaxiDomain.actions:
                if best_q_value == None or q_values[(state, action)] > best_q_value:
                    best_q_value = q_values[(state, action)]
                    best_action = action
            p[state] = best_action
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
            # initialize passenger and taxi locations for test_tdp
            passenger_loc = self.grid.depots[random.choice(self.valid_passenger_depots)]
            taxi_loc = (random.randint(0,self.grid.size[1]-1), random.randint(0,self.grid.size[0]-1))
            test_tdp.state = (taxi_loc, passenger_loc, False)
            # run this episode
            num_step = 0
            while not test_tdp.state == test_tdp.goal_state and num_step < 500:
                num_step += 1
                # print(f"\nStep No.: {num_step} [Evaluation]")
                # test_tdp.print_state()
                action = policy[test_tdp.state]
                next_state, reward = test_tdp.take_action(test_tdp.state, action)
                test_tdp.state = next_state
                dr += multiplier * reward
                multiplier *= discount
            drs.append(dr)
        return drs                

    def learn(self, num_episodes, compute_rewards):
        tdp = self.tdp
        num_updates = 0
        discounted_rewards = []
        average_discounted_rewards = []
        episode_nums = []
        for episode in range(num_episodes):
            print(f"Episode no.: {episode}")
            self.initialize_episode()
            num_step = 0
            while tdp.state != tdp.goal_state and num_step < 500:
                num_step += 1
                num_updates += 1
                # print(f"\nStep No.: {num_step}")
                a = self.choose_action(tdp.state, num_updates)
                next_state, reward = tdp.take_action(tdp.state, a)
                max_q_next_state = None
                for action in TaxiDomain.actions:
                    if max_q_next_state == None or max_q_next_state < self.q_values[(next_state, action)]:
                        max_q_next_state = self.q_values[(next_state, action)]
                sample = reward + self.discount * max_q_next_state
                old_q_value = self.q_values[(tdp.state, a)]
                self.q_values[(tdp.state, a)] = (1-self.learning_rate)*old_q_value + self.learning_rate*sample
                # tdp.print_state()
                # print(f"Q({tdp.state}, {a}) updated from {old_q_value} to {self.q_values[(tdp.state, a)]} by (1-{self.learning_rate})*{old_q_value}+{self.learning_rate}*[{reward}+{self.discount}*{max_q_next_state}]")
                tdp.state = next_state
            if compute_rewards and episode!=0 and episode%20 == 0:
                print(f"\nEvaluating episode no.: {episode}\n")
                policy = self.extract_policy(self.q_values)
                dr = self.compute_discounted_rewards(policy, 20)
                discounted_rewards.append(dr)
                average_discounted_rewards.append(sum(dr)/len(dr))
                episode_nums.append(episode)
        # evaluate last time
        print(f"\nEvaluating episode no.: {num_episodes-1}\n")
        policy = self.extract_policy(self.q_values)
        dr = self.compute_discounted_rewards(policy, 20)
        discounted_rewards.append(dr)
        average_discounted_rewards.append(sum(dr)/len(dr))
        episode_nums.append(num_episodes-1)
        return (episode_nums, discounted_rewards, average_discounted_rewards)

class SARSA_Learning:
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
        passenger_loc = self.grid.depots[random.choice(self.valid_passenger_depots)]
        taxi_loc = (random.randint(0,self.grid.size[1]-1), random.randint(0,self.grid.size[0]-1))
        passenger_in_taxi = False
        self.tdp.state = (taxi_loc, passenger_loc, passenger_in_taxi) 

    def choose_action(self, state, num_updates):
        sample = random.uniform(0.0, 1.0)
        exp_rate = self.exploration_rate
        if self.decay_exploration:
            exp_rate /= num_updates
        if sample <= exp_rate: # choose random action (explore)
            a = random.choice(list(TaxiDomain.actions))            
        else: # choose action with best q-value (greedy)
            best_action = None
            best_value = None
            for action in TaxiDomain.actions:
                qv = self.q_values[(state, action)]
                if best_value == None or best_value < qv:
                    best_value = qv
                    best_action = action
            a = best_action
        # print(a)
        return a

    def extract_policy(self, q_values):
        p = {}
        for state in self.tdp.all_states:
            best_action = None
            best_q_value = None
            for action in TaxiDomain.actions:
                if best_q_value == None or q_values[(state, action)] > best_q_value:
                    best_q_value = q_values[(state, action)]
                    best_action = action
            p[state] = best_action
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
            # initialize passenger and taxi locations for test_tdp
            passenger_loc = self.grid.depots[random.choice(self.valid_passenger_depots)]
            taxi_loc = (random.randint(0,self.grid.size[1]-1), random.randint(0,self.grid.size[0]-1))
            test_tdp.state = (taxi_loc, passenger_loc, False)
            # run this episode
            num_step = 0
            while not test_tdp.state == test_tdp.goal_state and num_step < 500:
                num_step += 1
                # print(f"\nStep No.: {num_step} [Evaluation]")
                # test_tdp.print_state()
                action = policy[test_tdp.state]
                next_state, reward = test_tdp.take_action(test_tdp.state, action)
                test_tdp.state = next_state
                dr += multiplier * reward
                multiplier *= discount
            drs.append(dr)
        return drs                

    def learn(self, num_episodes, compute_rewards):
        tdp = self.tdp
        num_updates = 0
        discounted_rewards = []
        average_discounted_rewards = []
        episode_nums = []
        for episode in range(num_episodes):
            print(f"Episode no.: {episode}")
            self.initialize_episode()
            num_step = 0
            while tdp.state != tdp.goal_state and num_step < 500:
                num_step += 1
                num_updates += 1
                # print(f"\nStep No.: {num_step}")
                a = self.choose_action(tdp.state, num_updates)
                next_state, reward = tdp.take_action(tdp.state, a)
                a_next = self.choose_action(next_state, num_updates)
                q_next_state = self.q_values[(next_state, a_next)]
                
                sample = reward + self.discount * q_next_state
                old_q_value = self.q_values[(tdp.state, a)]
                self.q_values[(tdp.state, a)] = (1-self.learning_rate)*old_q_value + self.learning_rate*sample
                # tdp.print_state()
                # print(f"Q({tdp.state}, {a}) updated from {old_q_value} to {self.q_values[(tdp.state, a)]} by (1-{self.learning_rate})*{old_q_value}+{self.learning_rate}*[{reward}+{self.discount}*{max_q_next_state}]")
                tdp.state = next_state
            if compute_rewards and episode!=0 and episode%20 == 0:
                print(f"\nEvaluating episode no.: {episode}\n")
                policy = self.extract_policy(self.q_values)
                dr = self.compute_discounted_rewards(policy, 20)
                discounted_rewards.append(dr)
                average_discounted_rewards.append(sum(dr)/len(dr))
                episode_nums.append(episode)
        # evaluate last time
        print(f"\nEvaluating episode no.: {num_episodes-1}\n")
        policy = self.extract_policy(self.q_values)
        dr = self.compute_discounted_rewards(policy, 20)
        discounted_rewards.append(dr)
        average_discounted_rewards.append(sum(dr)/len(dr))
        episode_nums.append(num_episodes-1)
        return (episode_nums, discounted_rewards, average_discounted_rewards)


def partB_2():
    alpha = 0.25
    gamma = 0.99
    epsilon = 0.1
    grid = Grid(1)

    # Q-Learning with epsilon greedy
    tdp1 = TaxiDomain(grid)
    ql1 = Q_Learning(tdp1, alpha, gamma, epsilon, False)
    ens1, drs1, adrs1 = ql1.learn(2000, True)
    
    # Q-Learning with decaying epsilon greedy rate
    tdp2 = TaxiDomain(grid)
    ql2 = Q_Learning(tdp2, alpha, gamma, epsilon, True)
    ens2, drs2, adrs2 = ql2.learn(2000, True)

    # SARSA Learning with epsilon greedy
    tdp3 = TaxiDomain(grid)
    ql3 = SARSA_Learning(tdp3, alpha, gamma, epsilon, False)
    ens3, drs3, adrs3 = ql3.learn(2000, True)

    # SARSA Learning with decaying epsilon greedy rate
    tdp4 = TaxiDomain(grid)
    ql4 = SARSA_Learning(tdp4, alpha, gamma, epsilon, True)
    ens4, drs4, adrs4 = ql4.learn(2000, True)

    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.plot(ens1, adrs1, label=f"Q-Learning")
    plt.plot(ens2, adrs2, label=f"Q-Learning with decaying exploration")
    plt.plot(ens3, adrs3, label=f"SARSA Learning")
    plt.plot(ens4, adrs4, label=f"SARSA Learning with decaying exploration")
    plt.xlabel("Episode No.")
    plt.ylabel("Average discounted reward (over 20 episodes)")
    plt.title("Average discounted reward vs. Training episodes")
    plt.legend()
    plt.show()

    fig_name = "PartB_2.png"
    fig.savefig(fig_name, dpi=100)

    print(f"Plot '{fig_name}' generated...")

def partB_3():
    alpha = 0.25
    gamma = 0.99
    epsilon = 0.1
    grid = Grid(1)
    # training using Q-Learning with decaying epsilon greedy rate
    tdp = TaxiDomain(grid)
    ql = Q_Learning(tdp, alpha, gamma, epsilon, True)
    ens, drs, adrs = ql.learn(2000, False)
    # extract policy
    policy = ql.extract_policy(ql.q_values)
    # test policy on 5 episodes
    test_tdp = TaxiDomain(grid)
    test_tdp.dest_depot = tdp.dest_depot
    test_tdp.dest_loc = tdp.dest_loc
    test_tdp.goal_state = tdp.goal_state

    for trial in range(5):
        print("\n\nEpisode No. = "+str(trial)+"\n")
        # initialize passenger and taxi locations
        passenger_loc = test_tdp.grid.depots[random.sample(ql.valid_passenger_depots, 1)[0]]
        taxi_loc = (random.randint(0,test_tdp.grid.size[1]-1), random.randint(0,test_tdp.grid.size[0]-1))
        test_tdp.state = (taxi_loc, passenger_loc, False)
        # run this episode
        num_step = 0
        while not test_tdp.state == test_tdp.goal_state and num_step < 500:
            num_step += 1
            print(f"\nStep No.: {num_step}")
            test_tdp.print_state()
            action = policy[test_tdp.state]
            next_state, reward = test_tdp.take_action(test_tdp.state, action)
            test_tdp.state = next_state

def partB_4():
    grid = Grid(1)
    gamma = 0.99
    alpha = 0.1

    # vary exploration rate
    print("\nRunning Q-Learning for different exploration rates...")
    ens_list, adrs_list = [], []
    exp_rates = [0, 0.05, 0.1, 0.5, 0.9]
    for exp in exp_rates:
        print(f"\n\nRunning for exploration rate={exp}\n")
        tdp = TaxiDomain(grid)
        ql = Q_Learning(tdp, alpha, gamma, exp, False)
        ens, drs, adrs = ql.learn(2000, True)
        ens_list.append(ens)
        adrs_list.append(adrs)

    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    for i in range(len(exp_rates)):
        exp = exp_rates[i]
        plt.plot(ens_list[i], adrs_list[i], label=f"Exploration rate: {exp}")
    plt.xlabel("Episode No.")
    plt.ylabel("Average discounted reward (over 20 episodes)")
    plt.title("Average discounted reward vs. Training episodes as exploration rate varies")
    plt.legend()
    plt.show()

    fig_name = "PartB_4_exploration.png"
    fig.savefig(fig_name, dpi=100)

    print(f"Plot '{fig_name}' generated...")

    # vary learning rate
    epsilon = 0.1
    ens_list, adrs_list = [], []
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    print("\nRunning Q-Learning for different learning rates...")
    for lr in learning_rates:
        print(f"\n\nRunning for learning rate={lr}\n")
        tdp = TaxiDomain(grid)
        ql = Q_Learning(tdp, lr, gamma, epsilon, False)
        ens, drs, adrs = ql.learn(2000, True)
        ens_list.append(ens)
        adrs_list.append(adrs)

    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    for i in range(len(learning_rates)):
        lr = learning_rates[i]
        plt.plot(ens_list[i], adrs_list[i], label=f"Learning rate: {lr}")
    plt.xlabel("Episode No.")
    plt.ylabel("Average discounted reward (over 20 episodes)")
    plt.title("Average discounted reward vs. Training episodes as learning rate varies")
    plt.legend()
    plt.show()

    fig_name = "PartB_4_learning.png"
    fig.savefig(fig_name, dpi=100)

    print(f"Plot '{fig_name}' generated...")

def partB_5():
    # SEEE: What are the best learning and exploration rates? Set them here
    alpha = 0.5
    gamma = 0.99
    epsilon = 0.2
    grid = Grid(2)
    
    # Q-Learning with decaying epsilon greedy rate
    tdp2 = TaxiDomain(grid)
    ql2 = Q_Learning(tdp2, alpha, gamma, epsilon, True)
    ens2, drs2, adrs2 = ql2.learn(10000, True)

    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.plot(ens2, adrs2, label=f"Q-Learning with decaying exploration")
    plt.xlabel("Episode No.")
    plt.ylabel("Average discounted reward (over 20 episodes)")
    plt.title("Average discounted reward vs. Training episodes on 10*10 Grid")
    plt.legend()
    plt.show()

    fig_name = "PartB_5.png"
    fig.savefig(fig_name, dpi=100)

    print(f"Plot '{fig_name}' generated...")

if __name__ == "__main__":
    # grid = Grid(1)
    # tdp = TaxiDomain(grid)
    # ql = Q_Learning(tdp, 0.25, 0.99, 0.1, False)
    # ql.learn(2000)
    # partB_2()
    # partB_3()
    #partB_4()
    partB_5()
