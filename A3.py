import random
import numpy as np
import matplotlib.pyplot as plt
import time

class Grid:

    def __init__(self, type):
        if type == 1:
            '''
            Grid:   4   R : . | . : . : G
                    3   . : . | . : . : .
                    2   . : . : . : . : .
                    1   . | . : . | . : . 
                    0   Y | . : . | B : .
                        0   1   2   3   4
            '''
            self.size = (5,5)
            self.array = [["Y", "|", ".", ":", ".", "|", "B", ":", "."],
                            [".", "|", ".", ":", ".", "|", ".", ":", "."],
                            [".", ":", ".", ":", ".", ":", ".", ":", "."],
                            [".", ":", ".", "|", ".", ":", ".", ":", "."],
                            ["R", ":", ".", "|", ".", ":", ".", ":", "G"],]
            self.right_walls = [[False for i in range(5)] for i in range(5)] # True for grid cells that have walls on their right side
            for cell in [(0,0), (1,0), (0,2), (1,2), (3,1), (4,1)]:
                self.right_walls[cell[0]][cell[1]] = True
            self.left_walls = [[False for i in range(5)] for i in range(5)] # True for grid cells that have walls on their left side
            for cell in [(0,1), (1,1), (0,3), (1,3), (3,2), (4,2)]:
                self.left_walls[cell[0]][cell[1]] = True
            # note that we model only vertical walls as there are no horizontal walls
            self.depots = {"R": (4,0), "B": (0,3), "G": (4,4), "Y": (0,0)}
            # cell locations are represented as (row_no, col_no) = (y_coord., x_coord.)
        elif type == 2:
            self.size = (10, 10)
            self.array = [[".", "|", ".", ":", ".", ":", ".", "|", "B", ":", ".", ":", ".", ":", ".", "|", ".", ":", "P"],
                            ["Y", "|", ".", ":", ".", ":", ".", "|", ".", ":", ".", ":", ".", ":", ".", "|", ".", ":", "."],
                            [".", "|", ".", ":", ".", ":", ".", "|", ".", ":", ".", ":", ".", ":", ".", "|", ".", ":", "."],
                            [".", "|", ".", ":", ".", ":", ".", "|", ".", ":", ".", ":", ".", ":", ".", "|", ".", ":", "."],
                            [".", ":", ".", ":", ".", ":", ".", ":", ".", ":", ".", "|", ".", ":", ".", ":", ".", ":", "."],
                            [".", ":", ".", ":", ".", ":", ".", ":", ".", ":", ".", "|", "M", ":", ".", ":", ".", ":", "."],
                            [".", ":", ".", ":", ".", "|", "W", ":", ".", ":", ".", "|", ".", ":", ".", "|", ".", ":", "."],
                            [".", ":", ".", ":", ".", "|", ".", ":", ".", ":", ".", "|", ".", ":", ".", "|", ".", ":", "."],
                            [".", ":", ".", ":", ".", "|", ".", ":", ".", ":", ".", ":", ".", ":", ".", "|", ".", ":", "."],
                            ["R", ":", ".", ":", ".", "|", ".", ":", ".", ":", "G", ":", ".", ":", ".", "|", "C", ":", "."]]
            self.right_walls = [[False for i in range(10)] for i in range(10)] # True for grid cells that have walls on their right side
            for cell in [(0,0), (1,0), (2,0), (3,0), (0,3), (1,3), (2,3), (3,3), (0,7), (1,7), (2,7), (3,7), (4,5), (5,5), (6,5), (7,5), (6,2), (7,2), (8,2), (9,2), (6,7), (7,7), (8,7), (9,7)]:
                self.right_walls[cell[0]][cell[1]] = True
            self.left_walls = [[False for i in range(10)] for i in range(10)] # True for grid cells that have walls on their left side
            for cell in [(0,1), (1,1), (2,1), (3,1), (0,4), (1,4), (2,4), (3,4), (0,8), (1,8), (2,8), (3,8), (4,6), (5,6), (6,6), (7,6), (6,3), (7,3), (8,3), (9,3), (6,8), (7,8), (8,8), (9,8)]:
                self.left_walls[cell[0]][cell[1]] = True
            self.depots = {"R": (9,0), "B": (0,4), "G": (9,5), "Y": (1,0), "W": (6,3), "M": (5,6), "C": (9,8), "P": (0,9)}     

class TaxiDomain:

    nav_actions = ["N", "E", "S", "W"]
    actions = {"N", "E", "S", "W", "PICKUP", "PUTDOWN"}

    def __init__(self, grid):
        self.grid = grid

        # initialize locations
        depot_names = list(self.grid.depots.keys())
        selected_depots = random.sample(depot_names, 2)
        # print(selected_depots)
        passenger_loc = grid.depots[selected_depots[0]]
        self.dest_depot = selected_depots[1]
        self.dest_loc = grid.depots[selected_depots[1]]
        taxi_loc = (random.randint(0,grid.size[1]-1), random.randint(0,grid.size[0]-1)) # random.randint(0, 4) returns a random integer in [0, 4]
        
        passenger_in_taxi = False
        self.initial_state = (taxi_loc, passenger_loc, passenger_in_taxi)
        self.state = (taxi_loc, passenger_loc, passenger_in_taxi)
        self.goal_state = (self.dest_loc, self.dest_loc, False)

        self.all_states = set()
        self.state_to_index = {}
        index = -1
        for taxi_loc_row in range(grid.size[0]):
            for taxi_loc_col in range(grid.size[1]):
                for passenger_loc_row in range(grid.size[0]):
                    for passenger_loc_col in range(grid.size[1]):
                        state = ((taxi_loc_row, taxi_loc_col), (passenger_loc_row, passenger_loc_col), False)
                        self.all_states.add(state)
                        index += 1
                        self.state_to_index[state] = index
        for loc_row in range(grid.size[0]):
            for loc_col in range(grid.size[1]):
                state = ((loc_row, loc_col), (loc_row, loc_col), True)
                self.all_states.add(state)
                index += 1
                self.state_to_index[state] = index
        self.num_states = len(self.all_states)

        # stats that are maintained
        self.last_action = ""
        self.last_reward = ""

    def T(self, s, a):
        '''
            INPUT: state s and action a to be taken on this state
            RETURNS: a dictionary of probability distribution of next state s' given s, a
                    probs: {s1': T(s, a, s1'), s2': T(s, a, s2') ...} 
        '''
        probs = {}
        taxi_loc, passenger_loc, passenger_in_taxi = s
        if s == self.goal_state: # no transitions take place from goal state
            return probs

        if a in TaxiDomain.nav_actions:
            # initialize probablility of 0.05 in all directions
            if taxi_loc[0] < 4: # if not at the top most location in grid
                if passenger_in_taxi:
                    passenger_next_loc = (taxi_loc[0]+1, taxi_loc[1])
                else:
                    passenger_next_loc = passenger_loc
                probs[((taxi_loc[0]+1, taxi_loc[1]), passenger_next_loc, passenger_in_taxi)] = 0.05
            else:
                if s in probs:
                    probs[s] += 0.05
                else:
                    probs[s] = 0.05
            if taxi_loc[1] < 4 and not self.grid.right_walls[taxi_loc[0]][taxi_loc[1]]: # if not at the right most location in grid and no wall on right side 
                if passenger_in_taxi:
                    passenger_next_loc = (taxi_loc[0], taxi_loc[1]+1)
                else:
                    passenger_next_loc = passenger_loc
                probs[((taxi_loc[0], taxi_loc[1]+1), passenger_next_loc, passenger_in_taxi)] = 0.05
            else:
                if s in probs:
                    probs[s] += 0.05
                else:
                    probs[s] = 0.05
            if taxi_loc[0] > 0: # if not at the bottom most location in grid
                if passenger_in_taxi:
                    passenger_next_loc = (taxi_loc[0]-1, taxi_loc[1])
                else:
                    passenger_next_loc = passenger_loc
                probs[((taxi_loc[0]-1, taxi_loc[1]), passenger_next_loc, passenger_in_taxi)] = 0.05
            else:
                if s in probs:
                    probs[s] += 0.05
                else:
                    probs[s] = 0.05
            if taxi_loc[1] > 0 and not self.grid.left_walls[taxi_loc[0]][taxi_loc[1]]: # if not at the left most location in grid and no wall on left side
                if passenger_in_taxi:
                    passenger_next_loc = (taxi_loc[0], taxi_loc[1]-1)
                else:
                    passenger_next_loc = passenger_loc
                probs[((taxi_loc[0], taxi_loc[1]-1), passenger_next_loc, passenger_in_taxi)] = 0.05
            else:
                if s in probs:
                    probs[s] += 0.05
                else:
                    probs[s] = 0.05

            # probability 0.85 in direction of a
            if a == "N":
                if taxi_loc[0] < 4: # if not at the top most location in grid
                    if passenger_in_taxi:
                        passenger_next_loc = (taxi_loc[0]+1, taxi_loc[1])
                    else:
                        passenger_next_loc = passenger_loc
                    probs[((taxi_loc[0]+1, taxi_loc[1]), passenger_next_loc, passenger_in_taxi)] = 0.85
                else:
                    if s in probs:
                        probs[s] += 0.85
                    else:
                        probs[s] = 0.85
            elif a == "E":
                if taxi_loc[1] < 4 and not self.grid.right_walls[taxi_loc[0]][taxi_loc[1]]: # if not at the right most location in grid and no wall on right side
                    if passenger_in_taxi:
                        passenger_next_loc = (taxi_loc[0], taxi_loc[1]+1)
                    else:
                        passenger_next_loc = passenger_loc
                    probs[((taxi_loc[0], taxi_loc[1]+1), passenger_next_loc, passenger_in_taxi)] = 0.85
                else:
                    if s in probs:
                        probs[s] += 0.85
                    else:
                        probs[s] = 0.85
            elif a == "S":
                if taxi_loc[0] > 0: # if not at the bottom most location in grid
                    if passenger_in_taxi:
                        passenger_next_loc = (taxi_loc[0]-1, taxi_loc[1])
                    else:
                        passenger_next_loc = passenger_loc
                    probs[((taxi_loc[0]-1, taxi_loc[1]), passenger_next_loc, passenger_in_taxi)] = 0.85
                else:
                    if s in probs:
                        probs[s] += 0.85
                    else:
                        probs[s] = 0.85
            else:
                if taxi_loc[1] > 0 and not self.grid.left_walls[taxi_loc[0]][taxi_loc[1]]: # if not at the left most location in grid and no wall on left side 
                    if passenger_in_taxi:
                        passenger_next_loc = (taxi_loc[0], taxi_loc[1]-1)
                    else:
                        passenger_next_loc = passenger_loc
                    probs[((taxi_loc[0], taxi_loc[1]-1), passenger_next_loc, passenger_in_taxi)] = 0.85
                else:
                    if s in probs:
                        probs[s] += 0.85
                    else:
                        probs[s] = 0.85

        else: # a is PICKUP or PUTDOWN
            if taxi_loc == passenger_loc:
                if a == "PUTDOWN":
                    probs[(taxi_loc, passenger_loc, False)] = 1
                elif a == "PICKUP":
                    probs[(taxi_loc, passenger_loc, True)] = 1
            else:
                probs[s] = 1
        
        return probs

    def R(self, s, a):
        '''
            INPUT: state s and action a to be taken on this state
            RETURNS: reward for this (s, a) tuple
                    (note that reward does not depend on s', i.e., R(s, a, s') = R(s, a) for all s')
        '''
        taxi_loc, passenger_loc, passenger_in_taxi = s
        if s == self.goal_state: # no rewards from goal state as no transitions take place from goal state
            return 0

        if a in TaxiDomain.nav_actions:
            return -1
        else:
            if taxi_loc == passenger_loc:
                if a == "PUTDOWN" and passenger_in_taxi and taxi_loc == self.dest_loc:
                    return 20
                else:
                    return -1
            else:
                return -10

    def take_action(self, s, a):
        '''
            INPUT: state s and action a taken on this state
                    a is in {"N", "E", "S", "W", "PICKUP", "PUTDOWN"}
            RETURNS: samples the next state s' given s, a and returns the tuple (s', R(s, a))
        '''
        next_state_probs = self.T(s, a)
        reward = self.R(s, a)
        if s == self.goal_state: # no action can be taken on goal state
            return (self.goal_state, 0)

        # sample a next state s' from the probability distribution
        next_state = random.choices(list(next_state_probs.keys()), weights=next_state_probs.values(), k=1)[0]
        self.last_action = a
        self.last_reward = reward

        return (next_state, reward)

    def print_state(self):
        '''
            prints the grid, locations of taxi and passenger and relevent stats
        '''
        taxi_loc, passenger_loc, passenger_in_taxi = self.state

        grid = [row[:] for row in self.grid.array]
        message = ""
        if taxi_loc == passenger_loc:
            if passenger_in_taxi:
                grid[taxi_loc[0]][2*taxi_loc[1]] = "O"
                message = "Passenger in taxi at O"
            else:
                grid[taxi_loc[0]][2*taxi_loc[1]] = "T"
                message = "Passenger and taxi at T"
        else:
            grid[taxi_loc[0]][2*taxi_loc[1]] = "T"
            grid[passenger_loc[0]][2*passenger_loc[1]] = "P"
            message = "Passenger at P and taxi at T"

        print()
        for i in range(len(grid)-1, -1, -1):
            row = grid[i]
            for point in row:
                print(point, end=" ")
            print()
        print(f"Action: {self.last_action}  Reward: {self.last_reward}  Dest: {self.dest_depot}")
        print(message)
        
    def value_iteration(self, discount, epsilon):
        u_dash = {state: (0, None) for state in self.all_states}
        #u = {state: 0 for state in self.all_states}
        num_iter=0
        maxnorm=[]
        while True:
            num_iter+=1
            delta=0
            u=dict(u_dash)

            for state in self.all_states:
                q_values=[]
                max=-9999
                best_a=""
                for a in TaxiDomain.actions:
                    action_q_value = self.q_value(state, a, u, discount, True)
                    #print(a, best_a, action_q_value, max)
                    if action_q_value>max:
                        best_a=a
                        max=action_q_value
                    q_values.append(action_q_value)
                optimal_action= np.argmax(q_values)
                #print(u_dash[state], u[state])
                u_dash[state] = (q_values[optimal_action], best_a)
                #print(u_dash[state], u[state])

                if abs(u_dash[state][0]-u[state][0])>delta:
                    #print(abs(u_dash[state]-u[state]))
                    delta = abs(u_dash[state][0]-u[state][0])
                    #print("Delta is "+str(delta)+" in "+str(num_iter)+"th iteration")
            maxnorm.append(delta)
            if delta <= epsilon*(1-discount)/discount:
                return (u, num_iter, maxnorm)

    def q_value(self, state, action, values, discount, bool):
        '''
            RETURNS: Q-value of (state, action) under the value function values
        '''
        if state == self.goal_state: # for any action on the goal state Q(s,a) is taken to be 0
            return 0
        reward = self.R(state, action)
        T_probs = self.T(state, action)
        q_value = 0
        for next_state, prob in T_probs.items():
            if bool==False:
                q_value += prob * (reward + discount * values[next_state])
            else:
                q_value += prob * (reward + discount * values[next_state][0])
        if q_value > 40:
            print(state, action, reward, T_probs, q_value)
        return q_value

    def policy_unchanged(self, pi_dash, pi):
        '''
            RETURNS: True if the policies pi_dash and pi are identical, else False
        '''
        for state in self.all_states:
            if pi_dash[state] != pi[state]:
                return False
        return True

    def value_loss(self, v_dash, v):
        '''
            RETURNS: max-norm distance between v_dash and v
        '''
        max_norm_dist = 0
        for state in self.all_states:
            max_norm_dist = max(max_norm_dist, abs(v_dash[state]-v[state]))
        return max_norm_dist    

    def value_converged(self, v_dash, v, error):
        '''
            RETURNS: True if max-norm distance between v_dash and v is less than error, else False
        '''
        if self.value_loss(v_dash, v) >= error:
            return False
        return True

    def policy_evaluation_iterative(self, policy, error, discount):
        '''
            iteratively finds value function for the given policy
            RETURNS: value function as {state: value} dictionary
        '''
        v = {state: -1 for state in self.all_states}
        v_dash = {state: 0 for state in self.all_states}
        while not self.value_converged(v_dash, v, error):
            v = v_dash
            v_dash = {}
            for state in self.all_states:
                v_dash[state] = self.q_value(state, policy[state], v, discount, False)
        return v_dash

    def policy_evaluation_LA(self, policy, discount):
        '''
            finds value function for the given policy by solving linear algebra equations relating values
            RETURNS: value function as {state: value} dictionary
        '''
        a = np.zeros((self.num_states, self.num_states))
        b = np.zeros(self.num_states)
        for state in self.all_states:
            i = self.state_to_index[state]
            a[i, i] = 1
            if state != self.goal_state:
                T_probs = self.T(state, policy[state])
                for next_state in T_probs:
                    next_state_i = self.state_to_index[next_state]
                    a[i, next_state_i] += -1 * T_probs[next_state] * discount
                b[i] = self.R(state, policy[state])
        print("\t\tSolving linear equations to evaluate policy...")
        x = np.linalg.solve(a, b)
        # print(f"\t\tAssert correct: {np.allclose(np.dot(a, x), b)}")
        values = {}
        for state in self.all_states:
            i = self.state_to_index[state]
            values[state] = x[i] 
        return values

    def policy_improvement(self, policy, values, discount):
        '''
            does a one-step lookahead from values of given policy to obtain improved policy
            RETURNS: improved policy as {state: action} dictionary
        '''
        pi_dash = {}
        for state in self.all_states:
            best_action = None
            best_q_value = None
            for action in TaxiDomain.actions:
                action_q_value = self.q_value(state, action, values, discount, False)
                if best_q_value == None or action_q_value > best_q_value:
                    best_q_value = action_q_value
                    best_action = action
            pi_dash[state] = best_action
        return pi_dash

    def policy_iteration(self, discount, type):
        '''
            finds optimal policy by policy iteration
            INPUT: type 1 uses iterative policy evaluation, type 2 uses linear algebra to evaluate policy
            RETURNS: a tuple (optimal policy, value functions)
                optimal policy is a {state: action} dictionary
                value functions is a list of {state: value} dictionary as policy iteration progresses
        '''
        pi = {state: "None" for state in self.all_states}
        pi_dash = {state: "PICKUP" for state in self.all_states}
        error = 1e-4
        value_functions = []
        iteration_num = 0

        while not self.policy_unchanged(pi_dash, pi) and not (iteration_num!=0 and self.value_converged(v_dash, v, error)):
            # policy iteration converges if policy is unchanged or evaluated value of policy has converged
            print(f"\t\tIteration: {iteration_num}")
            iteration_num += 1
            pi = pi_dash
            print(f"\t\tEvaluating policy...")
            if iteration_num == 1:
                if type == 1:
                    v = self.policy_evaluation_iterative(pi, error, discount)
                elif type == 2:
                    v = self.policy_evaluation_LA(pi, discount)
            else:
                v = v_dash
            value_functions.append(v)
            print(f"\t\tImproving policy (one step lookahead)...")
            pi_dash = self.policy_improvement(pi, v, discount)
            if type == 1:
                v_dash = self.policy_evaluation_iterative(pi_dash, error, discount)
            elif type == 2:
                v_dash = self.policy_evaluation_LA(pi_dash, discount)

        if self.policy_unchanged(pi_dash, pi):
            print("\t\tConverged due to unchanged policy...")
        elif self.value_converged(v_dash, v, error):
            print("\t\tConverged due to converged evaluated value function of policy...")

        v = v_dash # evaluation of optimal policy pi_dash
        value_functions.append(v)
        print(f"\tPolicy Iteration converged in {iteration_num} iterations")
        return (pi_dash, value_functions)

def partA_2a():
    grid = Grid(1)
    tdp = TaxiDomain(grid)
    epsilon = 0.01
    answer = tdp.value_iteration(0.9, epsilon)
    print("Epsilon chosen: "+str(epsilon)+", Total number of iterations: "+str(answer[1]))

def partA_2b():
    grid = Grid(1)
    tdp = TaxiDomain(grid)
    discount_values= [0.01, 0.1, 0.5, 0.8, 0.99]
    max_norm_list=[]
    x=[]
    for discount in discount_values:
        answer = tdp.value_iteration(discount, 0.01)
        max_norm_list.append(answer[2])
        x_local=[]
        for i in range(1, answer[1]+1):
            x_local.append(i)
        x.append(x_local)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    for i in range (len (discount_values)):
        plt.plot(x[i], max_norm_list[i], label="Discount:"+str(discount_values[i]))
    plt.xlabel("Iteration No.")
    plt.ylabel("Max Norm Distance")
    plt.title("Max Norm distance vs. Iteration No. as discount factor varies")
    plt.legend()
    plt.show()

    fig_name = "PartA_2b.png"
    fig.savefig(fig_name, dpi=100)

    print(f"Plot '{fig_name}' generated...")
    
def partA_2c():
    grid = Grid(1)
    discount_values= [0.1, 0.99]
    #action_list = ["S", "W", "N", "E", "PICKUP", "PUTDOWN"]
    for discount in discount_values:
        tdp = TaxiDomain(grid)
        print("\n\nRunning for discount="+str(discount)+"\n")
        answer = tdp.value_iteration(discount, 0.01)
        tdp.print_state()
        #print(answer[0])
        for i in range (20):
            print(f"\nStep {i}")
            action = answer[0][tdp.state][1]
            tdp.state, reward = tdp.take_action(tdp.state, action)
            tdp.print_state()
            if tdp.state == tdp.goal_state:
                break

def partA_3b(type):
    '''
        type 1 uses iterative policy evaluation, type 2 uses linear algebra to evaluate policy
    '''
    print("\nRunning Policy Iteration for different discount factors...")

    grid = Grid(1)
    tdp = TaxiDomain(grid)
    discount_values = [0.01, 0.1, 0.5, 0.8, 0.99] # 0.01, 0.1, 0.5, 0.8, 0.99
    policy_losses_list = []
    for discount in discount_values:
        print(f"\tRunning for discount={discount}")
        opt_policy, value_functions = tdp.policy_iteration(discount, type)
        opt_vf = value_functions[-1]
        policy_losses = []
        for vf in value_functions[:-1]:
            policy_loss = tdp.value_loss(vf, opt_vf)
            policy_losses.append(policy_loss)
        policy_losses_list.append(policy_losses)
    # print(opt_vf)
    # print(policy_losses_list)

    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    for i in range(len(discount_values)):
        discount = discount_values[i]
        policy_losses = policy_losses_list[i]
        plt.plot(list(range(len(policy_losses))), policy_losses, label=f"Discount: {discount}")
    plt.xlabel("Iteration No.")
    plt.ylabel("Policy Loss from Optimal Policy")
    plt.title(f"Policy Loss vs. Iteration No. as discount factor varies [Type {type}]")
    plt.legend()
    plt.show()

    fig_name = f"PartA_3b_type{type}.png"
    fig.savefig(fig_name, dpi=100)

    print(f"Plot '{fig_name}' generated...")


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
            # print(f"Episode no.: {episode}")
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
            # if episode == 20:
            #    print(len(self.q_values), self.q_values)
            #    return
            if compute_rewards and episode!=0 and episode%20 == 0:
                # print(f"\nEvaluating episode no.: {episode}\n")
                policy = self.extract_policy(self.q_values)
                dr = self.compute_discounted_rewards(policy, 20)
                discounted_rewards.append(dr)
                average_discounted_rewards.append(sum(dr)/len(dr))
                episode_nums.append(episode)
        # evaluate last time
        # print(f"\nEvaluating episode no.: {num_episodes-1}\n")
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
        test_tdp.last_action = ""
        test_tdp.last_reward = ""
        # run this episode
        num_step = 0
        print("Pasenger: ", passenger_loc)
        print("Taxi: ", taxi_loc)
        acc_reward=0.0
        while not test_tdp.state == test_tdp.goal_state and num_step < 500:
            num_step += 1
            #print(f"\nStep No.: {num_step}")
            #test_tdp.print_state()
            action = policy[test_tdp.state]
            next_state, reward = test_tdp.take_action(test_tdp.state, action)
            acc_reward+=reward*(gamma**(num_step-1))
            test_tdp.state = next_state
        # print last state
        num_step += 1
        print("Total Reward: ", acc_reward)
        print(f"\nStep No.: {num_step}")
        test_tdp.print_state()

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
    # We are the best learning and exploration rates here
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
    # tdp.print_state()
    # tdp.state, reward = tdp.take_action(tdp.state, "N")
    # tdp.print_state()
    # tdp.state, reward = tdp.take_action(tdp.state, "N")
    # tdp.print_state()
    # tdp.state, reward = tdp.take_action(tdp.state, "E")
    # tdp.print_state()
    # partA_2a()
    # partA_2b()
    # partA_2c()
    # s1 = time.process_time()
    # partA_3b(1)
    # t1 = time.process_time() - s1
    # s2 = time.process_time()
    # partA_3b(2)
    # t2 = time.process_time() - s2
    # print(f"Time taken in Part A 3(b) using iterative policy evaluation = {t1} s")
    # print(f"Time taken in Part A 3(b) using linear algebra policy evaluation = {t2} s")
    # grid = Grid(1)
    # tdp = TaxiDomain(grid)
    # ql = Q_Learning(tdp, 0.25, 0.99, 0.1, False)
    # ql.learn(2000)
    # partB_2()
    # partB_3()
    # partB_4()
    partB_5()
