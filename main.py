import random
import numpy as np
import matplotlib.pyplot as plt


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
        u_dash = {state: 0 for state in self.all_states}
        #u = {state: 0 for state in self.all_states}
        num_iter=0
        maxnorm=[]
        while True:
            num_iter+=1
            delta=0
            u=dict(u_dash)

            for state in self.all_states:
                q_values=[]
                for a in TaxiDomain.actions:
                    action_q_value = self.q_value(state, a, u, discount)
                    q_values.append(action_q_value)
                optimal_action= np.argmax(q_values)
                #print(u_dash[state], u[state])
                u_dash[state] = q_values[optimal_action]
                #print(u_dash[state], u[state])

                if abs(u_dash[state]-u[state])>delta:
                    #print(abs(u_dash[state]-u[state]))
                    delta = abs(u_dash[state]-u[state])
                    #print("Delta is "+str(delta)+" in "+str(num_iter)+"th iteration")
            maxnorm.append(delta)
            if delta <= epsilon*(1-discount)/discount:
                return (u, num_iter, maxnorm)

    def q_value(self, state, action, values, discount):
        '''
            RETURNS: Q-value of (state, action) under the value function values
        '''
        if state == self.goal_state: # for any action on the goal state Q(s,a) is taken to be 0
            return 0
        reward = self.R(state, action)
        T_probs = self.T(state, action)
        q_value = 0
        for next_state, prob in T_probs.items():
            q_value += prob * (reward + discount * values[next_state])
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
                v_dash[state] = self.q_value(state, policy[state], v, discount)
        return v_dash

    def policy_evaluation_LA(self, policy, discount):
        '''
            finds value function for the given policy by solving linear algebra equations relating values
            RETURNS: value function as {state: value} dictionary
        '''
        a = np.zeros((self.num_states, self.num_states))
        b = np.zeros(self.num_states)
        row_num = -1
        for state in self.all_states:
            i = self.state_to_index[state]
            a[i, i] = 1
            if state != self.goal_state:
                T_probs = self.T(state, policy[state])
                for next_state in T_probs:
                    next_state_i = self.state_to_index[next_state]
                    a[i, next_state_i] = -1 * T_probs[next_state] * discount
                b[i] = self.R(state, policy[state])
        print("\tSolving linear equations to evaluate new policy...")
        x = np.linalg.solve(a, b)
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
        pi_dash = {state: "N" for state in self.all_states}
        for state in self.all_states:
            best_action = None
            best_q_value = None
            for action in TaxiDomain.actions:
                action_q_value = self.q_value(state, action, values, discount)
                if best_q_value == None or action_q_value > best_q_value:
                    best_q_value = action_q_value
                    best_action = action
            pi_dash[state] = best_action
        return pi_dash

    def policy_iteration(self, discount):
        '''
            finds optimal policy by policy iteration
            RETURNS: a tuple (optimal policy, value functions)
                optimal policy is a {state: action} dictionary
                value functions is a list of {state: value} dictionary as policy iteration progresses
        '''
        pi = {state: "None" for state in self.all_states}
        pi_dash = {state: "N" for state in self.all_states}
        error = 0.01
        value_functions = []
        iteration_num = 0

        while not self.policy_unchanged(pi_dash, pi):
            iteration_num += 1
            pi = pi_dash
            v = self.policy_evaluation_iterative(pi, error, discount)
            # v = self.policy_evaluation_LA(pi, discount)
            value_functions.append(v)
            pi_dash = self.policy_improvement(pi, v, discount)
        v = self.policy_evaluation_iterative(pi_dash, error, discount)
        # v = self.policy_evaluation_LA(pi_dash, discount)
        value_functions.append(v)
        print(f"\tPolicy Iteration converged in {iteration_num} iterations")
        return (pi_dash, value_functions)

def partA_2a():
    grid = Grid(1)
    tdp = TaxiDomain(grid)
    epsilon = 0.5
    answer = tdp.value_iteration(0.9, epsilon)
    #print (answer[0])
    print ("Epsilon chosen: "+str(epsilon)+", total number of iterations: "+str(answer[1]))

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
    
def partA_3b():
    print("\nRunning Policy Iteration for different discount factors...")

    grid = Grid(1)
    tdp = TaxiDomain(grid)
    discount_values = [0.01, 0.1, 0.5, 0.8] # 0.99
    policy_losses_list = []
    for discount in discount_values:
        print(f"\tRunning for discount={discount}")
        opt_policy, value_functions = tdp.policy_iteration(discount)
        opt_vf = value_functions[-1]
        policy_losses = []
        for vf in value_functions[:-1]:
            policy_loss = tdp.value_loss(vf, opt_vf)
            policy_losses.append(policy_loss)
        policy_losses_list.append(policy_losses)
    # print(policy_losses_list)

    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    for i in range(len(discount_values)):
        discount = discount_values[i]
        policy_losses = policy_losses_list[i]
        plt.plot(list(range(len(policy_losses))), policy_losses, label=f"Discount: {discount}")
    plt.xlabel("Iteration No.")
    plt.ylabel("Policy Loss from Optimal Policy")
    plt.title("Policy Loss vs. Iteration No. as discount factor varies")
    plt.legend()
    plt.show()

    fig_name = "PartA_3b.png"
    fig.savefig(fig_name, dpi=100)

    print(f"Plot '{fig_name}' generated...")



if __name__ == "__main__":
    grid = Grid(1)
    tdp = TaxiDomain(grid)
    tdp.print_state()
    tdp.state, reward = tdp.take_action(tdp.state, "N")
    tdp.print_state()
    tdp.state, reward = tdp.take_action(tdp.state, "N")
    tdp.print_state()
    tdp.state, reward = tdp.take_action(tdp.state, "E")
    tdp.print_state()
    partA_2a()
    partA_2b()
    partA_3b()

