import random
import numpy as np

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

class TaxiDomain:

    nav_actions = ["N", "E", "S", "W"]

    def __init__(self, grid):
        self.grid = grid

        # initialize locations
        depot_names = ["R", "B", "G", "Y"]
        selected_depots = random.sample(depot_names, 2)
        # print(selected_depots)
        passenger_loc = grid.depots[selected_depots[0]]
        self.dest_depot = selected_depots[1]
        self.dest_loc = grid.depots[selected_depots[1]]
        taxi_loc = (random.randint(0,4), random.randint(0,4)) # random.randint(0, 4) returns a random integer in [0, 4]
        
        passenger_in_taxi = False
        self.state = (taxi_loc, passenger_loc, passenger_in_taxi)

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

        if a in TaxiDomain.nav_actions:
            # initialize probablility of 0.05 in all directions
            if taxi_loc[0] < 4: # if not at the top most location in grid
                probs[((taxi_loc[0]+1, taxi_loc[1]), passenger_loc, passenger_in_taxi)] = 0.05
            else:
                if s in probs:
                    probs[s] += 0.05
                else:
                    probs[s] = 0.05
            if taxi_loc[1] < 4 and not self.grid.right_walls[taxi_loc[0]][taxi_loc[1]]: # if not at the right most location in grid and no wall on right side 
                probs[((taxi_loc[0], taxi_loc[1]+1), passenger_loc, passenger_in_taxi)] = 0.05
            else:
                if s in probs:
                    probs[s] += 0.05
                else:
                    probs[s] = 0.05
            if taxi_loc[0] > 0: # if not at the bottom most location in grid
                probs[((taxi_loc[0]-1, taxi_loc[1]), passenger_loc, passenger_in_taxi)] = 0.05
            else:
                if s in probs:
                    probs[s] += 0.05
                else:
                    probs[s] = 0.05
            if taxi_loc[1] > 0 and not grid.left_walls[taxi_loc[0]][taxi_loc[1]]: # if not at the left most location in grid and no wall on left side 
                probs[((taxi_loc[0], taxi_loc[1]-1), passenger_loc, passenger_in_taxi)] = 0.05
            else:
                if s in probs:
                    probs[s] += 0.05
                else:
                    probs[s] = 0.05

            # probability 0.85 in direction of a
            if a == "N":
                if taxi_loc[0] < 4: # if not at the top most location in grid
                    probs[((taxi_loc[0]+1, taxi_loc[1]), passenger_loc, passenger_in_taxi)] = 0.85
                else:
                    if s in probs:
                        probs[s] += 0.85
                    else:
                        probs[s] = 0.85
            elif a == "E":
                if taxi_loc[1] < 4 and not self.grid.right_walls[taxi_loc[0]][taxi_loc[1]]: # if not at the right most location in grid and no wall on right side 
                    probs[((taxi_loc[0], taxi_loc[1]+1), passenger_loc, passenger_in_taxi)] = 0.85
                else:
                    if s in probs:
                        probs[s] += 0.85
                    else:
                        probs[s] = 0.85
            elif a == "S":
                if taxi_loc[0] > 0: # if not at the bottom most location in grid
                    probs[((taxi_loc[0]-1, taxi_loc[1]), passenger_loc, passenger_in_taxi)] = 0.85
                else:
                    if s in probs:
                        probs[s] += 0.85
                    else:
                        probs[s] = 0.85
            else:
                if taxi_loc[1] > 0 and not grid.left_walls[taxi_loc[0]][taxi_loc[1]]: # if not at the left most location in grid and no wall on left side 
                    probs[((taxi_loc[0], taxi_loc[1]-1), passenger_loc, passenger_in_taxi)] = 0.85
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
        delta=0
        #states list? Transition T?
        u_dash = np.zeros((5,5))
        while delta > epsilon*(1-discount)/discount:
            delta=0
            u=np.copy(u_dash)
            for state in self.states:
                q_values=[]
                for a in self.actions:
                    q=0
                    for (probaility, state_next) in T(state_next, a):
                        q+=probability*(self.last_reward + discount*u[state_next])
                    q_values.append(q)
                optimal_action= np.argmax(q_values)
                u_dash[state] = q_values[optimal_action]
                if abs(u_dash[state]-u[state])>delta:
                    delta = abs(u_dash[state]-u[state])

        return u


    #      

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

    # partA_1b()
    # partA_1c()
