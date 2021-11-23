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
            for cell in [(0,0), (0,1), (2,0), (2,1), (1,3), (1,4)]:
                self.right_walls[cell[0]][cell[1]] = True
            self.left_walls = [[False for i in range(5)] for i in range(5)] # True for grid cells that have walls on their left side
            for cell in [(1,0), (1,1), (3,0), (3,1), (2,3), (2,4)]:
                self.left_walls[cell[0]][cell[1]] = True
            # note that we model only vertical walls as there are no horizontal walls
            self.depots = {"R": [0,4], "B": [3,0], "G": [4,4], "Y": [0,0]}
        elif type == 2:
            self.size = (10, 10)        

class TaxiDomain:

    def __init__(self, grid):
        self.grid = grid

        # initialize locations
        depot_names = ["R", "B", "G", "Y"]
        selected_depots = random.sample(depot_names, 2)
        # print(selected_depots)
        self.passenger_loc = grid.depots[selected_depots[0]]
        self.dest_loc = grid.depots[selected_depots[1]]
        self.taxi_loc = [random.randint(0,4), random.randint(0,4)] # random.randint(0, 4) returns a random integer in [0, 4]
        
        self.passenger_in_taxi = False
        # state is defined as (self.taxi_loc, self.passenger_loc, self.passenger_in_taxi)

        # stats that are maintained
        self.last_action = ""
        self.last_result = ""
        self.last_reward = ""

    def take_action(self, a):
        '''
            INPUT: action a represents action to be taken by the taxi agent 
                    a is in {"N", "E", "S", "W", "PICKUP", "PUTDOWN"}
            RETURNS: updates self.state and returns reward according to transition and reward model
        '''
        self.last_action = a
        self.last_result = ""
        self.last_reward = -1
        nav_actions = ["N", "E", "S", "W"]

        if a in nav_actions:
            sample = random.uniform(0,1)
            if sample >= 0.15: # move in direction of a with probability 0.85
                final_direction = a
            else:
                nav_actions.remove(a)
                # print(nav_actions)
                final_direction = random.choice(nav_actions)
            self.last_result = final_direction

            # execute final_action
            if final_direction == "N":
                if self.taxi_loc[0] < 4: # if not at the top most location in grid
                    self.taxi_loc[0] += 1
            elif final_direction == "E":
                if self.taxi_loc[1] < 4 and not self.grid.right_walls[self.taxi_loc[0]][self.taxi_loc[1]]: # if not at the right most location in grid and no wall on right side 
                    self.taxi_loc[1] += 1
            elif final_direction == "S":
                if self.taxi_loc[0] > 0: # if not at the bottom most location in grid
                    self.taxi_loc[0] -= 1
            else:
                if self.taxi_loc[1] > 0 and not self.grid.left_walls[self.taxi_loc[0]][self.taxi_loc[1]]: # if not at the left most location in grid and no wall on left side 
                    self.taxi_loc[1] -= 1
            if self.passenger_in_taxi:
                self.passenger_loc = self.taxi_loc
            return -1

        else:
            if self.taxi_loc == self.passenger_loc:
                if a == "PUTDOWN" and self.passenger_in_taxi and self.taxi_loc == self.dest_loc:
                    self.last_result = "FINISH"
                    self.last_reward = 20
                    return 20
                if a == "PICKUP":
                    if not self.passenger_in_taxi:
                        self.last_result = "PICKUP"
                        self.passenger_in_taxi = True
                    else:
                        self.last_result = "NONE"
                else: # a == "PUTDOWN"
                    if not self.passenger_in_taxi:
                        self.last_result = "NONE"
                    else:
                        self.last_result = "PUTDOWN"
                return -1
            else:
                self.last_result = "FAIL"
                self.last_reward = -10
                return -10

    def print_state(self):
        '''
            prints the grid, locations of taxi and passenger and relevent stats
        '''
        grid = [row[:] for row in self.grid.array]
        message = ""
        if self.taxi_loc == self.passenger_loc:
            if passenger_in_taxi:
                grid[self.taxi_loc[0]][2*self.taxi_loc[1]] = "O"
                message = "Passenger in taxi at O"
            else:
                grid[self.taxi_loc[0]][2*self.taxi_loc[1]] = "T"
                message = "Passenger and taxi at T"
        else:
            grid[self.taxi_loc[0]][2*self.taxi_loc[1]] = "T"
            grid[self.passenger_loc[0]][2*self.passenger_loc[1]] = "P"
            message = "Passenger at P and taxi at T"

        print()
        for i in range(len(grid)-1, -1, -1):
            row = grid[i]
            for point in row:
                print(point, end=" ")
            print()
        print(f"Action: {self.last_action}  Result: {self.last_result}  Reward: {self.last_reward}")
        print(message)
        
    # def partA_1b():
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


    # def partA_1c():        

if __name__ == "__main__":
    grid = Grid(1)
    tdp = TaxiDomain(grid)
    tdp.print_state()
    tdp.take_action("N")
    tdp.print_state()
    tdp.take_action("N")
    tdp.print_state()
    tdp.take_action("E")
    tdp.print_state()

    # partA_1b()
    # partA_1c()
