import numpy as np
from queue import PriorityQueue

class Astar():

    def __init__(self, map,drone_pos, wounded_pos):

        self.map = map
        self.height = map.shape[0]
        self.width = map.shape[1]
        self.grid_value = []
        self.start = drone_pos
        self.goal = wounded_pos


    def grid(self):

        dict = {}
        map = self.map

        for x in range(self.width):
            for y in range(self.height):
                dict[(x, y)] = {'E': 0, 'W': 0, 'N': 0, 'S': 0}
                self.grid_value.append((x, y))


        for x in range(6, self.width - 6):

            for y in range(6, self.height - 6):

                if map[y, x] == 0:

                    cw = 0
                    for i in range(1, 6):
                        if map[y, x - i] == 0 and map[y - i, x] == 0 and map[y + i, x] == 0:
                            cw += 1
                    for i in range(1, 6):
                        if map[y + i, x - 5] == 0 and map[y - i, x - 5] == 0:
                            cw += 1
                    if cw == 10:
                        dict[(x, y)]['W'] = 1

                    cn = 0
                    for i in range(1, 6):
                        if map[y, x - i] == 0 and map[y - i, x] == 0 and map[y, x + i] == 0:
                            cn += 1
                    for i in range(1, 6):
                        if map[y - 5, x - i] == 0 and map[y - 5, x + i] == 0:
                            cn += 1
                    if cn == 10:
                        dict[(x, y)]['N'] = 1

                    cs = 0
                    for i in range(1, 6):
                        if map[y, x - i] == 0 and map[y, x + i] == 0 and map[y + i, x] == 0:
                            cs += 1
                    for i in range(1, 6):
                        if map[y + 5, x - i] == 0 and map[y + 5, x + i] == 0:
                            cs += 1
                    if cs == 10:
                        dict[(x, y)]['S'] = 1

                    ce = 0
                    for i in range(1, 6):
                        if map[y, x + i] == 0 and map[y - i, x] == 0 and map[y + i, x] == 0:
                            ce += 1
                    for i in range(1, 6):
                        if map[y + i, x + 5] == 0 and map[y - i, x + 5] == 0:
                            ce += 1
                    if ce == 10:
                        dict[(x, y)]['E'] = 1

        return dict

    def h(self, cell1,cell2):

        x1,y1 = cell1
        x2,y2 = cell2

        return abs(x1-x2) + abs(y1-y2)

    def aStar(self):

        start = self.start
        goal = self.goal
        dict = self.grid()

        g_score = {cell : float('inf') for cell in self.grid_value}
        g_score[start] = 0

        f_score = {cell: float('inf') for cell in self.grid_value}
        f_score[start] = self.h(start, goal)

        open = PriorityQueue()
        open.put((self.h(start, goal),self.h(start, goal),start))
        aPath = {}

        while not open.empty():

            currCell = open.get()[2]

            if currCell == goal :
                break

            for d in 'ESNW' :

                if dict[currCell][d] == 1 :
                    if d == 'E':
                        childCell = (currCell[0]+1,currCell[1])

                    if d == 'S':
                        childCell = (currCell[0],currCell[1]+1)

                    if d == 'N':
                        childCell = (currCell[0],currCell[1]-1)

                    if d == 'W':
                        childCell = (currCell[0]-1,currCell[1])

                    temp_g_score = g_score[currCell]+1
                    temp_f_score = temp_g_score+self.h(childCell,goal)

                    if temp_f_score < f_score[childCell]:

                        g_score[childCell] = temp_g_score
                        f_score[childCell] = temp_f_score
                        open.put((temp_f_score,self.h(childCell,goal),childCell))
                        aPath[childCell] = currCell

        fwdPath = {}
        cell = goal

        while cell!= start :

            fwdPath[aPath[cell]] = cell
            cell = aPath[cell]

        return fwdPath

# if __name__ == '__main__':
