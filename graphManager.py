import sys

import heapq

'''This vertex and graph creation was expanded from the Dijkstra tutorial here:
https://www.bogotobogo.com/python/python_Dijkstras_Shortest_Path_Algorithm.php '''

class Vertex:
    def __init__(self, node):
        self.name = node
        self.adjacent = {}
        # Set distance to infinity for all nodes
        self.distance = float("inf")
        # Mark all nodes unvisited
        self.visited = False
        # Predecessor
        self.previous = None
        self.data = None
        self.neighbor = {}

    # def __eq__(self, other):
    #    if isinstance(other, self.__class__):
    #        return self.distance == other.distance
    #    return NotImplemented

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.distance < other.distance
        return NotImplemented

    def addData(self, data):
        self.data = data

    def addNeighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def getConnections(self):
        return self.adjacent.keys()

    def getName(self):
        return self.name

    def getWeight(self, neighbor):
        return self.adjacent[neighbor]

    def setDistance(self, dist):
        self.distance = dist

    def getDirections(self, neighbor):
        return self.neighbor[neighbor]

    def getDistance(self):
        return self.distance

    def setPrevious(self, prev):
        self.previous = prev

    def setVisited(self):
        self.visited = True

    def __str__(self):
        return str(self.name) + ' adjacent: ' + str([x.id for x in self.adjacent])


class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def addVertex(self, node):
        vertex = self.getVertex(node)
        if vertex:
            return vertex
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def getVertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def addEdge(self, frm, to, cost=0):
        if frm not in self.vert_dict:
            self.addVertex(frm)
        if to not in self.vert_dict:
            self.addVertex(to)

        self.vert_dict[frm].addNeighbor(self.vert_dict[to], cost)
        # self.vert_dict[to].addNeighbor(self.vert_dict[frm], cost)

    def getVertices(self):
        return self.vert_dict.keys()

    def setPrevious(self, current):
        self.previous = current

    def getPrevious(self, current):
        return self.previous

    def shortestPath(self, start, target):
        dijkstra(self, self.getVertex(start), self.getVertex(target))

    def shortest(self, v, path):
        ''' make shortest path from v.previous'''
        if v.previous:
            path.append(v.previous.getName())
            self.shortest(v.previous, path)
        return


def dijkstra(aGraph, start, target):
    print
    '''Dijkstra's shortest path'''
    # Set the distance for the start node to zero
    start.setDistance(0)

    # Put tuple pair into the priority queue
    unvisited_queue = [(v.getDistance(), v) for v in aGraph]
    heapq.heapify(unvisited_queue)

    while len(unvisited_queue):
        # Pops a vertex with the smallest distance
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        current.setVisited()

        # for next in v.adjacent:
        for next in current.adjacent:
            # if visited, skip
            if next.visited:
                continue
            new_dist = current.getDistance() + current.getWeight(next)

            if new_dist < next.getDistance():
                next.setDistance(new_dist)
                next.setPrevious(current)
            else:
                pass
        # Rebuild heap
        # 1. Pop every item
        while len(unvisited_queue):
            heapq.heappop(unvisited_queue)
        # 2. Put all vertices not visited into the queue
        unvisited_queue = [(v.getDistance(), v) for v in aGraph if not v.visited]
        heapq.heapify(unvisited_queue)


