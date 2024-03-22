from abc import abstractmethod, ABC
from typing import Tuple

from pdm4ar.exercises.ex02.structures import AdjacencyList, X, Path, OpenedNodes


class GraphSearch(ABC):
    @abstractmethod
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        """
        :param graph: The given graph as an adjacency list
        :param start: The initial state (i.e. a node)
        :param goal: The goal state (i.e. a node)
        :return: The path from start to goal as a Sequence of states, None if a path does not exist
        """
        pass


class DepthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        # initialisation of search query Q, visited nodes V, opened nodes opened_nodes 
        # parent dictionary that includes the parent(s) for each node
        opened_nodes = []
        Q = [start]
        V = set()
        V.add(start)
        parent = {start: None}
        
        # run depth search as long as Q is not empty
        while Q:
            s = Q.pop(0)
            opened_nodes += [s]

            # if goal has been reached then get the path from start to goal
            if s == goal:
                if s not in V:
                    V.add(s)

                # extract the path
                p = parent[goal]
                path = [goal]
                while p is not None:
                    path.insert(0, p)
                    p = parent[p]
                return path, opened_nodes

            # sort neighbours
            neighbours = list(graph.get(s))
            neighbours.sort(reverse=True)
            # go through neighbours and add those to the front
            # of the quers, that have not been visited yet
            for n in neighbours:
                if n not in V:
                    Q.insert(0, n)
                    V.add(n)
                    parent[n] = s

        return [], opened_nodes
            


class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        # initialisation of search query Q, visited nodes V, opened nodes opened_nodes 
        # parent dictionary that includes the parent(s) for each node
        opened_nodes = []
        Q = [start]
        V = set()
        V.add(start)
        parent = {start: None}
        
        # run breadth search as long as Q is not empty
        while Q:
            s = Q.pop(0)
            opened_nodes += [s]

            # if goal has been reached then get the path from start to goal
            if s == goal:
                if s not in V:
                    V.add(s)

                # extract the path
                p = parent[goal]
                path = [goal]
                while p is not None:
                    path.insert(0, p)
                    p = parent[p]
                return path, opened_nodes
            
            # sort neighbours
            neighbours = list(graph.get(s))
            neighbours.sort()
            # go through neighbours and append those to the query
            # that have not been visited yet
            for n in neighbours:
                if n not in V:
                    Q += [n]
                    V.add(n)
                    parent[n] = s

        return [], opened_nodes


class IterativeDeepening(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        # setting MAX_DEPTH to the length of the graph
        # there's probably a better way to solve this
        MAX_DEPTH = len(graph)
        opened_nodes = []

        # I use MAX_DEPTH + 1 as upperbound to ensure that the trivial graph is included
        for d in range(1, MAX_DEPTH + 1):
            # initialisation of search query Q which includes also the depth of query node, 
            # visited nodes V, opened nodes opened_nodes 
            # parent dictionary that includes the parent(s) for each node
            Q = [{start: 1}]
            V = set()
            V.add(start)
            opened_nodes = []
            parent = {start: None}

            current_max_depth = d
            # run depth search as long as Q is not empty
            while Q:
                s, current_depth = next(iter(Q.pop(0).items()))
                opened_nodes += [s]

                # if goal has been reached then get the path from start to goal
                if s == goal:
                    if s not in V:
                        V.add(s)

                    # extract the path
                    p = parent[goal]
                    path = [goal]
                    while p is not None:
                        path.insert(0, p)
                        p = parent[p]
                    return path, opened_nodes

                # sort neighbours
                neighbours = list(graph.get(s))
                neighbours.sort(reverse=True)

                # go through neighbours only if current max depth has not been reached yet
                if current_depth <= current_max_depth:
                    for n in neighbours:
                        if n not in V:
                            Q.insert(0, {n: current_depth + 1})
                            V.add(n)
                            parent[n] = s

        return [], opened_nodes