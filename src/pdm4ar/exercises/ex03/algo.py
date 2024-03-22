from abc import ABC, abstractmethod
from dataclasses import dataclass
import heapq    # you may find this helpful

from osmnx.distance import great_circle_vec

from pdm4ar.exercises.ex02.structures import X, Path
from pdm4ar.exercises.ex03.structures import WeightedGraph, TravelSpeed


@dataclass
class InformedGraphSearch(ABC):
    graph: WeightedGraph

    @abstractmethod
    def path(self, start: X, goal: X) -> Path:
        # Abstract function. Nothing to do here.
        pass

@dataclass
class UniformCostSearch(InformedGraphSearch):
    def path(self, start: X, goal: X) -> Path:
        # initilisation
        Q = [(0, start)]
        cost_to_reach = {start: 0}
        parent = {start: None}

        while Q:
            s = heapq.heappop(Q)[1]

            if s == goal:
                # reconstruct the path
                path = [goal]
                p = parent[goal]
                while p is not None:
                    path.append(p)
                    p = parent[p]
                
                return path[::-1]
            
            # go through neighbours
            neighbours = self.graph.adj_list.get(s)
            for neighbour in neighbours:
                # calculate new cost to reach
                new_cost_to_reach = cost_to_reach.get(s) + self.graph.get_weight(s, neighbour)
                
                # check if there exists already a cost_to_reach value for the neighbour
                # if not set it to infinity
                cost_to_reach_elem = cost_to_reach.get(neighbour)
                if cost_to_reach_elem is None:
                    cost_to_reach[neighbour] = float('inf')
                    cost_to_reach_elem = float('inf')
                
                # update current_cost_to_reach with new one if the new is smaller than the current one
                if new_cost_to_reach < cost_to_reach_elem:
                        cost_to_reach[neighbour] = new_cost_to_reach
                        parent[neighbour] = s
                        heapq.heappush(Q, (new_cost_to_reach, neighbour))


        return []

@dataclass
class Astar(InformedGraphSearch):

    # Keep track of how many times the heuristic is called
    heuristic_counter: int = 0
    # Allows the tester to switch between calling the students heuristic function and
    # the trivial heuristic (which always returns 0). This is a useful comparison to
    # judge how well your heuristic performs.
    use_trivial_heuristic: bool = False

    def heuristic(self, u: X, v: X) -> float:
        # Increment this counter every time the heuristic is called, to judge the performance
        # of the algorithm
        self.heuristic_counter += 1
        if self.use_trivial_heuristic:
            return 0
        else:
            # return the heuristic that the student implements
            return self._INTERNAL_heuristic(u, v)

    # Implement the following two functions

    def _INTERNAL_heuristic(self, u: X, v: X) -> float:
        # Implement your heuristic here. Your `path` function should NOT call
        # this function directly. Rather, it should call `heuristic`
        lon1, lat1 = self.graph.get_node_coordinates(u)
        lon2, lat2 = self.graph.get_node_coordinates(v)
        
        return great_circle_vec(lat1, lon1, lat2, lon2) / TravelSpeed.HIGHWAY.value
        
    def path(self, start: X, goal: X) -> Path:
        # initialisation
        Q = [(0, start)]
        cost_to_reach = {start: 0}
        parent = {start: None}

        while Q:
            s = heapq.heappop(Q)[1]

            if s == goal:
                # reconstruct the path
                path = [goal]
                p = parent[goal]
                while p is not None:
                    path.append(p)
                    p = parent[p]
                
                return path[::-1]
            
            # iterate through neighbours
            neighbours = self.graph.adj_list.get(s)
            for neighbour in neighbours:
                # calculate new cost to reach
                new_cost_to_reach = cost_to_reach.get(s) + self.graph.get_weight(s, neighbour)
                
                # check if there exists already a cost_to_reach value for the neighbour
                # if not set it to infinity
                cost_to_reach_elem = cost_to_reach.get(neighbour)
                if cost_to_reach_elem is None:
                    cost_to_reach[neighbour] = float('inf')
                    cost_to_reach_elem = float('inf')
                
                # update current_cost_to_reach with new one if the new is smaller than the current one
                if new_cost_to_reach < cost_to_reach_elem:
                        cost_to_reach[neighbour] = new_cost_to_reach
                        parent[neighbour] = s
                        # add the heuristic to the new_cost_to_reach
                        heapq.heappush(Q, (new_cost_to_reach + self.heuristic(neighbour, goal), neighbour))

        return []


def compute_path_cost(wG: WeightedGraph, path: Path):
    """A utility function to compute the cumulative cost along a path"""
    if not path:
        return float("inf")
    total: float = 0
    for i in range(1, len(path)):
        inc = wG.get_weight(path[i - 1], path[i])
        total += inc
    return total
