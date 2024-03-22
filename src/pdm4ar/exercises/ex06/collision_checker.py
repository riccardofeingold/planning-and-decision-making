from typing import List
from dg_commons import SE2Transform
from pdm4ar.exercises.ex06.collision_primitives import CollisionPrimitives
from pdm4ar.exercises_def.ex06.structures import (
    Polygon,
    GeoPrimitive,
    Point,
    Segment,
    Circle,
    Triangle,
    Path,
)
import shapely
import numpy as np

##############################################################################################
############################# This is a helper function. #####################################
# Feel free to use this function or not

COLLISION_PRIMITIVES = {
    Point: {
        Circle: lambda x, y: CollisionPrimitives.circle_point_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_point_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_point_collision(y, x),
    },
    Segment: {
        Circle: lambda x, y: CollisionPrimitives.circle_segment_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_segment_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_segment_collision_aabb(y, x),
    },
    Triangle: {
        Point: CollisionPrimitives.triangle_point_collision,
        Segment: CollisionPrimitives.triangle_segment_collision,
    },
    Circle: {
        Point: CollisionPrimitives.circle_point_collision,
        Segment: CollisionPrimitives.circle_segment_collision,
    },
    Polygon: {
        Point: CollisionPrimitives.polygon_point_collision,
        Segment: CollisionPrimitives.polygon_segment_collision_aabb,
    },
}


def check_collision(p_1: GeoPrimitive, p_2: GeoPrimitive) -> bool:
    """
    Checks collision between 2 geometric primitives
    Note that this function only uses the functions that you implemented in CollisionPrimitives class.
        Parameters:
                p_1 (GeoPrimitive): Geometric Primitive
                p_w (GeoPrimitive): Geometric Primitive
    """
    assert type(p_1) in COLLISION_PRIMITIVES, "Collision primitive does not exist."
    assert (
        type(p_2) in COLLISION_PRIMITIVES[type(p_1)]
    ), "Collision primitive does not exist."

    collision_func = COLLISION_PRIMITIVES[type(p_1)][type(p_2)]

    return collision_func(p_1, p_2)


##############################################################################################
############################# This is a helper function. #####################################
# Feel free to use this function or not

def geo_primitive_to_shapely(p: GeoPrimitive):
    if isinstance(p, Point):
        return shapely.Point(p.x, p.y)
    elif isinstance(p, Segment):
        return shapely.LineString([[p.p1.x, p.p1.y], [p.p2.x, p.p2.y]])
    elif isinstance(p, Circle):
        return shapely.Point(p.center.x, p.center.y).buffer(p.radius)
    elif isinstance(p, Triangle):
        return shapely.Polygon([[p.v1.x, p.v1.y], [p.v2.x, p.v2.y], [p.v3.x, p.v3.y]])
    else: #Polygon
        vertices = []
        for vertex in p.vertices:
            vertices += [(vertex.x, vertex.y)]
        return shapely.Polygon(vertices)
    
def geo_primitive_to_shapely_with_transformation(p: GeoPrimitive, t: Point, s: float):
    if isinstance(p, Point):
        return shapely.Point(int((p.x - t.x)*s), int((p.y - t.y)*s))
    elif isinstance(p, Segment):
        return shapely.LineString([[int((p.p1.x - t.x)*s), int((p.p1.y - t.y)*s)], [int((p.p2.x - t.x)*s), int((p.p2.y - t.y)*s)]])
    elif isinstance(p, Circle):
        return shapely.Point(int((p.center.x - t.x)*s), int((p.center.y - t.y)*s)).buffer(p.radius)
    elif isinstance(p, Triangle):
        return shapely.Polygon([[int((p.v1.x - t.x)*s), int((p.v1.y - t.y)*s)], [int((p.v2.x - t.x)*s), int((p.v2.y - t.y)*s)], [int((p.v3.x - t.x)*s), int((p.v3.y - t.y)*s)]])
    else: #Polygon
        vertices = []
        for vertex in p.vertices:
            vertices += [(int((vertex.x - t.x)*s), int((vertex.y - t.y)*s))]
        return shapely.Polygon(vertices)

class CollisionChecker:
    """
    This class implements the collision check ability of a simple planner for a circular differential drive robot.

    Note that check_collision could be used to check collision between given GeoPrimitives
    check_collision function uses the functions that you implemented in CollisionPrimitives class.
    """

    def __init__(self):
        pass

    def path_collision_check(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        # sample points on the circle
        max_points_on_circle = 50
        circle_points = []
        for i in range(max_points_on_circle):
            angle = 2 * np.pi * i / max_points_on_circle
            circle_points.append(Point(r * np.cos(angle), r * np.sin(angle)))

        # check if one of the segments hits an obstacle
        segments_in_collision = set()
        for index in range(len(t.waypoints) - 1):
            s = Segment(Point(t.waypoints[index].x, t.waypoints[index].y), Point(t.waypoints[index+1].x, t.waypoints[index+1].y))
            
            # define segment mathematically
            start_point = np.array([s.p1.x, s.p1.y], dtype=np.float64)
            end_point = np.array([s.p2.x, s.p2.y], dtype=np.float64)
            line_direction = end_point - start_point
            line_length = np.linalg.norm(line_direction)
            if line_length > 0:
                line_direction /= line_length

            max_points_on_segment = int(line_length/r)
            l = lambda s: start_point + s * line_direction
            k = line_length/max_points_on_segment

            exit_flag = False
            for o in obstacles:
                if check_collision(s, o):
                    segments_in_collision.add(index)
                    break
                else:
                    for i in range(max_points_on_segment + 1):
                        center = l(k*i)
                        
                        for cp in circle_points:
                            point = Point(center[0] + cp.x, center[1] + cp.y)
                            
                            if check_collision(point, o):
                                segments_in_collision.add(index)
                                exit_flag = True
                                break
                        
                        if exit_flag:
                            break
                
                if exit_flag:
                    break
        
        return list(segments_in_collision)

    def path_collision_check_occupancy_grid(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will generate an occupancy grid of the given map.
        Then, occupancy grid will be used to check collisions.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        # get min and max point to define the boundaries of the occupancy grid
        waypoints = np.array([(p.x, p.y) for p in t.waypoints])
        min_x, min_y = waypoints.min(axis=0)
        max_x, max_y = waypoints.max(axis=0)

        env_width = int(abs(max_x - min_x))
        env_height = int(abs(max_y - min_y))
        env_size = env_width if env_height < env_width else env_height
        scale = 2.5
        grid_size = int(env_size*scale)
        resolution = env_size / grid_size # how many real units does a cell in the map cover in the real world

        occupancy_grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        scaling = 1/resolution
        shapely_obstacles = [geo_primitive_to_shapely(o) for o in obstacles]
        shapely_segments = [geo_primitive_to_shapely(Segment(Point(t.waypoints[i].x, t.waypoints[i].y), Point(t.waypoints[i+1].x, t.waypoints[i+1].y))).buffer(r) for i in range(0, len(t.waypoints) - 1)]
        
        convert_x = lambda x: np.clip(int((x - min_x)*scaling), 0, grid_size)
        convert_y = lambda y: np.clip(int((y - min_y)*scaling), 0, grid_size)
        # rasterize obstacles
        for o in shapely_obstacles:
            o_x_min, o_y_min, o_x_max, o_y_max = o.bounds

            o_x_min = convert_x(o_x_min)
            o_y_min = convert_y(o_y_min)
            o_x_max = convert_x(o_x_max)
            o_y_max = convert_y(o_y_max)
            for row in range(o_y_min, o_y_max):
                y = min_y + row * resolution
                for col in range(o_x_min, o_x_max):
                    x = min_x + col * resolution
                    real_point = shapely.Point([x, y])
                    
                    if o.contains(real_point):
                        occupancy_grid[row, col] = 1

        segments_in_collision = set()
        segment_index = 0
        for s in shapely_segments:
            s_x_min, s_y_min, s_x_max, s_y_max = s.bounds

            s_x_min = convert_x(s_x_min)
            s_y_min = convert_y(s_y_min)
            s_x_max = convert_x(s_x_max)
            s_y_max = convert_y(s_y_max)

            exit_flag = False
            for row in range(s_y_min, s_y_max):
                y = min_y + row * resolution
                for col in range(s_x_min, s_x_max):
                    x = min_x + col * resolution
                    real_point = shapely.Point([x, y])
                    if occupancy_grid[row, col] + 1 == 2:
                        if s.contains(real_point):
                            segments_in_collision.add(segment_index)
                            exit_flag = True
                            break
                if exit_flag:
                    break
            
            segment_index += 1
        
        return list(segments_in_collision)
    
    def path_collision_check_r_tree(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will build an R-Tree of the given obstacles.
        You are free to implement your own R-Tree or you could use STRTree of shapely module.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        shapely_obstacles = [geo_primitive_to_shapely(o) for o in obstacles]
        r_tree_obstacles = shapely.STRtree(shapely_obstacles)
        
        segments_in_collisions = []
        for index in range(len(t.waypoints)):
            if index == len(t.waypoints) - 1:
                break

            shapely_segment = geo_primitive_to_shapely(Segment(Point(t.waypoints[index].x, t.waypoints[index].y), Point(t.waypoints[index+1].x, t.waypoints[index+1].y)))
            shapely_segment_with_buffer = shapely_segment.buffer(r)
            indices_intersections = r_tree_obstacles.query(shapely_segment_with_buffer, predicate="intersects")
            if len(indices_intersections) > 0:
                segments_in_collisions += [index]

        return segments_in_collisions

    def collision_check_robot_frame(
        self,
        r: float,
        current_pose: SE2Transform,
        next_pose: SE2Transform,
        observed_obstacles: List[GeoPrimitive],
    ) -> bool:
        """
        Returns there exists a collision or not during the movement of a circular differential drive robot until its next pose.

            Parameters:
                    r (float): Radius of circular differential drive robot
                    current_pose (SE2Transform): Current pose of the circular differential drive robot
                    next_pose (SE2Transform): Next pose of the circular differential drive robot
                    observed_obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives in robot frame
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        segment_shapely = shapely.LineString([[current_pose.p[0], current_pose.p[1]], [next_pose.p[0], next_pose.p[1]]])
        segment_shapely_with_buffer = segment_shapely.buffer(r)

        cp = current_pose.p
        theta = current_pose.theta
        affine_transformation = [
            np.cos(theta),
            -np.sin(theta),
            np.sin(theta),
            np.cos(theta),
            cp[0],
            cp[1]
        ]

        obstacle_shapely = [geo_primitive_to_shapely(o) for o in observed_obstacles]

        for o in obstacle_shapely:
            o_fixed_frame = shapely.affinity.affine_transform(o, affine_transformation)

            if segment_shapely_with_buffer.intersects(o_fixed_frame):
                return True

        return False

    def path_collision_check_safety_certificate(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will implement the safety certificates procedure for collision checking.
        You are free to use shapely to calculate distance between a point and a GoePrimitive.
        For more information, please check Algorithm 1 inside the following paper:
        https://journals.sagepub.com/doi/full/10.1177/0278364915625345.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        S_free = []  # Safe points without collision {name_of_point: (x, y)}
        S_obs = []  # Points in collision with obstacles {name_of_point: (x, y)}
        Dist = {}  # Distance map {name_of_point: d_min}
        Certificates = {} # disctionary that for each certificate point a list of points is append which it vertifies {name_of_certificate_point: [(x1, y1), (x2, y2), ...]}

        segments_in_collisions = set()
        segments_shapely = [shapely.LineString([[t.waypoints[i].x, t.waypoints[i].y], [t.waypoints[i+1].x, t.waypoints[i+1].y]]) for i in range(len(t.waypoints) - 1)]
        obstacles_shapely = [geo_primitive_to_shapely(o).buffer(r) for o in obstacles]

        max_sample_point_per_segment = 20
        segment_index = 0
        for s in segments_shapely:
            x, y = s.xy
            
            start_point = np.array([x[0], y[0]], dtype=np.float64)
            end_point = np.array([x[1], y[1]], dtype=np.float64)
            line_direction = end_point - start_point
            line_length = np.linalg.norm(line_direction)
            if line_length > 0:
                line_direction /= line_length

            l = lambda s: start_point + s * line_direction
            k = line_length/max_sample_point_per_segment

            for sample in range(max_sample_point_per_segment + 1):
                p = l(k*sample)
                point = shapely.Point([p[0], p[1]])

                if S_free:
                    distances = [shapely.distance(p, point) for p in S_free]
                    d_min_free = np.min(distances)
                    point_free = np.argmin(distances)
                
                if S_obs:
                    distances = [shapely.distance(p, point) for p in S_obs]
                    d_min_obs = np.min(distances)
                    point_obs = np.argmin(distances)

                if Dist != {}:
                    if S_free != {}:
                        if d_min_free <= Dist[S_free[point_free]]:
                            Certificates[point] = S_free[point_free]
                    
                    elif S_obs != {}:
                        if d_min_obs <= Dist[S_obs[point_obs]]:
                            segments_in_collisions.add(segment_index)


                distances = [point.distance(o) for o in obstacles_shapely]
                d_obs = np.min(distances)
                nearest_obstacle_index = np.argmin(distances)

                if d_obs > 0:
                    S_free += [point]
                    Dist[point] = d_obs
                    Certificates[point] = point
                else:
                    d_free = shapely.distance(point, obstacles_shapely[nearest_obstacle_index].boundary)
                    S_obs += [point]
                    Dist[point] = d_free
                    segments_in_collisions.add(segment_index)


            segment_index += 1

            
        return list(segments_in_collisions)