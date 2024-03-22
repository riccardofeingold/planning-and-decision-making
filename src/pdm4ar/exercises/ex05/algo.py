from typing import Sequence

from dg_commons import SE2Transform

from pdm4ar.exercises.ex05.structures import *
from pdm4ar.exercises_def.ex05.utils import extract_path_points


class PathPlanner(ABC):
    @abstractmethod
    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        pass


class Dubins(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> List[SE2Transform]:
        """ Generates an optimal Dubins path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        path = calculate_dubins_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


class ReedsShepp(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        """ Generates a Reeds-Shepp *inspired* optimal path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow 
        """
        path = calculate_reeds_shepp_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list

# EXERCISE 1
def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
    # TODO implement here your solution
    min_radius = wheel_base / np.tan(max_steering_angle)
    return DubinsParam(min_radius=min_radius)

# EXERCISE 2
def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
    # TODO implement here your solution
    current_position = current_config.p
    heading_theta = current_config.theta
    heading_vec = np.array([
        np.cos(heading_theta), np.sin(heading_theta)
    ])
    shift_to_right = np.dot(R(-np.pi/2), heading_vec)

    p_center_right = current_position + shift_to_right*radius
    p_center_left = current_position - shift_to_right*radius
    center_right = SE2Transform(p_center_right, 0)
    center_left = SE2Transform(p_center_left, 0)

    right_circle = Curve.create_circle(center=center_right, config_on_circle=current_config, radius=radius, curve_type=DubinsSegmentType.RIGHT)
    left_circle = Curve.create_circle(center=center_left, config_on_circle=current_config, radius=radius, curve_type=DubinsSegmentType.LEFT)
   
    return TurningCircle(left=left_circle, right=right_circle)

# EXERCISE 3
def calculate_tangent_btw_circles(circle_start: Curve, circle_end: Curve) -> List[Line]:
    # TODO implement here your solution
    # Get the outer tangents
    C = circle_end.center.p - circle_start.center.p
    D = np.linalg.norm(C)
    lines = []

    # either LL or RR
    if circle_start.type == circle_end.type:
        if D > abs(circle_start.radius - circle_end.radius):
            theta = np.arccos((circle_start.radius - circle_end.radius)/D)  
            if circle_start.type == DubinsSegmentType.LEFT:
                theta *= -1
            n = np.dot(R(theta), C)
            n /= np.linalg.norm(n)
            E = C - (circle_start.radius - circle_end.radius) * n # has the same direction has D since parallel

            line_start = circle_start.center.p + circle_start.radius*n
            line_end = line_start + E
            
            # get the heading of the tangent line
            E_norm = 1/np.linalg.norm(E) * E
            alpha = np.arctan2(E_norm[1], E_norm[0])
            
            lines.append(Line(start_config=SE2Transform(line_start, alpha), end_config=SE2Transform(line_end, alpha)))
    else:
        ### Get the start and end point of the first tangent
        if D >= abs(circle_start.radius + circle_end.radius):
            theta = np.arccos((circle_start.radius + circle_end.radius)/D)
            if circle_start.type == DubinsSegmentType.LEFT:
                theta *= -1
            
            n = np.dot(R(theta), C)
            n /= np.linalg.norm(n)
            E = C - (circle_start.radius + circle_end.radius) * n # has the same direction has D since parallel

            line_start = circle_start.center.p + circle_start.radius*n
            line_end = line_start + E
            
            # get the heading of the tangent line
            E_mag = np.linalg.norm(E)
            if E_mag > 0:
                E_norm = 1/E_mag * E
                alpha = np.arctan2(E_norm[1], E_norm[0])
            else:
                phi = np.pi/2 if circle_start.type == DubinsSegmentType.LEFT else -np.pi/2
                heading = np.dot(R(phi), n)
                alpha = np.arctan2(heading[1], heading[0])
            
            lines.append(Line(start_config=SE2Transform(line_start, alpha), end_config=SE2Transform(line_end, alpha)))
   
    return lines

# EXERCISE 4
def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins path!
    
    # circles at the start and at the end
    start_turning_circles = calculate_turning_circles(current_config=start_config, radius=radius)
    list_start_circles = [start_turning_circles.left, start_turning_circles.right]
    end_turning_circles = calculate_turning_circles(current_config=end_config, radius=radius)
    list_end_circles = [end_turning_circles.left, end_turning_circles.right]

    # Get the CSC Paths
    paths = []
    for s_circles in list_start_circles:
        for e_circles in list_end_circles:
            tangents = calculate_tangent_btw_circles(s_circles, e_circles)
            straight_line = tangents[0] if tangents != [] else []

            if tangents != []:
                start_v1 = start_config.p - s_circles.center.p
                start_v1 /= np.linalg.norm(start_v1)
                start_v2 = straight_line.start_config.p - s_circles.center.p
                start_v2 /= np.linalg.norm(start_v2)
                start_arc_angle = get_arc_angle(v1=start_v1, v2=start_v2, curve_type=s_circles.type)

                end_v1 = straight_line.end_config.p - e_circles.center.p
                end_v1 /= np.linalg.norm(end_v1)
                end_v2 = end_config.p - e_circles.center.p
                end_v2 /= np.linalg.norm(end_v2)
                end_arc_angle = get_arc_angle(v1=end_v1, v2=end_v2, curve_type=e_circles.type)

                start_curve = Curve(
                            start_config=start_config, 
                            end_config=straight_line.start_config, 
                            radius=s_circles.radius,
                            center=s_circles.center,
                            curve_type=s_circles.type,
                            arc_angle=start_arc_angle
                        )
                end_curve = Curve(
                        start_config=straight_line.end_config,
                        end_config=end_config,
                        center=e_circles.center,
                        radius=e_circles.radius,
                        curve_type=e_circles.type,
                        arc_angle=end_arc_angle
                    )

                paths.append([start_curve, straight_line, end_curve])

    # CCC paths
    rlr_paths = get_ccc_dubins_paths(start_turning_circles.right, end_turning_circles.right, radius, start_config, end_config)
    lrl_paths = get_ccc_dubins_paths(start_turning_circles.left, end_turning_circles.left, radius, start_config, end_config)

    if lrl_paths != []:
        for pl in lrl_paths:
            paths.append(pl)
    if rlr_paths != []:
        for pr in rlr_paths:
            paths.append(pr)
    
    # get the shortest path from paths
    shortest_length = float('inf')
    best_path = None
    for p in paths:
        current_length = sum(s.length for s in p)
        if current_length < shortest_length:
            shortest_length = current_length
            best_path = p

    return best_path

# EXERCISE 5
def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins/reeds path!
    paths = []

    # calculate the best forward and backward dubin path
    best_forward_dubin_path = calculate_dubins_path(start_config, end_config, radius)
    if best_forward_dubin_path != []:
        paths.append(best_forward_dubin_path)
    best_backward_dubin_path = calculate_dubins_path(end_config, start_config, radius)

    if best_backward_dubin_path != []:
        if best_backward_dubin_path[1].type == DubinsSegmentType.STRAIGHT:
            for segment in best_backward_dubin_path:
                segment.gear = Gear.REVERSE

            temp = best_backward_dubin_path[0]
            best_backward_dubin_path[0] = best_backward_dubin_path[2]
            best_backward_dubin_path[2] = temp

            for b in best_backward_dubin_path:
                t = b.start_config
                b.start_config = b.end_config
                b.end_config = t

            paths.append(best_backward_dubin_path)
 
    shortest_length = float('inf')
    best_path = []
    for p in paths:
        current_length = sum(s.length for s in p)
        if current_length < shortest_length:
            shortest_length = current_length
            best_path = p
    return best_path  # e.g., [Curve(..,gear=Gear.REVERSE), Curve(),..]


### CUSTOM ###
def R(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def get_arc_angle(v1: np.ndarray, v2: np.ndarray, curve_type: DubinsSegmentType):
    theta = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])

    if theta < 0 and curve_type == DubinsSegmentType.LEFT:
        theta += 2*np.pi
    elif theta > 0 and curve_type == DubinsSegmentType.RIGHT:
        theta -= 2*np.pi

    return np.abs(theta)
    
def get_ccc_dubins_paths(start_circle: Curve, end_circle: Curve, radius: float, start_config: SE2Transform, end_config: SE2Transform):
    signs = [-1, 1]
    paths = []
    for s in signs:
        distance = np.linalg.norm(
            start_circle.center.p - end_circle.center.p
        )

        if distance != 0 and distance < 4*radius:
            theta = np.arccos(distance/(4*radius))        
            distance_vec_norm = 1/distance * (end_circle.center.p - start_circle.center.p)

            theta = np.arctan2(distance_vec_norm[1], distance_vec_norm[0]) + s*theta
            
            middle_center = start_circle.center.p + 2*radius*np.array([np.cos(theta), np.sin(theta)])
            
            v_middle_to_start = start_circle.center.p - middle_center
            v_middle_to_start /= np.linalg.norm(v_middle_to_start)
            start_tangent_point = middle_center + v_middle_to_start*radius
            
            # rotation with respect to the middle curve
            phi = np.pi/2 if start_circle.type == DubinsSegmentType.RIGHT else -np.pi/2
            heading_start_contact_point = np.dot(R(phi), v_middle_to_start)
            alpha_start = np.arctan2(heading_start_contact_point[1], heading_start_contact_point[0])
        
            v_middle_to_end = end_circle.center.p - middle_center
            v_middle_to_end /= np.linalg.norm(v_middle_to_end)
            end_tangent_point = middle_center + v_middle_to_end*radius

            heading_end_contact_point = np.dot(R(phi), v_middle_to_end)
            alpha_end = np.arctan2(heading_end_contact_point[1], heading_end_contact_point[0])
            
            # calculate start_arc_angle
            start_v1 = start_config.p - start_circle.center.p
            start_v1 /= np.linalg.norm(start_v1)
            start_v2 = start_tangent_point - start_circle.center.p
            start_v2 /= np.linalg.norm(start_v2)
            start_arc_angle = get_arc_angle(v1=start_v1, v2=start_v2, curve_type=start_circle.type)

            # calculate middle_arc_angle
            middle_curve_type = DubinsSegmentType.LEFT if start_circle.type == DubinsSegmentType.RIGHT else DubinsSegmentType.RIGHT
            middle_v1 = start_tangent_point - middle_center
            middle_v1 /= np.linalg.norm(middle_v1)
            middle_v2 = end_tangent_point - middle_center
            middle_v2 /= np.linalg.norm(middle_v2)
            middle_arc_angle = get_arc_angle(v1=middle_v1, v2=middle_v2, curve_type=middle_curve_type)

            # calculate end_arc_angle
            end_v1 = end_tangent_point - end_circle.center.p
            end_v1 /= np.linalg.norm(end_v1)
            end_v2 = end_config.p - end_circle.center.p
            end_v2 /= np.linalg.norm(end_v2)
            end_arc_angle = get_arc_angle(v1=end_v1, v2=end_v2, curve_type=end_circle.type)

            # define curves
            start_curve = Curve(
                start_config=start_config,
                end_config=SE2Transform(start_tangent_point, alpha_start),
                center=start_circle.center,
                radius=radius,
                curve_type=start_circle.type,
                arc_angle=start_arc_angle
            )

            middle_curve = Curve(
                start_config=SE2Transform(start_tangent_point, alpha_start),
                end_config=SE2Transform(end_tangent_point, alpha_end),
                center=SE2Transform(middle_center, 0),
                radius=radius,
                curve_type=middle_curve_type,
                arc_angle=middle_arc_angle
            )

            end_curve = Curve(
                start_config=SE2Transform(end_tangent_point, alpha_end),
                end_config=end_config,
                center=end_circle.center,
                radius=radius,
                curve_type=end_circle.type,
                arc_angle=end_arc_angle
            )

            paths.append([start_curve, middle_curve, end_curve])
    return paths