from pdm4ar.exercises_def.ex06.structures import *
from triangle import triangulate
import numpy as np

class CollisionPrimitives:
    """
    Class of collusion primitives
    """

    @staticmethod
    def circle_point_collision(c: Circle, p: Point) -> bool:
        point_vec = np.array([
            [p.x],
            [p.y]
        ])
        circle_vec = np.array([
            [c.center.x],
            [c.center.y]
        ])
        dist = np.linalg.norm(circle_vec - point_vec)

        if dist < c.radius:
            return True
        return False

    @staticmethod
    def triangle_point_collision(t: Triangle, p: Point) -> bool:
        area_original = abs((t.v2.x - t.v1.x)*(t.v3.y - t.v1.y) - (t.v3.x - t.v1.x)*(t.v2.y - t.v1.y))
        area1 = abs((t.v1.x - p.x)*(t.v2.y - p.y) - (t.v2.x - p.x)*(t.v1.y - p.y))
        area2 = abs((t.v2.x - p.x)*(t.v3.y - p.y) - (t.v3.x - p.x)*(t.v2.y - p.y))
        area3 = abs((t.v3.x - p.x)*(t.v1.y - p.y) - (t.v1.x - p.x)*(t.v3.y - p.y))

        total_area = area1 + area2 + area3
        if np.isclose(total_area, area_original):
            return True
        return False

    @staticmethod
    def polygon_point_collision(poly: Polygon, p: Point) -> bool:
        # first triangulate polygon
        vertices = [(vertex.x, vertex.y) for vertex in poly.vertices]
        dict_poly = {
            "vertices": vertices
        }
        poly_triangulated = triangulate(dict_poly)
        new_vertices = poly_triangulated["vertices"]
        indices_triangle_vertices = poly_triangulated["triangles"]

        for triangle in indices_triangle_vertices:
            v1 = Point(new_vertices[triangle[0]][0], new_vertices[triangle[0]][1])
            v2 = Point(new_vertices[triangle[1]][0], new_vertices[triangle[1]][1])
            v3 = Point(new_vertices[triangle[2]][0], new_vertices[triangle[2]][1])

            t = Triangle(v1=v1, v2=v2, v3=v3)

            if CollisionPrimitives.triangle_point_collision(t, p):
                return True

        return False

    @staticmethod
    def circle_segment_collision(c: Circle, segment: Segment) -> bool:
        # first check if either the start or the end point is inside the circle
        if CollisionPrimitives.circle_point_collision(c, segment.p1) or CollisionPrimitives.circle_point_collision(c, segment.p2):
            return True

        circle_center = np.array([c.center.x, c.center.y], dtype=np.float64)
        start_point = np.array([segment.p1.x, segment.p1.y], dtype=np.float64)
        end_point = np.array([segment.p2.x, segment.p2.y], dtype=np.float64)

        start_to_end = end_point - start_point
        start_to_circle = circle_center - start_point
        line_length = np.linalg.norm(start_to_end)
        if line_length > 0:
            start_to_end /= line_length
            dot_product = float(np.dot(start_to_end, start_to_circle))

        
        nearest_point = dot_product*start_to_end + start_point
        d1 = np.linalg.norm(start_point - nearest_point)
        d2 = np.linalg.norm(end_point - nearest_point)

        if np.isclose(d1+d2, line_length):
            if CollisionPrimitives.circle_point_collision(c, Point(nearest_point[0], nearest_point[1])):
                return True
        return False

    @staticmethod
    def triangle_segment_collision(t: Triangle, segment: Segment) -> bool:
        # first checking if either the start or the end point of the segment
        # lies inside the triangle
        if CollisionPrimitives.triangle_point_collision(t, segment.p1) or CollisionPrimitives.triangle_point_collision(t, segment.p2):
            return True
        
        # define line lambda function
        start_point = np.array([segment.p1.x, segment.p1.y], dtype=np.float64)
        end_point = np.array([segment.p2.x, segment.p2.y], dtype=np.float64)
        line_direction = end_point - start_point
        line_length = np.linalg.norm(line_direction)
        if line_length > 0:
            line_direction /= line_length

        l = lambda s: start_point + s * line_direction

        max_points = 50
        s = line_length/max_points
        for i in range(1, max_points):
            point = l(s*i)
            
            if CollisionPrimitives.triangle_point_collision(t, Point(point[0], point[1])):
                return True
            
        return False

    @staticmethod
    def polygon_segment_collision(p: Polygon, segment: Segment) -> bool:
        # first check if start or end point is inside the polygon
        if CollisionPrimitives.polygon_point_collision(p, segment.p1) or CollisionPrimitives.polygon_point_collision(p, segment.p2):
            return True
        
        # Sample points and check with polygon_point_collision
        start_point = np.array([segment.p1.x, segment.p1.y], dtype=np.float64)
        end_point = np.array([segment.p2.x, segment.p2.y], dtype=np.float64)
        line_direction = end_point - start_point
        line_length = np.linalg.norm(line_direction)
        if line_length > 0:
            line_direction /= line_length

        l = lambda s: start_point + s * line_direction

        max_points = 50
        s = line_length/max_points
        for i in range(1, max_points):
            point = l(s*i)
            
            if CollisionPrimitives.polygon_point_collision(p, Point(point[0], point[1])):
                return True

        return False

    @staticmethod
    def polygon_segment_collision_aabb(p: Polygon, segment: Segment) -> bool:
        aabb_rect = CollisionPrimitives._poly_to_aabb(p)
        
        # first check if start or end point is inside the polygon
        if segment.p1.x > aabb_rect.p_min.x and segment.p1.x < aabb_rect.p_max.x and segment.p1.y > aabb_rect.p_min.y and segment.p1.y < aabb_rect.p_max.y:
            return True
        
        if segment.p2.x > aabb_rect.p_min.x and segment.p2.x < aabb_rect.p_max.x and segment.p2.y > aabb_rect.p_min.y and segment.p2.y < aabb_rect.p_max.y:
            return True
        
        # Sample points and check with polygon_point_collision
        start_point = np.array([segment.p1.x, segment.p1.y], dtype=np.float64)
        end_point = np.array([segment.p2.x, segment.p2.y], dtype=np.float64)
        line_direction = end_point - start_point
        line_length = np.linalg.norm(line_direction)
        if line_length > 0:
            line_direction /= line_length

        l = lambda s: start_point + s * line_direction

        max_points = 50
        s = line_length/max_points
        for i in range(1, max_points):
            point = l(s*i)
            
            if point[0] > aabb_rect.p_min.x and point[0] < aabb_rect.p_max.x and point[1] > aabb_rect.p_min.y and point[1] < aabb_rect.p_max.y:
                if CollisionPrimitives.polygon_point_collision(p, Point(point[0], point[1])):
                    return True

        return False

    @staticmethod
    def _poly_to_aabb(g: Polygon) -> AABB:
        # todo feel free to implement functions that upper-bound a shape with an
        #  AABB or simpler shapes for faster collision checks
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        for vertex in g.vertices:
            if vertex.x > max_x:
                max_x = vertex.x
            
            if vertex.x < min_x:
                min_x = vertex.x

            if vertex.y > max_y:
                max_y = vertex.y
            
            if vertex.y < min_y:
                min_y = vertex.y
        
        return AABB(p_min=Point(min_x, min_y), p_max=Point(max_x, max_y))
