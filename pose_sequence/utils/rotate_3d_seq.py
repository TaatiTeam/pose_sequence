import math
import logging
import numpy as np

logger = logging.getLogger(__name__)

def get_seq_rotation(
    pose_seq,
    start=0, end=-1,
    joint_name='MidHip',
    level_dim=1,
    rotate_dim=2):
    """Find the origin and angle of rotation to get a particular joint's level_dim
    to be constant over time by rotating with respect to rotate_dim
    (e.g. to avoid the sinking problem with HuMoR output).

    Results will be returned in the space (rotate_dim, level_dim) [e.g. (z, y) by default],
    and the rotation will only work in this space.

    Args:
        pose_seq (pose2gait.PoseSequence): PoseSequence to be rotated
        start (int, optional): Frame index of pose_seq to use as the start point. 
            The result will rotate the system so the y values of start and 
            end are equal (and equal to the y value of start). Defaults to 0.
        end (int, optional): Frame index of pose_seq to use as the end point.
            The result will rotate the system so the y values of start and 
            end are equal. Defaults to -1.
        joint_name (str, optional): Name of the joint to level out. Defaults to 'MidHip'.
        level_dim (int, optional): Dimension to equalize over time. Defaults to 1, or the y dimension.
        rotate_dim (int, optional): Dimension to rotate with regard to. Defaults to 2, or the z dimension.

    Returns:
        float, list: The angle to rotate by, and the origin to rotate around 
            in (rotate_dim, level_dim) space (e.g. (z, y) by default).
    """
    joint_locs = pose_seq.get_joint_locations(joint_name)
    first = joint_locs[start]
    last = joint_locs[end]
    logger.debug(f"first={first}")
    logger.debug(f"last={last}")
    
    a0 = first[rotate_dim]
    b0 = first[level_dim]
    a1 = last[rotate_dim]
    b1 = last[level_dim]

    origin = a0, b0 
    movement_vector = [a1-a0, b1-b0]
    logger.debug(f"movement_vector={movement_vector}")
    constant_vector = [a1-a0, 0]
    logger.debug(f"constant_vector={constant_vector}")
    angle = angle_between_two_vectors(movement_vector, constant_vector)
    return angle, origin


def angle_between_two_vectors(v1, v2):
    """Generic implementation of getting the angle between two 2D vectors 

    Args:
        v1 (list or tuple of float): Iterable with two float elements representing a vector
        v2 (list or tuple of float): Iterable with two float elements representing a vector

    Returns:
        float: Angle in radians between v1 and v2
    """ 
    x1, y1 = v1
    x2, y2 = v2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.acos(inner_product/(len1*len2))
    

def rotate_point_around_origin(point, origin, angle, clockwise=False):
    """Rotate a point counterclockwise by a given angle around a given origin.

    Args:
        point (list of float): list of two float elements representing a point in space
        origin (list of float): list with two float elements around which to rotate point
        angle (float): Angle in radians by which to rotate point around origin

    Returns:
        list of float: a list of two float elements representing the rotated point
    """

    if clockwise:
        angle = -1 * angle
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qx, qy]

def rotate_pose_seq_around_origin(pose_seq, origin, angle, dims=[2, 1], clockwise=False):
    """Rotate a pose in 2D by a given angle around a given origin.
    # TODO: this seems inefficient 
    
    Args:
        pose_seq (pose2gait.PoseSequence): The pose sequence to rotate
        origin (list of float): Two element list reprenting the point around which to rotate.
            This point is in the space defined by given dims.
        angle (float): Angle by which to rotate, in the space defined by dims.
        dims (list, optional): Dimensions in which the rotation is taking place.
            Defaults to [2, 1], which is [z, y].

    Returns:
        pose2gait.PoseSequence: The same pose sequence with every joint location
            rotated by angle around origin in the space defined by dims.
    """
    dim0, dim1 = dims
    joint_locs = pose_seq.joint_locations()
    new_joint_locs = []
    for frame in joint_locs:
        new_frame = []
        for joint_loc in frame:
            to_rotate = [joint_loc[dim0], joint_loc[dim1]]
            rotated = rotate_point_around_origin(to_rotate, origin, angle, clockwise=clockwise)
            new_joint = joint_loc.copy()
            for i, dim in enumerate(dims):
                new_joint[dim] = rotated[i]
            new_frame.append(new_joint)
        new_joint_locs.append(new_frame)
    pose_seq.set_joint_locations(np.array(new_joint_locs))
    return pose_seq


def equalize_y_wrt_z(
        pose_seq,
        start=0, end=-1,
        joint_name='MidHip'):
    dims = [2, 1]
    angle, origin = get_seq_rotation(
        pose_seq, start=start, end=end, joint_name=joint_name,
        level_dim=dims[1], rotate_dim=dims[0])
    return rotate_pose_seq_around_origin(pose_seq, origin, angle, dims)


def equalize_x_wrt_z(
        pose_seq,
        start=0, end=-1,
        joint_name='MidHip'):
    dims = [2, 0]
    angle, origin = get_seq_rotation(
        pose_seq, start=start, end=end, joint_name=joint_name,
        level_dim=dims[1], rotate_dim=dims[0])
    logger.debug(f"angle: {angle}, origin: {origin}")
    return rotate_pose_seq_around_origin(pose_seq, origin, angle, dims)


