from pose2gait.utils.rotate_3d_seq import angle_between_two_vectors, rotate_point_around_origin
import pytest


@pytest.fixture
def angle():
    point1 = [0., 1.]
    point2 = [1., 0.]
    
    movement = [point2[0] - point1[0], point2[1] - point1[1]]
    constant = [1 - point1[0], 1 - point1[1]]

    angle = angle_between_two_vectors(movement, constant)
    return angle

@pytest.fixture
def origin():
    point1 = [0., 1.]
    return point1

def test_angle(angle):
    assert angle == pytest.approx(0.7853982)

def test_rotation_identity(origin, angle):
    result = rotate_point_around_origin(origin, origin, angle)
    assert result == origin

def test_rotation_point2(origin, angle):
    point2 = [1., 0.]
    result = rotate_point_around_origin(point2, origin, angle)
    assert result[0] > 1
    assert result[1] == 1

def test_rotation_point3(origin, angle): 
    point3 = [2, -1]
    result = rotate_point_around_origin(point3, origin, angle)
    assert result[1] == 1
    assert result[0] > 2

def test_rotation_point4(origin, angle):
    point4 = [-1, 2]
    result = rotate_point_around_origin(point4, origin, angle)
    assert result[1] == pytest.approx(1)
    assert result[0] < -1