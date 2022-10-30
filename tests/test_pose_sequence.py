import pytest
from pose_sequence import PoseSequence
import numpy as np

@pytest.fixture
def pose_seq():
    walk_id = "dummy_walk"
    seq_id = "sequence_1"
    fps = 30
    joint_names = ["LAnkle", "RAnkle"]
    connections = ()
    joint_locs = np.array(
                  [[[0.1, 0.3], [-0.4, 0.3]],  # frame 1
                  [[0.0, 0.2], [-0.2, 0.3]]])  # frame 2
    joint_confs = np.array(
                   [[0.5, 0.9],  # frame 1
                   [0.2, 0.9]])  # frame 2
    joint_info = np.concatenate((joint_locs, 
                                 np.expand_dims(joint_confs, axis=-1)), axis=-1)
    return PoseSequence(walk_id, seq_id, fps, joint_names, connections,
                        joint_info=joint_info)

def test_location_by_name(pose_seq):
    lankle_locs = pose_seq.location_by_name("LAnkle")
    expected = np.array([[0.1, 0.3], [0.0, 0.2]])
    assert (lankle_locs == expected).all()

def test_data_by_name(pose_seq):
    lankle_locs = pose_seq.data_by_name("LAnkle")
    expected = np.array([[0.5], [0.2]])
    assert (lankle_locs == expected).all()

def test_filter_joints(pose_seq):
    filtered = pose_seq.filter_joints(["RAnkle"])
    expected = np.array([[[0.1, 0.3, 0.5]], [[0.0, 0.2, 0.2]]])
    assert (filtered.joint_info == expected).all()

def test_shift(pose_seq):
    pose_seq.shift([1., 1.])
    expected_lankle_loc = np.array([[1.1, 1.3], [1.0, 1.2]])
    assert (pose_seq.location_by_name("LAnkle") == expected_lankle_loc).all()
    lankle_confs = pose_seq.data_by_name("LAnkle")
    expected_lankle_conf = np.array([[0.5], [0.2]])
    assert (lankle_confs == expected_lankle_conf).all()

def test_scale(pose_seq):
    pose_seq.scale([2., 2.])
    expected_lankle_loc = np.array([[0.2, 0.6], [0.0, 0.4]])
    assert (pose_seq.location_by_name("LAnkle") == expected_lankle_loc).all()
    lankle_confs = pose_seq.data_by_name("LAnkle")
    expected_lankle_conf = np.array([[0.5], [0.2]])
    assert (lankle_confs == expected_lankle_conf).all()

def test_to_and_from_file(pose_seq, tmp_path):
    pose_seq.to_file(tmp_path)
    new_seq = PoseSequence.from_file(tmp_path)
    assert pose_seq.walk_id == new_seq.walk_id
    assert pose_seq.seq_id == new_seq.seq_id
    assert pose_seq.fps == new_seq.fps
    assert pose_seq.joint_names == new_seq.joint_names
    assert list(pose_seq.connections) == list(new_seq.connections)
    assert (pose_seq.joint_info == new_seq.joint_info).all()
    assert pose_seq.metadata == new_seq.metadata

def test_max_bound(pose_seq):
    max_bound = pose_seq.get_max_bound()
    expected_max_bound = [0.1, 0.3]
    for pred, expected in zip(max_bound, expected_max_bound):
        assert pred == pytest.approx(expected)

def test_min_bound(pose_seq):
    min_bound = pose_seq.get_min_bound()
    expected_min_bound = [-0.4, 0.2]
    for pred, expected in zip(min_bound, expected_min_bound):
        assert pred == pytest.approx(expected)