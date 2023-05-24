import pytest
from pose_sequence import PoseSequence
from pose_sequence.utils.mirror_sequence import mirror_sequence
import numpy as np

@pytest.fixture
def pos_sequence():
    walk_id = "dummy_walk"
    seq_id = "sequence_1"
    fps = 30
    joint_names = ["LAnkle", "RElbow", "LElbow", "RAnkle"]
    connections = ()
    joint_locs = np.array(
                  [[[0.1, 0.3], [-0.2, 0.5], [0.3, 0.5], [-0.4, 0.3], ],  # frame 1
                  [[0.0, 0.2], [-0.3, 0.4], [0.2, 0.5], [-0.2, 0.3]]])  # frame 2
    joint_confs = np.array(
                   [[0.5, 0.3, 0.3, 0.9],  # frame 1
                   [0.2, 0.3, 0.4, 0.9]])  # frame 2
    joint_info = np.concatenate((joint_locs, 
                                 np.expand_dims(joint_confs, axis=-1)), axis=-1)
    sequence = PoseSequence(walk_id, seq_id, fps, joint_names, connections,
                joint_info=joint_info)
    return mirror_sequence(sequence)

def test_location_by_name(mir_seq):
    lankle_locs = mir_seq.location_by_name("LAnkle")
    expected_ankle = np.array([[0.4, 0.3], [0.2, 0.3]])
    relbow_locs = mir_seq.location_by_name("RElbow")
    expected_elbow = np.array([[-0.3, 0.5], [-0.2, 0.5]])
    assert ((lankle_locs == expected_ankle).all() and (relbow_locs == expected_elbow).all())

def test_data_by_name(mir_seq):
    rankle_locs = mir_seq.data_by_name("RAnkle")
    expected_ankle = np.array([[0.5], [0.2]])
    lelbow_locs = mir_seq.data_by_name("LElbow")
    expected_elbow = np.array([[0.3], [0.3]])
    assert ((rankle_locs == expected_ankle).all() and (lelbow_locs == expected_elbow).all())

def test_seq_id(mir_seq):
    assert (mir_seq == "sequence_1_mirrored")

def test_filter_joints(mir_seq):
    filtered = mir_seq.filter_joints(["LAnkle", "RElbow"])
    expected = np.array([[[0.2, 0.5, 0.3], [-0.1, 0.3, 0.5]], [[0.3, 0.4, 0.3], [0.0, 0.2, 0.2]]])
    assert (filtered.joint_info == expected).all()