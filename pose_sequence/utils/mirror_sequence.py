import copy


def mirror_sequence(sequence):
    """Create a mirrored version of the sequence by flipping the left and right
    joints and multiplying the x values by -1.
    Assumes that the joints are named for example LShoulder, RShoulder and no
    center joints start with L, and there are no unpaired left/right joints.

    Args:
        sequence (PoseSequence): The pose sequence to mirror.
    """
    mirrored_seq = copy.copy(sequence)
    # multiply x values by -1 to mirror
    mirrored_seq.joint_info[:, :, 0] = sequence.joint_info[:, :, 0] * -1
    # switch left and right joints
    joints = [name[1:] for name in sequence.joint_names if name.startswith("L")]
    left_joints = [sequence.joint_names.indexof("L" + joint) for joint in joints]
    right_joints = [sequence.joint_names.indexof("R" + joint) for joint in joints]
    mirrored_seq.joint_info[:, left_joints + right_joints, :] = mirrored_seq.joint_info[:, right_joints + left_joints, :]
    mirrored_seq.seq_id = mirrored_seq.seq_id + "_mirrored"
    return mirrored_seq