import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import savefig
from matplotlib.colors import to_rgb
import os
import numpy as np
import imageio
import logging

logger = logging.getLogger(__name__)


def is_right(joint_name):
    if joint_name.lower().startswith('r'):
        return True
    else:
        return False


def get_rgba_color(color_name, alpha):
    r, g, b = to_rgb(color_name)
    return (r, g, b, float(alpha))


def visualize_sequence_matplotlib(
        sequence, video_name, tempdir='./temp',
        padding=[10, 10, 10], joint_colors=None,
        save_frames=False, dims=[0, 1], axes=False,
        exclude_joints=None):
    """_summary_

    Args:
        sequence (_type_): _description_
        video_name (str): name of output file to save video to
        tempdir (str, optional): _description_. Defaults to './temp'.
        padding (list, optional): _description_. Defaults to [10, 10, 10].
        joint_colors (_type_, optional): _description_. Defaults to None.
        save_frames (bool, optional): _description_. Defaults to False.
        dims (list, optional): _description_. Defaults to [0, 1].
        axes (bool, optional): _description_. Defaults to False.
        exclude_joints ((np.array) -> bool, optional):
            A function that takes in the joint data
            and returns true if the joint should be excluded from
            visualization and false otherwise. Defaults to None,
            which includes all joints.

    Returns:
        _type_: _description_
    """
    dim0 = dims[0]
    dim1 = dims[1]

    # compute min and max bounds
    mins = sequence.get_min_bound()
    maxs = sequence.get_max_bound()
    logger.debug(f"minimums: {mins}")
    logger.debug(f"maximums: {maxs}")
    xmin = int(mins[dim0] - padding[dim0])
    xmax = int(maxs[dim0] + padding[dim0])
    ymin = int(mins[dim1] - padding[dim1])
    ymax = int(maxs[dim1] + padding[dim1])
    images = []
    fps = sequence.fps
    logger.debug(f"fps: {fps}")

    all_joint_locs = sequence.get_joint_locations()
    all_joint_data = sequence.get_joint_data()

    # write individual images to temporary directory
    os.makedirs(tempdir, exist_ok=True)
    for i in range(sequence.num_frames):
        joint_locs = all_joint_locs[i]
        joint_data = all_joint_data[i]
        p = plt.figure(figsize=(4, 6))
        ax = p.add_axes([0, 0, 1, 1])
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_axis_off()

        if axes:
            ax.axhline(0, color='grey')
            ax.axvline(0, color="grey")
        if joint_colors is None:
            frame_colors = ["red" if is_right(name) else "blue"
                            for name in sequence.joint_names]
        else:
            frame_colors = joint_colors[i]
        pose_info = zip(sequence.joint_names, joint_locs,
                        joint_data, frame_colors)
        to_graph = []
        skip_joints = []

        for name, loc, data, color in pose_info:
            if exclude_joints is not None and exclude_joints(data):
                logger.debug(f"Skipping joint {name} with data {data}")
                skip_joints.append(name)
                continue
            color = get_rgba_color(color, 1.0)
            to_graph.append([loc[dim0], loc[dim1], color])
        if len(to_graph) == 0:
            logger.warn("No joints in this frame above threshold:"
                        " writing blank frame")
        else:
            to_graph = np.array(to_graph, dtype=object)
            ax.scatter(to_graph[:, 0], to_graph[:, 1], color=to_graph[:, 2])

            for conn in sequence.connections:
                if conn[0] in skip_joints or conn[1] in skip_joints:
                    continue
                line = [sequence.location_by_name(name)[i] for name in conn]
                color = "red" if is_right(conn[0]) and is_right(conn[1])\
                        else "blue"
                line = Line2D([p[0] for p in line], [p[1] for p in line],
                              color=color)
                ax.add_line(line)

        x = 10
        y = 10
        ax.text(x, y, f"walk_id: {sequence.walk_id}", transform=None)
        img = os.path.join(tempdir, str(i) + ".png")
        logger.debug(f"writing {i}th frame to {img}")
        savefig(img)
        plt.close()
        images.append(img)

    # create video from images
    writer = imageio.get_writer(video_name, fps=fps)
    for im in images:
        writer.append_data(imageio.imread(im))
        # remove temporary image file from disk
        if not save_frames:
            os.remove(im)
    writer.close()
    logger.info(f"wrote video to {video_name}")

    return video_name
