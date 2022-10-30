import numpy as np
import logging
import toml
import os

logger = logging.getLogger(__name__)

class PoseSequence:
    
    def __init__(self, action_id, seq_id, fps, joint_names, connections,
                 dims=2, joint_info=None, metadata=None, pose_func=None):
        """A sequence of poses, such as those extracted from consecutive
        frames of a video. Has the option to initialize with empty joint locations
        and confidences and pass in a function to load them at access time,
        for efficient lazy loading of datasets.

        Args:
            action_id (String): an id string for the source action bout that is being represented
                by the sequence. For example, an id for the orignal video.
            seq_id (String): an id string for the sequence. There can be multiple
                sequences per source action, from different cameras or pose estimators.
            fps (int): the number of frames/poses per second
            joint_names (list of string): A list of names for the J joints, in the same order
                as joint_locs
            connections (list of tuple of string): A list of joint name pairs signifying connected joints
            joint_info (np.array, optional): a FxJx(D+M) array,
                where F is the number of frames, J is the number of joints,
                and D is the number of dimensions of each joint, and M is the length of optional
                additional per-joint information such as a confidence score or ground contacts
                (called "joint data"). Can be None if pose_func supplied for lazy loading.
            dims (int): Number of location dimensions in the pose sequence. Defaults to 2, but 3
                is also a valid value.
            metadata (dict, optional): Dictionary from string keys to string values 
                representing additional metadata, such as camera id, direction of walk.
                Defaults to None.
            pose_func(() -> np.array): A zero argument function that returns
                joint_info as described above, for lazy loading. Default is None.
        """
        self.walk_id = action_id
        self.seq_id = seq_id
        self.fps = float(fps)
        self.joint_names = joint_names
        self.connections = connections
        self.dims = dims
        self.num_joints = len(joint_names)
        self.metadata = {} if metadata is None else metadata
        self.pose_func = pose_func
        if joint_info is not None:
            self.set_joint_info(joint_info)

    def set_joint_info(self, joint_info):
        """Setter for joint info and num_frames. Converts the joint info
        to a numpy array and checks that the sequence is valid.

        Args:
            joint_info (np.array): a FxJx(D+M) array,
                where F is the number of frames, J is the number of joints,
                and D is the number of dimensions of each joint, and M is the length of optional
                additional per-joint information such as a confidence score or ground contacts.
        """
        self.joint_info = joint_info
        self.num_frames = self.joint_info.shape[0]
        self.__check_valid_seq()

    def location_by_name(self, joint_name):
        """Get the location of a joint by name for all frames of the sequence.

        Args:
            joint_name (string): Name of the joint

        Returns:
            np.array: A FxD numpy array containing the location of the joint
                over F frames, where D is the number of dimensions. 
        """        
        index = self.joint_names.index(joint_name)
        return self.joint_info[:, index, :self.dims]
    
    def data_by_name(self, joint_name):
        """Get the data of a joint by name for all frames of the sequence.

        Args:
            joint_name (string): Name of the joint

        Returns:
            np.array: A FxM numpy array containing the extra information about the joint
                over F frames, where M is the number of extra pieces of information.
        """        
        index = self.joint_names.index(joint_name)
        return self.joint_info[:, index, self.dims:]
    
    def get_joint_locations(self):
        """Get just the locations of each joint (ignore additional joint data)
        
        Returns:
            np.array: an FxJxD numpy array representing the locaiton of
                each joint in each frame """
        return self.joint_info[:, :, :self.dims]

    def set_joint_locations(self, joint_locations):
        """Set just the locations of each joint, leaving additional joint data as is

        Args:
            joint_locations (np.array): an FxJxD array representing the locaiton of
                each joint in each frame
        """
        self.joint_info[:, :, :self.dims] = joint_locations
    
    def get_joint_data(self):
        """Get only the additional joint data for each joint in each frame

        Returns:
            np.array: an FxJxM array representing the additional data of
                each joint in each frame
        """
        return self.joint_info[:, :, self.dims:]

    def set_joint_data(self, joint_data):
        """Set only the additional joint data, leaving locations the same

        Args:
            joint_data (np.array): an FxJxM array representing the additional data of
                each joint in each frame
        """
        self.joint_info[:, :, self.dims:] = joint_data

    def filter_joints(self, exclude):
        """Return a new PoseSequence with specified joints excluded

        Args:
            exclude (list of string): names of joints to remove
        
        Returns: a new PoseSequence with specified joints excluded
        """
        indices = []
        for e in exclude:
            if e in self.joint_names:
                indices.append(self.joint_names.index(e))
        indices.sort()
        joint_names = [j for j in self.joint_names if j not in exclude]
        joint_info = np.delete(self.joint_info, indices, axis=1)
        connections = ((a, b) for a, b in self.connections\
                        if a in joint_names and b in joint_names)
        return PoseSequence(self.walk_id, self.seq_id, self.fps, 
                            joint_names, connections,
                            joint_info=joint_info,
                            metadata=self.metadata)

    def get_min_bound(self):
        """Get the minimum values for joint locations in each dimension 
        over the full pose sequence.

        Returns:
            np.array: A numpy array containing the minimum values in each dimension.
        """
        joint_locs = self.joint_info[:, :, :self.dims]
        return self.__get_bound(joint_locs, np.nanmin)

    def get_max_bound(self):
        """Get the maximum values for joint locations in each dimension 
        over the full pose sequence.

        Returns:
            np.array: A numpy array containing the maximum values in each dimension.
        """
        joint_locs = self.joint_info[:, :, :self.dims]
        return self.__get_bound(joint_locs, np.nanmax)

    def __get_bound(self, joint_locs, func):
        return func(joint_locs, axis=(0, 1))

    def to_numpy(self, confs=False):
        """Convert the PoseSequence into a numpy array of values.

        Args:
            confs (bool, optional): Flag indicating whether to include
            joint confidence scores in the array. Defaults to False.

        Returns:
            np.array: an FxJxD array, where F is length of sequence,
                J is number of joints, and D is dimension of each joint
                including confidence score if applicable
        """
        if not confs:
            return self.joint_info[:, :, :self.dims]
        else:
            return self.joint_info

    def shift(self, vector):
        """Shift all joints in the pose in the direction specified by vector.
        Vector must have same dims as joints. Operates in-place."""
        self.joint_info[:, :, :self.dims] = self.joint_info[:, :, :self.dims] + np.array(vector)

    def scale(self, scalar):
        """Scale all joint locations by scalar. Operates in-place."""
        self.joint_info[:, :, :self.dims] = self.joint_info[:, :, :self.dims] * scalar

    def to_file(self, dirname):
        """Write this pose sequence to files in the specified directory.
        There are no constraints on how it is stored - currently it is
        one metadata file, one joint locations file, and one joint confs file.

        Args:
            dirname (string): The directory in which to store the pose sequence.
            If it is not empty, some files may be overwritten if there is a naming clash.
        """
        os.makedirs(dirname, exist_ok=True)
        metadata_file = os.path.join(dirname, "meta.toml")
        metadata = self.metadata.copy()
        metadata["walk_id"] = self.walk_id
        metadata["seq_id"] = self.seq_id
        metadata["fps"] = self.fps
        metadata['joint_names'] = self.joint_names
        metadata['connections'] = self.connections
        metadata['dims'] = self.dims
        with open(metadata_file, 'w') as f:
            toml.dump(metadata, f)
        
        joint_info_file = os.path.join(dirname, "joint_info.npy")
        np.save(joint_info_file, self.joint_info)
        logger.debug(f"Wrote pose seq {self.walk_id} {self.seq_id} to {dirname}")

    @staticmethod
    def from_file(dirname):
        """Static method to read a pose sequence from the specified directory. 
        Assumes that the pose sequence was written using PoseSequence.to_file()
        with the same directory name.

        Args:
            dirname (string): Directory from which to read the pose sequence

        Returns:
            PoseSequence: pose sequence stored in the given directory.
        """
        metadata_file = os.path.join(dirname, "meta.toml")
        with open(metadata_file, 'r') as f:
            metadata = toml.load(f)
            walk_id = metadata.pop("walk_id")
            seq_id = metadata.pop("seq_id")
            fps = metadata.pop("fps")
            joint_names = metadata.pop('joint_names')
            connections = metadata.pop('connections')
            dims = metadata.pop('dims')
        
        joint_info_file = os.path.join(dirname, "joint_info.npy")
        joint_info = np.load(joint_info_file)

        return PoseSequence(walk_id, seq_id, fps, joint_names, connections,
                                dims=dims, joint_info=joint_info,
                                metadata=metadata)

    def __getattr__(self, name):
        """Overwriting the function called when default attribute access fails. 
        If the attribute is joint_locs, joint_confs, num_frames, or dims,
        load and store poses the joint locations and confidences,
        then return the attribute. This allows lazy loading at the time the
        attributes are accessed. For other attributes, the default
        funcitonality remains.

        Returns:
            The attribute value for attribute with name name, after potentially
            loading the poses.
        
        Raises: 
            ValueError if it tries to load the poses and self.pose_func is None.
            AttributeError if the attribute is not found in the class
        """
        lazy_attrs = ['joint_info', 'num_frames']
        if name in lazy_attrs:
            self._load_joints()
        return self.__getattribute__(name)

    def _load_joints(self):
        """Call the provided pose_func to load joint locations and confidences.

        Raises:
            ValueError: raised if pose_func is null.
        """
        if self.pose_func() is None:
            raise ValueError("No function provided for lazy loading of poses")
        self.set_joint_info(*self.pose_func())

    def __check_valid_seq(self):
        """ Check that the sequence is valid, in that the joint information
         matches the metadata provided

        Raises:
            ValueError: if the pose sequence is not valid
        """
        if self.joint_info.shape[1] != len(self.joint_names):
            raise ValueError(f"Number of joint locations ({self.joint_info.shape[1]})"\
            f" must equal number of joint names({len(self.joint_names)})")