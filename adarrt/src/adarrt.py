#!/usr/bin/env python

import argparse
import time

import adapy
import numpy as np
import rospy
import pdb

class AdaRRT():
    """
    Rapidly-Exploring Random Trees (RRT) for the ADA controller.
    """
    joint_lower_limits = np.array([-100, -100, -100, -100, -100, -100])
    joint_upper_limits = np.array([100,100, 100, 100,100,100])
    periodic_index = [0,3]

    class Node():
        """
        A node for a doubly-linked tree structure.
        """
        def __init__(self, state, parent):
            """
            :param state: np.array of a state in the search space.
            :param parent: parent Node object.
            """
            self.state = np.asarray(state)
            self.parent = parent
            self.children = []

        def __iter__(self):
            """
            Breadth-first iterator.
            """
            nodelist = [self]
            while nodelist:
                node = nodelist.pop(0)
                nodelist.extend(node.children)
                yield node

        def __repr__(self):
            return 'Node({})'.format(', '.join(map(str, self.state)))

        def add_child(self, state):
            """
            Adds a new child at the given state.

            :param state: np.array of new child node's statee
            :returns: child Node object.
            """
            child = AdaRRT.Node(state=state, parent=self)
            self.children.append(child)
            return child

    def __init__(self,
                 start_state,
                 goal_state,
                 ada,
                 joint_lower_limits=None,
                 joint_upper_limits=None,
                 ada_collision_constraint=None,
                 step_size=0.25,
                 goal_precision=0.2,
                 max_iter=10000):
        """
        :param start_state: Array representing the starting state.
        :param goal_state: Array representing the goal state.
        :param ada: libADA instance.
        :param joint_lower_limits: List of lower bounds of each joint.
        :param joint_upper_limits: List of upper bounds of each joint.
        :param ada_collision_constraint: Collision constraint object.
        :param step_size: Distance between nodes in the RRT.
        :param goal_precision: Maximum distance between RRT and goal before
            declaring completion.
        :param sample_near_goal_prob:
        :param sample_near_goal_range:
        :param max_iter: Maximum number of iterations to run the RRT before
            failure.
        """
        self.start = AdaRRT.Node(start_state, None)
        self.goal = AdaRRT.Node(goal_state, None)
        self.ada = ada
        self.joint_lower_limits = joint_lower_limits or AdaRRT.joint_lower_limits
        self.joint_upper_limits = joint_upper_limits or AdaRRT.joint_upper_limits
        self.ada_collision_constraint = ada_collision_constraint
        self.step_size = step_size
        self.goal_precision = goal_precision
        self.max_iter = max_iter

    def build(self):
        """
        Build an RRT.

        In each step of the RRT:
            1. Sample a random point.
            2. Find its nearest neighbor.
            3. Attempt to create a new node in the direction of sample from its
                nearest neighbor.
            4. If we have created a new node, check for completion.

        Once the RRT is complete, add the goal node to the RRT and build a path
        from start to goal.

        :returns: A list of states that create a path from start to
            goal on success. On failure, returns None.
        """
        for k in range(self.max_iter):
            # FILL in your code here
            choice = np.random.choice([True, False], 1, p=[0.2, 0.8])
            random_sample = None
            if choice:
                random_sample = self._get_random_sample_near_goal() # step 1: sample a random point near goal
            else:
                random_sample = self._get_random_sample()
            nearest_neighbor = self._get_nearest_neighbor(random_sample) # step 2: find nearest neighbor for the corresponding sample point
            new_node = self._extend_sample(random_sample,nearest_neighbor) # step 3: create a new node in the direction of sample from its NN

            if new_node and self._check_for_completion(new_node): # step 4: if a new node is created, check completion condition
                # FILL in your code here
                self.goal.parent = new_node # step 5: add the goal node to the RRT by connecting goal node with the new_node
                                            # and store the parent/children relationship correspondingly for each node
                new_node.children.append(self.goal)
                path = self._trace_path_from_start(self.goal) # step 6: build a path from start to goal by tracing parent-children pointers
                return path

        print("Failed to find path from {0} to {1} after {2} iterations!".format(
            self.start.state, self.goal.state, self.max_iter)) # fail to find a path from start to the goal node in RRT tree with a max iteration

    def _get_random_sample(self):
        """
        Uniformly samples the search space.

        :returns: A vector representing a randomly sampled point in the search
            space.
        """
        # FILL in your code here
        return np.random.uniform(self.joint_lower_limits,
                                 self.joint_upper_limits)
    
    def _get_random_sample_near_goal(self):
        lower_limits = self.goal.state - 0.05
        lower_limits = np.maximum(lower_limits, self.joint_lower_limits)
        upper_limits = self.goal.state + 0.05
        upper_limits = np.minimum(upper_limits, self.joint_upper_limits)
        
        return np.random.uniform(lower_limits, upper_limits)
        
    def angle_difference_mapping(self,state_x,state_y):
        '''
        Input: 2 state vector state_x and state_y represents angle sets
        Output: angle difference vector after converting into the interval (-pi,pi) to reflect the real angle difference
        '''
        ## for joint1 and joint4, the value of the angle is ranging from -pi to pi(periodic), L2 norm can't measure the difference
        ## of the corner case eg: -170 degree and 170 degree has a 20 degree difference
        state_x_copy = np.asarray(state_x,dtype=float)
        state_y_copy = np.asarray(state_y,dtype=float) # avoid changing the original array
        difference = state_x_copy - state_y_copy 
        # mapping angle diffference to (-pi,pi) only for periodic joints
        for j in self.periodic_index:
            difference[j] = (difference[j] + np.pi) % (2 * np.pi) - np.pi
        # mapping difference to the interval in (-pi,pi),modulus has a 2pi period
        # only apply conversion for the corner case on the boundary, other state difference has the same value before and after conversion
        return difference


    def _get_nearest_neighbor(self, sample):
        """
        Finds the closest node to the given sample in the search space,
        excluding the goal node.

        :param sample: The target point to find the closest neighbor to.
        :returns: A Node object for the closest neighbor.
        """
        # FILL in your code here
        random_sample = np.asarray(sample) # ensure the data type of variable sample is a numpy array, consistency

        nearest_nn = None # initialize a variable to store the NN node for a given random sample
        optimal_distance = float('inf') # initialize a variable to store the distance from the NN node to the random sample

        # iterate through all nodes in the RRT tree to obtain the nearest neighbor, BFS style search for iterable object Node
        for RRT_node in self.start:
            if RRT_node is self.goal: # skip if the node retrieved from the tree is goal node, we want to approach it
                continue
            
            # state_difference = np.linalg.norm(RRT_node.state - random_sample)
            state_difference = self.angle_difference_mapping(RRT_node.state,random_sample) # angle difference after the conversion
            distance = np.linalg.norm(state_difference) # evaluate how close from the node in RRT to the sample point 

            if distance < optimal_distance: # if some node retrieved from the RRT has a pose/state closer to the random sample 
                nearest_nn = RRT_node       # record it as NN 
                optimal_distance = distance
        return nearest_nn

    def _extend_sample(self, sample, neighbor):
        """
        Adds a new node to the RRT between neighbor and sample, at a distance
        step_size away from neighbor. The new node is only created if it will
        not collide with any of the collision objects (see
        RRT._check_for_collision)

        :param sample: target point
        :param neighbor: closest existing node to sample
        :returns: The new Node object. On failure (collision), returns None.
        """
        # FILL in your code here
        if neighbor is None: # if nn doesn't exist, can't extend the RRT and create a new node
            return None 
        # vector pointing from neighbor node to the sample node(sample - neighbor), be careful of the order in the input parameter
        # mapping difference to the interval in (-pi,pi) to obtain real difference
        direction_mapping = self.angle_difference_mapping(sample,neighbor.state)
        distance = np.linalg.norm(direction_mapping) # vector distance
        if distance == 0: # neighbor node is coincident with the sample node, skip extend process
            return None
        step_size = min(self.step_size,distance) # case: distance is smaller than the step size, only extend on the line between NN and sample
        new_node_state = neighbor.state + (direction_mapping / distance)*step_size # unit vector points from neighbor to the sample
                                                                                   # multiply with step_size indicate the magnitude of extension
        # ensure periodic state generated by extension is within the boundary (-pi,pi)
        for j in self.periodic_index:
            new_node_state[j] = (new_node_state[j] + np.pi) % (2 * np.pi) - np.pi
        # ensure all angles within the boundary
        for j in range(len(new_node_state)):
            if j not in self.periodic_index:
                new_node_state[j] = np.minimum(np.maximum(new_node_state[j],self.joint_lower_limits[j]),self.joint_upper_limits[j])
        
        extend_difference = self.angle_difference_mapping(new_node_state,neighbor.state) # vector points from neighor to the extended node
        if np.linalg.norm(extend_difference) < 1e-6: # neighbor node is coincident with the sample node, skip extend process
            return None

        # collision checking the line from neighbor to the extended node by interpolation
        interpolation_rate = self.step_size * 0.125 # how many points we want to interpolate can be adjusted
        num_interpolation_points = int(np.ceil(np.linalg.norm(extend_difference) / interpolation_rate)) # distance of line / step_size = num of points

        if num_interpolation_points < 1: # always check the collision status for the new_node 
            num_interpolation_points = 1

        for p in range(1,num_interpolation_points + 1): # proportion on the straight line decide where to interpolate
            interpolated_point = neighbor.state + extend_difference *(float(p) / num_interpolation_points)
            for j in self.periodic_index:
                interpolated_point[j] = (interpolated_point[j] + np.pi) % (2 * np.pi) - np.pi
            for j in range(len(interpolated_point)):
                if j not in self.periodic_index:
                    interpolated_point[j] = np.minimum(np.maximum(interpolated_point[j],self.joint_lower_limits[j]),self.joint_upper_limits[j]) 
            # ensure all angles within the boundary
            if self._check_for_collision(interpolated_point): # True -- collision False -- collision-free
                return None

        ## if new_node pass the straight-line collision test
        new_node  = neighbor.add_child(new_node_state) # link extended node as the children of the NN node and return
        return new_node


    def _check_for_completion(self, node):
        """
        Check whether node is within self.goal_precision distance of the goal.

        :param node: The target Node
        :returns: Boolean indicating node is close enough for completion.
        """
        # FILL in your code here
        '''
        distance = np.linalg.norm(node.state - self.goal.state)
        return distance <= self.goal_precision
        '''
        difference_mapping = self.angle_difference_mapping(self.goal.state,node.state) #vector points from target node to the goal node
        distance = np.linalg.norm(difference_mapping) # # vector L2 distance between 2 poses difference

        if distance <= self.goal_precision: # if the target node is close enough to the goal node
            # check the straight line from target node to the goal has no collision with the obstacle by interpolation
            interpolation_rate = self.step_size * 0.125 # how many points we want to interpolate can be adjusted
            num_interpolation_points = int(np.ceil(distance / interpolation_rate)) # distance of line / step_size = num of points
            num_interpolation_points =  max(num_interpolation_points,1) # always check the collision status for the new_node 

            for p in range(1,num_interpolation_points + 1): # proportion on the straight line decide where to interpolate
                interpolated_point = node.state + difference_mapping *(float(p) / num_interpolation_points)
                for j in self.periodic_index:
                    interpolated_point[j] = (interpolated_point[j] + np.pi) % (2 * np.pi) - np.pi 
                for j in range(len(interpolated_point)):
                    if j not in self.periodic_index:
                        interpolated_point[j] = np.minimum(np.maximum(interpolated_point[j],self.joint_lower_limits[j]),self.joint_upper_limits[j])
                # ensure all angles within the boundary
                if self._check_for_collision(interpolated_point): # True -- collision False -- collision-free
                    return False # collsion happens, this node is not counted as task completion

            return True # close enough to the goal node + collision-free indicates task completion

        return False # node is not close enough to the goal node, keep iteration to find other candidate nodes



    def _trace_path_from_start(self, node=None):
        """
        Traces a path from start to node, if provided, or the goal otherwise.

        :param node: The target Node at the end of the path. Defaults to
            self.goal
        :returns: A list of states (not Nodes!) beginning at the start state and
            ending at the goal state.
        """
        # FILL in your code here
        if node is None:
            node = self.goal # tracing the path starts from the goal node

        node_record_in_path = [] # initialize a list to store all RRT node state from the path beginning at the start state and
                                #  ending at the goal state
        current_node = node # initialize a pointer firstly point to the goal node. start of the path tracing

        # From goal node, trace back to the parent node layer by layer in a bottom-up fashion
        while current_node is not None:  # ONLY start node has no parent,other RRT nodes is created and obviously has a parent linked with
                                         # upon tracing to the start node, stop the iteration
            node_record_in_path.append(current_node.state.tolist()) 
            current_node = current_node.parent # start next iteration from the parent node, trace one layer up

        node_record_in_path.reverse() # adjust list of states beginning at the start state and ending at the goal state
        return node_record_in_path

    def _check_for_collision(self, sample):
        """
        Checks if a sample point is in collision with any collision object.

        :returns: A boolean value indicating that sample is in collision.
        """
        if self.ada_collision_constraint is None:
            return False
        return self.ada_collision_constraint.is_satisfied(
            self.ada.get_arm_state_space(),
            self.ada.get_arm_skeleton(), sample)


def main(is_sim):
    
    if not is_sim:
        from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
        roscpp_init('adarrt', [])

    # instantiate an ada
    ada = adapy.Ada(is_sim)

    armHome = [-1.5, 3.22, 1.23, -2.19, 1.8, 1.2]
    goalConfig = [-2.37, 3.81, 1.31, -0.05, 0.78, 0.25]
    delta = 0.25
    eps = 1.0

    if is_sim:
        ada.set_positions(goalConfig)
    else:
        raw_input("Please move arm to home position with the joystick. " +
            "Press ENTER to continue...")


    # launch viewer
    viewer = ada.start_viewer("dart_markers/simple_trajectories", "map")

    # add objects to world
    canURDFUri = "package://pr_assets/data/objects/can.urdf"
    sodaCanPose = [0.25, -0.35, 0.0, 0, 0, 0, 1]
    tableURDFUri = "package://pr_assets/data/furniture/uw_demo_table.urdf"
    tablePose = [0.3, 0.0, -0.7, 0.707107, 0, 0, 0.707107]
    world = ada.get_world()
    can = world.add_body_from_urdf(canURDFUri, sodaCanPose)
    table = world.add_body_from_urdf(tableURDFUri, tablePose)

    # add collision constraints
    collision_free_constraint = ada.set_up_collision_detection(
            ada.get_arm_state_space(),
            ada.get_arm_skeleton(),
            [can, table])
    full_collision_constraint = ada.get_full_collision_constraint(
            ada.get_arm_state_space(),
            ada.get_arm_skeleton(),
            collision_free_constraint)

    # easy goal
    adaRRT = AdaRRT(
        start_state=np.array(armHome),
        goal_state=np.array(goalConfig),
        ada=ada,
        ada_collision_constraint=full_collision_constraint,
        step_size=delta,
        goal_precision=eps)

    rospy.sleep(1.0)

    if not is_sim:
        ada.start_trajectory_executor()

    path = adaRRT.build()
    if path is not None:
        print("Path waypoints:")
        print(np.asarray(path))
        waypoints = []
        for i, waypoint in enumerate(path):
            waypoints.append((0.0 + i, waypoint))

        t0 = time.clock()
        traj = ada.compute_smooth_joint_space_path(
            ada.get_arm_state_space(), waypoints, None)
        t = time.clock() - t0
        print(str(t) + "seconds elapsed")
        raw_input('Press ENTER to execute trajectory and exit')
        ada.execute_trajectory(traj)
        rospy.sleep(10.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', dest='is_sim', action='store_true')
    parser.add_argument('--real', dest='is_sim', action='store_false')
    parser.set_defaults(is_sim=True)
    args = parser.parse_args()
    main(args.is_sim)
