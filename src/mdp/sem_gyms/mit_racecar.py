import logging
import os
from typing import List

URDF_PATH = "racecar/racecar_differential.urdf"

_logger = logging.getLogger("mit-racecar")

class CarOptions:
    start_position:list
    start_orientation:list

class ConstraintConfig:
    value_1:int
    value_2:int
    gear_ratio:int
    gear_aux_link:int

    def __init__(self, value_1, value_2, gear_ratio, gear_aux_link=None) -> None:
        self.value_1 = value_1
        self.value_2 = value_2
        self.gear_ratio = gear_ratio
        self.gear_aux_link = gear_aux_link

class MitRacecar:
    def __init__(self, bullet_client, urdf_root_path, start_pos, start_orn, time_step=0.01) -> None:
        self.bullet_client = bullet_client
        self.urdf_root_path = urdf_root_path
        self.start_pos = start_pos
        self.start_orn = start_orn
        self.time_step = time_step

        self.n_motors = 2
        self.max_force = 20
        self.steering_links = [0, 2]
        self.motorized_wheels = [8, 15]
        self.speed_multiplier = 20.
        self.steering_multiplier = 0.5

        self.constraints = []
        self.constraints.append(ConstraintConfig(9, 11, 1))
        self.constraints.append(ConstraintConfig(10, 13, -1))
        self.constraints.append(ConstraintConfig(9, 13, -1))
        self.constraints.append(ConstraintConfig(16, 18, 1))
        self.constraints.append(ConstraintConfig(16, 19, -1))
        self.constraints.append(ConstraintConfig(17, 19, -1))
        self.constraints.append(ConstraintConfig(1, 18, -1, 15))
        self.constraints.append(ConstraintConfig(3, 19, -1, 15))

    def update_constraints(self, car, constraints:List[ConstraintConfig]):
        joint_type = self.bullet_client.JOINT_GEAR
        joint_axis = [0, 1, 0]
        parent_frame_position = [0, 0, 0]
        child_frame_position = [0, 0, 0]
        max_force = 10000

        for config in constraints:
            constraint = self.bullet_client.createConstraint(
                car,
                config.value_1,
                car,
                config.value_2,
                jointType = joint_type,
                jointAxis = joint_axis,
                parentFramePosition = parent_frame_position,
                childFramePosition = child_frame_position
            )

            if config.gear_aux_link == None:
                self.bullet_client.changeConstraint(constraint, gearRatio=config.gear_ratio, maxForce=max_force)
            else:
                self.bullet_client.changeConstraint(constraint, gearRatio=config.gear_aux_link, maxForce=max_force)

    def reset(self, options:CarOptions=None):
        start_position = self.start_pos if options == None else options.start_position
        start_orientation = self.start_orn if options == None else options.start_orientation

        car = self.bullet_client.loadURDF(
            os.path.join(self.urdf_root_path, URDF_PATH),
            basePosition=start_position,
            baseOrientation=start_orientation,
            useFixedBase=False
        )

        self.racecarUniqueId = car

        for wheel in range(self.bullet_client.getNumJoints(car)):
            self.bullet_client.setJointMotorControl2(
                car,
                wheel,
                self.bullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )

        self.update_constraints(car, self.constraints)

    def apply_action(self, motor_commands):
        # _logger.info(f"motorCommands: {motor_commands}")

        target_velocity = motor_commands[0] * self.speed_multiplier
        steering_angle = motor_commands[1] * self.steering_multiplier
        # _logger.info(f"target_velocity: {target_velocity}")
        # _logger.info(f"steering_angle: {steering_angle}")

        for motor in self.motorized_wheels:
            self.bullet_client.setJointMotorControl2(
                self.racecarUniqueId,
                motor,
                self.bullet_client.VELOCITY_CONTROL,
                targetVelocity=target_velocity,
                force=self.max_force
            )

        for steer in self.steering_links:
            self.bullet_client.setJointMotorControl2(
                self.racecarUniqueId,
                steer,
                self.bullet_client.POSITION_CONTROL,
                targetPosition=steering_angle
            )
