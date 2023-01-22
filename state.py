
import random

import numpy as np
import math

from torch import arctan2, cross, sqrt, unsqueeze

from rlgym.utils.common_values import CAR_MAX_SPEED, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z, BALL_RADIUS, CAR_MAX_ANG_VEL, \
    BALL_MAX_SPEED


from numpy import random as rand

from rlgym.utils import StateSetter
from rlgym.utils.state_setters import DefaultState, StateWrapper, RandomState
from rlgym_tools.extra_state_setters.symmetric_setter import KickoffLikeSetter
from rlgym_tools.extra_state_setters.wall_state import WallPracticeState

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Z = CEILING_Z - BALL_RADIUS

PITCH_LIM = np.pi / 2
YAW_LIM = np.pi
ROLL_LIM = np.pi

GOAL_X_MAX = 800.0
GOAL_X_MIN = -800.0

PLACEMENT_BOX_X = 5000
PLACEMENT_BOX_Y = 2000
PLACEMENT_BOX_Y_OFFSET = 3000

GOAL_LINE = 5100

YAW_MAX = np.pi

# important for unstandardizing the obs
POS_STD = 2300 
ANG_STD = math.pi



# goal is to use all the extra tools state setters and a random one + maybe write one of my own (aerial shot state?) (air dribble / reset state?) 
# then give each a certain prob


class AerialPracticeState(StateSetter):

    def __init__(self, reset_to_max_boost=True):
        """
        AerialPracticeState constructor.
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or keep from last episode
        """
        super().__init__()
        self.team_turn = 0  # swap every reset which car is making the aerial play
        self.reset_to_max_boost = reset_to_max_boost

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to set a new aerial play
        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        self._reset_ball_and_cars(state_wrapper, self.team_turn, self.reset_to_max_boost)

        # which team will make the next aerial play
        self.team_turn = (self.team_turn + 1) % 2

    def _place_car_in_box_area(self, car, team_delin):
        """
        Function to place a car in an allowed area
        :param car: car to be modified
        :param team_delin: team number delinator to look at when deciding where to place the car
        """

        y_pos = (PLACEMENT_BOX_Y - (rand.random() * PLACEMENT_BOX_Y))

        if team_delin == 0:
            y_pos -= PLACEMENT_BOX_Y_OFFSET
        else:
            y_pos += PLACEMENT_BOX_Y_OFFSET

        car.set_pos(rand.random() * PLACEMENT_BOX_X - PLACEMENT_BOX_X / 2, y_pos, z=17)

    def _reset_ball_and_cars(self, state_wrapper: StateWrapper, team_turn, reset_to_max_boost):
        """
        Function to set a new ball in the air towards a goal
        :param state_wrapper: StateWrapper object to be modified.
        :param team_turn: team who's making the aerial play
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or keep from last episode
        """

        # reset ball
        pos, lin_vel, ang_vel = self._get_ball_parameters(team_turn)
        state_wrapper.ball.set_pos(pos[0], pos[1], pos[2])
        state_wrapper.ball.set_lin_vel(lin_vel[0], lin_vel[1], lin_vel[2])
        state_wrapper.ball.set_ang_vel(ang_vel[0], ang_vel[1], ang_vel[2])

        # reset cars relative to the ball
        first_set = False
        for car in state_wrapper.cars:
            # set random position and rotation for all cars based on pre-determined ranges

            if car.team_num == team_turn and not first_set:
                car_pos = (pos[0] * random.uniform(.9, 1.1), pos[1] * random.uniform(.9, 1.1), pos[2] * random.uniform(.9, 1.05))
                car.set_pos(car_pos[0], y=car_pos[1],  z=car_pos[2])
                pos_diff = pos-car_pos
                car_lin_vel = (lin_vel[0] * random.uniform(.9, 1.1), lin_vel[1] * random.uniform(.9, 1.1), lin_vel[2] * random.uniform(.9, 1.1))
                car.set_lin_vel(car_lin_vel[0] + pos_diff[0]*random.uniform(20, 50), car_lin_vel[1] + pos_diff[1]*random.uniform(20, 50), car_lin_vel[0] + pos_diff[1]*random.uniform(20, 50))
                first_set = True
            else:
                self._place_car_in_box_area(car, car.team_num)

            if reset_to_max_boost:
                car.boost = 100

            car.set_rot(0, rand.random() * YAW_MAX - YAW_MAX / 2, 0)


    def _get_ball_parameters(self, team_turn):
        """
        Function to set a new ball up for an aerial play
        
        :param team_turn: team who's making the aerial play
        """

        INVERT_IF_BLUE = (-1 if team_turn == 0 else 1)  # invert shot for blue

        # set positional values
        x_pos = random.uniform(GOAL_X_MIN, GOAL_X_MAX)
        y_pos = 4000 * random.uniform(-.1, 1) * INVERT_IF_BLUE
        z_pos = CEILING_Z * random.uniform(.3, 1)
        pos = np.array([x_pos, y_pos, z_pos])

        # set lin velocity values
        x_vel_randomizer = (random.uniform(-.5, .5))
        y_vel_randomizer = (random.uniform(-.1, .3))
        z_vel_randomizer = (random.uniform(.1, 1))

        x_vel = (750 * x_vel_randomizer)
        y_vel = (500 * y_vel_randomizer * INVERT_IF_BLUE)
        z_vel = (600 * z_vel_randomizer)
        lin_vel = np.array([x_vel, y_vel, z_vel])


        ang_vel = np.array([0, 0, 0])

        return pos, lin_vel, ang_vel







class ConfigurableState(StateSetter):

    def __init__(self, states=None, reset_to_max_boost=False):
        """
        ConfigurableState constructor. To be used with pivotal_states.
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or keep from last episode
        """
        super().__init__()
        self.states = states
        self.reset_to_max_boost = reset_to_max_boost
        self.mu=1
        self.sigma=.005

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to set a new state
        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        state_pick = random.choice(self.states)
        self._state_setup(state_wrapper, state_pick)

    def replace_states(self, new_states):
        self.states=new_states
        # print(f"replacing states with num new pivotal states = {len(new_states)}", flush=True)

    def _state_setup(self, state_wrapper: StateWrapper, state_pick):
        """
        Function to setup a state from obs
        :param state_wrapper: StateWrapper object to be modified with desired state values.
        :param state_pick: obs state that will be used by the state setter
        """

        team_side = 0 if random.randrange(2) == 1 else 1
        team_inverter = 1 if team_side == 0 else -1

        # add small normal noise along the way

        # set ball 
        ball_x_pos = state_pick[0]*POS_STD*np.random.normal(self.mu, self.sigma)*team_inverter
        ball_y_pos = state_pick[1]*POS_STD*np.random.normal(self.mu, self.sigma)*team_inverter
        ball_z_pos = state_pick[2]*POS_STD*np.random.normal(self.mu, self.sigma)
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = state_pick[3]*POS_STD*np.random.normal(self.mu, self.sigma)*team_inverter
        ball_y_vel = state_pick[4]*POS_STD*np.random.normal(self.mu, self.sigma)*team_inverter
        ball_z_vel = state_pick[5]*POS_STD*np.random.normal(self.mu, self.sigma)
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)

        ball_x_ang = state_pick[6]*ANG_STD*np.random.normal(self.mu, self.sigma)*team_inverter
        ball_y_ang = state_pick[7]*ANG_STD*np.random.normal(self.mu, self.sigma)*team_inverter
        ball_z_ang = state_pick[8]*ANG_STD
        state_wrapper.ball.set_ang_vel(ball_x_ang, ball_y_ang, ball_z_ang)
        

        # set chosen car
        chosen_car = [car for car in state_wrapper.cars if car.team_num == team_side][0]

        car_x_pos = state_pick[57]*POS_STD*np.random.normal(self.mu, self.sigma)*team_inverter
        car_y_pos = state_pick[58]*POS_STD*np.random.normal(self.mu, self.sigma)*team_inverter
        car_z_pos = state_pick[59]*POS_STD*np.random.normal(self.mu, self.sigma)
        chosen_car.set_pos(car_x_pos, car_y_pos, car_z_pos)

        car_x_vel = state_pick[69]*POS_STD*np.random.normal(self.mu, self.sigma)*team_inverter
        car_y_vel = state_pick[70]*POS_STD*np.random.normal(self.mu, self.sigma)*team_inverter
        car_z_vel = state_pick[71]*POS_STD*np.random.normal(self.mu, self.sigma)
        chosen_car.set_lin_vel(car_x_vel, car_y_vel, car_z_vel)

        car_x_ang = state_pick[72]*ANG_STD*np.random.normal(self.mu, self.sigma)*team_inverter
        car_y_ang = state_pick[73]*ANG_STD*np.random.normal(self.mu, self.sigma)*team_inverter
        car_z_ang = state_pick[74]*ANG_STD*np.random.normal(self.mu, self.sigma)
        chosen_car.set_ang_vel(car_x_ang, car_y_ang, car_z_ang)

        chosen_car.boost = state_pick[75]
        # print(f"new state setup with car_x_pos = {car_x_pos}, car_y_pos = {car_y_pos}, car_z_pos = {car_z_pos}", flush=True)


        # set not chosen car
        not_chosen_car = [car for car in state_wrapper.cars if car.team_num != team_side][0]

        car_x_pos = state_pick[86]*POS_STD*np.random.normal(self.mu, self.sigma)*team_inverter*(-1)*team_inverter
        car_y_pos = state_pick[87]*POS_STD*np.random.normal(self.mu, self.sigma)*team_inverter*(-1)*team_inverter
        car_z_pos = state_pick[88]*POS_STD*np.random.normal(self.mu, self.sigma)
        not_chosen_car.set_pos(car_x_pos, car_y_pos, car_z_pos)

        car_x_vel = state_pick[98]*POS_STD*np.random.normal(self.mu, self.sigma)*team_inverter*(-1)*team_inverter
        car_y_vel = state_pick[99]*POS_STD*np.random.normal(self.mu, self.sigma)*team_inverter*(-1)*team_inverter
        car_z_vel = state_pick[100]*POS_STD*np.random.normal(self.mu, self.sigma)
        not_chosen_car.set_lin_vel(car_x_vel, car_y_vel, car_z_vel)

        car_x_ang = state_pick[101]*ANG_STD*np.random.normal(self.mu, self.sigma)*team_inverter*(-1)*team_inverter
        car_y_ang = state_pick[102]*ANG_STD*np.random.normal(self.mu, self.sigma)*team_inverter*(-1)*team_inverter
        car_z_ang = state_pick[103]*ANG_STD*np.random.normal(self.mu, self.sigma)
        not_chosen_car.set_ang_vel(car_x_ang, car_y_ang, car_z_ang)

        not_chosen_car.boost = state_pick[104]


        #  invert yaw, pitch roll is (+,-,+)
        chosen_car.set_rot(state_pick[66]*team_inverter, state_pick[67]*team_inverter, state_pick[68]*team_inverter)
        not_chosen_car.set_rot(state_pick[95], state_pick[96]*(-1)*team_inverter, state_pick[97])

        








# setting initial probabilities for each state here. Will update over time as increase prob of complex states for curriculum learning.
class EuropaStateSetter(StateSetter):
    def __init__(
            self, *,
            default_prob=6/10,
            wall_prob=.5/10,
            kickofflike_prob=0/10,
            random_prob=1.5/10,
            aerial_prob=2/10

            #added for testing
            # default_prob=10/10,
            # wall_prob=0/10,
            # kickofflike_prob=0/10,
            # random_prob=0/10,
            # aerial_prob=0/10
    ):

        super().__init__()


        self.setters = [
            DefaultState(),
            WallPracticeState(air_dribble_odds=7/10, backboard_roll_odds=1/10, side_high_odds=2/10),
            KickoffLikeSetter(),
            RandomState(),
            AerialPracticeState(),
            ConfigurableState(),
        ]
        self.probs = np.array([default_prob, wall_prob, kickofflike_prob, random_prob, aerial_prob, 0/10])

        assert self.probs.sum() == 1, f"Probabilities must sum to 1 instead of {self.probs.sum()}"

    def reset(self, state_wrapper: StateWrapper):
        i = np.random.choice(len(self.setters), p=self.probs)
        self.setters[i].reset(state_wrapper)


    def modify_states(self, pivotal_states=None, replacement_index=0, new_prob=.4):
        if pivotal_states is not None:
            # configure setter with new states and give probability from old states if at 0
            # print(f"modifying states with num new pivotal states = {len(pivotal_states)}", flush=True)

            # new_prob=1 #added for testing
            self.setters[-1].replace_states(pivotal_states)
            if self.probs[-1] == 0:
                # print(f"replacing probs with new probs = {new_prob}", flush=True)
                assert self.probs[replacement_index] >= new_prob, "Probability in replacement index must be >= new_prob"
                self.probs[replacement_index] = self.probs[replacement_index] - new_prob
                self.probs[-1] = new_prob





