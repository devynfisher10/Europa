
from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward, AlignBallGoal, VelocityReward
from rlgym.utils.common_values import BALL_MAX_SPEED, CAR_MAX_SPEED, BLUE_GOAL_BACK, \
    BLUE_GOAL_CENTER, ORANGE_GOAL_BACK, ORANGE_GOAL_CENTER, BLUE_TEAM, ORANGE_TEAM, BACK_WALL_Y, BALL_RADIUS
#from rlgym_tools.extra_rewards import JumpTouchReward
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward




class VelocityBallToGoalReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM :
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = objective - state.ball.position
        
        # Regular component velocity
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        norm_vel = vel / BALL_MAX_SPEED

        multiplier_ball_height = 1
        if state.ball.position[2] > BALL_RADIUS*2:
            multiplier_ball_height = 1.25

        return multiplier_ball_height*float(np.dot(norm_pos_diff, norm_vel))


class KickoffReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.kickoff_state=False
        self.used_first_flip_kickoff=0
        self.land_kickoff=0
        self.used_second_flip_kickoff=0
        self.post_kickoff_counter=0
        self.kickoff_first_touch=0

    def reset(self, initial_state: GameState):
        self.kickoff_state=False
        self.used_first_flip_kickoff=0
        self.land_kickoff=0
        self.used_second_flip_kickoff=0
        self.post_kickoff_counter=0
        self.kickoff_first_touch=0

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:

        # get state data
        car_position = player.car_data.position
        car_vel = player.car_data.linear_velocity
        ball_position = state.ball.position
        ball_vel = state.ball.linear_velocity

        reward_kickoff = 0
        reward_post_kickoff = 0
        if self.kickoff_state or (ball_position[0] == 0 and ball_position[1] == 0):
            self.kickoff_state = True
            vel = car_vel
            pos_diff = ball_position - car_position
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / CAR_MAX_SPEED
            reward_vel_to_ball = float(np.dot(norm_pos_diff, norm_vel))
            # while approaching ball, kickoff reward is just vel to ball
            reward_kickoff = reward_vel_to_ball

            if (self.used_first_flip_kickoff == 0) and (not player.on_ground):
                self.used_first_flip_kickoff = 1
            elif (self.used_first_flip_kickoff == 1) and (player.on_ground):
                self.land_kickoff = 1
            elif (self.used_first_flip_kickoff == 1) and (self.land_kickoff == 1) and (not player.on_ground) and (player.ball_touched):
                self.used_second_flip_kickoff = 1

            # contact has been made on kickoff
            if not (ball_position[0] == 0 and ball_position[1] == 0):
                if player.ball_touched:
                    self.kickoff_first_touch = 1
                self.last_kickoff_vel = self.prev_player_vel
                self.kickoff_state = False
                self.post_kickoff_counter=1
            # if still in kickoff state, just reward kickoff reward and end call to rewards
            else:
                return float(reward_kickoff)
        # if after kickoff, keep count and give final reward on 8th step
        elif self.post_kickoff_counter > 0:
            self.post_kickoff_counter = self.post_kickoff_counter + 1
            # evaluate kickoff results at 8 steps after first contact (about half a second)
            if self.post_kickoff_counter == 8:
                post_kickoff_ball_velocity = reward_velocity_ball_to_goal 
                reward_post_kickoff = .5*self.used_first_flip_kickoff + .5*self.used_second_flip_kickoff
                + post_kickoff_ball_velocity + np.linalg.norm(self.last_kickoff_vel) / CAR_MAX_SPEED
                + self.kickoff_first_touch
                #reset variables
                self.post_kickoff_counter = 0
                self.used_first_flip_kickoff=0
                self.used_second_flip_kickoff = 0
        # total kickoff reward is either the vel to ball during kickoff, or the lump sum reward 8 steps after kickoff
        # reward kickoff ranges [-1, 1] during kickoff and up to 4 at post kickoff
        reward_kickoff = reward_kickoff + reward_post_kickoff

        return reward_kickoff


class TouchVelChange(RewardFunction):
    """Reward for changing velocity on ball"""
    def __init__(self):
        self.last_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.last_vel = np.zeros(3)
    
    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.ball_touched:
            vel_difference = abs(np.linalg.norm(self.last_vel - state.ball.linear_velocity))
            reward = 1.0*vel_difference / BALL_MAX_SPEED

        self.last_vel = state.ball.linear_velocity


        multiplier_touch_height = 1.0 + (state.ball.position[2] / 4000.0)**.4

        return reward*multiplier_touch_height



class JumpTouchReward(RewardFunction):
    def __init__(self, min_height=92.75):
        self.min_height = min_height
        self.max_height = 2044-92.75
        self.range = self.max_height - self.min_height

    def reset(self, initial_state: GameState):
        pass
    
    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            # adding const min to reward, from .02 at 1.2B
            return (state.ball.position[2] - self.min_height) / self.range

        return 0


class ExploreAir(RewardFunction):
    """Rewards exploring certain height once per episode away from walls"""
    def __init__(self): 
        self.explores_air = 0

    def reset(self, initial_state: GameState):
        self.explores_air = 0
    
    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        car_position = player.car_data.position
        if (self.explores_air==0) and (not player.on_ground and abs(car_position[1]) <= 4000 and abs(car_position[0]) <= 3000) and car_position[2] > 642.775:
            self.explores_air = 1
            return 1
        return 0


class AerialReward(RewardFunction):
    """Rewards every step car is in air and away from walls."""
    def __init__(self, min_height=25): # from 25
        self.min_height = min_height
        self.max_height = 2044-92.75

    def reset(self, initial_state: GameState):
        pass
    
    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        # reward if car off ground above min height and away from any walls
        car_vel = player.car_data.linear_velocity
        if not player.on_ground and abs(player.car_data.position[1]) <= 4000 and abs(player.car_data.position[0]) <= 3000: # and self.prev_has_flip and not player.has_flip:
            multiplier_z_vel = 1 + ((max(car_vel[2], 0) / CAR_MAX_SPEED))**.2
            # ranges from [1] * [1, 2]
            return 1 * multiplier_z_vel
        return 0


class BoostInAir(RewardFunction):
    """Rewards boosting in air"""
    def __init__(self): #
        self.last_registered_boost = {}

    def reset(self, initial_state: GameState):
        self.last_registered_boost = {}
        for player in initial_state.players:
            self.last_registered_boost[player.car_id] = player.boost_amount
    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        car_position = player.car_data.position
        boost_diff = player.boost_amount - self.last_registered_boost[player.car_id]
        reward_boost_in_air = 0
        if (car_position[2] > 2 * BALL_RADIUS) and (not player.on_ground):
            # range is [0, 1], but really smaller bc can't use all 100 boost in one step. limited to how much boost can be consumed in a step
            reward_boost_in_air = -1*float(boost_diff)
        self.last_registered_boost[player.car_id] = player.boost_amount
        return reward_boost_in_air


class DoubleTapReward(RewardFunction):
    """Class to reward agent for behavior related to double taps. Agent gets reward if making air touch after ball hits backboard before the ball hits the ground"""
    def __init__(self):
        # off_backboard and air_touch vars initialized to False. Set to true when each event occurrs, set back to false when ball touches ground
        self.off_backboard = False
        self.first_air_touch = False
        self.second_air_touch = False
        self.min_height = BALL_RADIUS + 5
        self.min_backboard_height = 500 # top of goal is 642.775, changed from 250
        self.min_car_dist_from_backboard = BALL_RADIUS*6 
        self.num_steps = 0
        self.prev_ball_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.off_backboard = False
        self.first_air_touch = False
        self.second_air_touch = False
        self.num_steps = 0
        self.prev_ball_vel = np.zeros(3)

    
    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        # reward if shot in air before hitting ground after backboard touch
        reward = 0

        # need to account for attacking backboard based on team
        if player.team_num == BLUE_TEAM:
            ball_position = state.ball.position
        else:
            ball_position = state.inverted_ball.position

        # need to account for order, only reward for backboard -> air touch not other way around otherwise that just rewards hitting ball into backboard

        # if ball hits backboard, and previous ball velocity was mostly towards the backboard, set off_backboard value. Requires height to be in air
        if (ball_position[2] >= self.min_backboard_height) and (abs(self.prev_ball_vel[1]) > max(abs(self.prev_ball_vel[0]) + (abs(self.prev_ball_vel[2])), 20)) and (ball_position[1] >= BACK_WALL_Y - BALL_RADIUS - 10):
            self.off_backboard=True

        # if make air touch, set first_air_touch value. Only set this value if has not yet touched the backboard 
        if (not self.off_backboard) and (player.ball_touched) and (ball_position[2] >= self.min_height):
            # adding checks to make sure car is a min distance from wall when making touches to prevent dribbling on wall
            if (player.team_num == ORANGE_TEAM) and (abs(-1.0*BACK_WALL_Y - player.car_data.position[1]) >= self.min_car_dist_from_backboard):
                self.first_air_touch=True
            elif (player.team_num == BLUE_TEAM) and (abs(1.0*BACK_WALL_Y - player.car_data.position[1]) >= self.min_car_dist_from_backboard):
                self.first_air_touch=True


        # if make air touch, set second_air_touch value. Only set this value if has already touched the backboard 
        if self.off_backboard and player.ball_touched and ball_position[2] >= self.min_height:
            # adding checks to make sure car is a min distance from wall when making touches to prevent dribbling on wall
            if (player.team_num == ORANGE_TEAM) and (abs(-1.0*BACK_WALL_Y - player.car_data.position[1]) >= self.min_car_dist_from_backboard):
                self.second_air_touch=True
            elif (player.team_num == BLUE_TEAM) and (abs(1.0*BACK_WALL_Y - player.car_data.position[1]) >= self.min_car_dist_from_backboard):
                self.second_air_touch=True

        if self.off_backboard and self.second_air_touch and self.num_steps < 5:
            if player.team_num == BLUE_TEAM:
                objective = np.array(ORANGE_GOAL_BACK)
            else:
                objective = np.array(BLUE_GOAL_BACK)
            vel = state.ball.linear_velocity
            pos_diff = objective - state.ball.position
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            if np.linalg.norm(vel) > 0:
                norm_vel = vel / np.linalg.norm(vel)
            else:
                norm_vel = vel
            dot = float(np.dot(norm_pos_diff,norm_vel))
            # check to see if velocity of ball towards goal is positive after final touch, only give reward if true
            if dot > .75:
                reward=1
                # 3x reward if get full double tap instead of just backboard read
                if self.first_air_touch:
                    reward = 3
            # increment steps to make sure only rewards for initial hit + follow up
            self.num_steps += 1

        # if ball hits ground, reset conditions and no reward
        # reset in 5 steps? Avoids cheat code of hitting off backboard then dribbling ball around on top of car to get max continual reward
        if ball_position[2] < self.min_height or self.num_steps > 5:
            self.first_air_touch=False
            self.second_air_touch=False
            self.off_backboard=False
 
        self.prev_ball_vel = state.ball.linear_velocity

        return reward



class AirDribbleReward(RewardFunction):
    """Class to reward agent for air dribbles. Gets progressively rewarded for behavior closer and closer to ideal air dribble behavior."""
    def __init__(self):
        # off_backboard and air_touch vars initialized to False. Set to true when each event occurrs, set back to false when ball touches ground
        self.wall_touch = False
        self.off_sidewall = False
        self.towards_ball = False
        self.first_air_touch = False
        self.second_air_touch = False
        self.min_wall_touch_height = 100
        self.min_air_dribble_height = 500 # top of goal is 642.775
        self.min_pos_diff = 300
        self.min_vel_diff = 400
        self.num_steps = 0
        self.max_height = 2044-92.75
        self.num_steps = 0


    def reset(self, initial_state: GameState):
        self.wall_touch = False
        self.off_sidewall = False
        self.towards_ball = False
        self.first_air_touch = False
        self.second_air_touch = False
        self.num_steps = 0
        self.num_steps = 0

    
    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        # reward starts at 0
        reward = 0

        # need to account for attacking goal based on team
        if player.team_num == BLUE_TEAM:
            ball_position = state.ball.position
            ball_vel = state.ball.linear_velocity
        else:
            ball_position = state.inverted_ball.position
            ball_vel = state.inverted_ball.linear_velocity


        # set wall_touch to true if touching ball while car is on wall
        if player.ball_touched and player.on_ground and ball_position[2] >= self.min_wall_touch_height and abs(ball_position[1]) <= 3500 and self.num_steps < 15: # backboard is 5120
            self.wall_touch = True
            self.num_steps = self.num_steps + 1
            reward = max(player.boost_amount, .33)*.2*((ball_position[2] - self.min_wall_touch_height)/self.max_height)**.5 # diminshing returns, increasing rewards for higher touches on wall, from .1 at 1.25

        # add reward if hits ball off sidewall up into air towards goal. only true if already wall touch
        if not self.off_sidewall and self.wall_touch and ball_vel[2] > 0 and ball_vel[1] > 0 and abs(ball_position[0]) < 4096 - BALL_RADIUS*2:
            self.num_steps = 0
            self.off_sidewall = True
            reward = max(player.boost_amount, .33)*.25

        # add reward if car off wall towards ball
        if self.wall_touch and self.off_sidewall and not player.on_ground:
            pos_diff = state.ball.position - player.car_data.position
            vel_diff = state.ball.linear_velocity - player.car_data.linear_velocity
            norm_pos_diff = np.linalg.norm(pos_diff)
            norm_vel_diff = np.linalg.norm(vel_diff)
            normed_pos_diff = pos_diff / norm_pos_diff
            normed_vel_diff = vel_diff / norm_vel_diff
            vel_to_ball = float(np.dot(normed_pos_diff, normed_vel_diff))
            # reward as long as stays close to ball and with a similar velocity as ball
            if norm_pos_diff < self.min_pos_diff and norm_vel_diff < self.min_vel_diff and vel_to_ball > .2 and self.num_steps < 30:
                self.num_steps = self.num_steps + 1
                self.towards_ball = True
                reward = .15

        # add reward if hits ball in air after pop
        if not self.first_air_touch and self.wall_touch and self.off_sidewall and self.towards_ball and player.ball_touched and not player.on_ground and ball_position[2] >=  self.min_air_dribble_height:
            self.first_air_touch = True
            reward = 1.5

        # add reward for each following air touch
        if self.wall_touch and self.off_sidewall and self.towards_ball and self.first_air_touch and  player.ball_touched and not player.on_ground and ball_position[2] >=  self.min_air_dribble_height:
            self.second_air_touch = True
            reward = 2


        # add extra reward if ball going towards goal
        if self.wall_touch and self.off_sidewall and self.towards_ball and self.second_air_touch:
            if player.team_num == BLUE_TEAM:
                objective = np.array(ORANGE_GOAL_BACK)
            else:
                objective = np.array(BLUE_GOAL_BACK)
            vel = state.ball.linear_velocity
            pos_diff = objective - state.ball.position
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            if np.linalg.norm(vel) > 0:
                norm_vel = vel / np.linalg.norm(vel)
            else:
                norm_vel = vel
            # only care about going towards back of net in x, y plane. z coordinate can vary.
            dot = float(norm_pos_diff[0]*norm_vel[0] + norm_pos_diff[1]*norm_vel[1])
            if dot > .75:
                reward = reward + 1


        # if ball hits ground, reset conditions and no reward
        if ball_position[2] < self.min_wall_touch_height or self.num_steps > 5:
            self.wall_touch = False
            self.off_sidewall = False
            self.towards_ball = False
            self.first_air_touch = False
            self.num_steps = 0
            reward = 0
 

        return reward


class GoalSpeedAndPlacementReward(RewardFunction):
    def __init__(self):
        self.prev_score_blue = 0
        self.prev_score_orange = 0
        self.prev_state_blue = GameState(None)
        self.prev_state_orange = GameState(None)
        self.min_height = BALL_RADIUS + 10
        self.height_reward = 1.75

    def reset(self, initial_state: GameState):
        self.prev_score_blue = initial_state.blue_score
        self.prev_score_orange = initial_state.orange_score
        self.prev_state_blue = initial_state
        self.prev_state_orange = initial_state

    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            score = state.blue_score
            # check to see if goal scored
            if score > self.prev_score_blue:
                reward = np.linalg.norm(self.prev_state_blue.ball.linear_velocity) / BALL_MAX_SPEED
                if self.prev_state_blue.ball.position[2] > self.min_height:
                    reward = self.height_reward * reward
                self.prev_state_blue = state
                self.prev_score_blue = score
                return reward
            else:
                self.prev_state_blue = state
                return 0.0
        else:
            score = state.orange_score
            # check to see if goal scored
            if score > self.prev_score_orange:
                reward = np.linalg.norm(self.prev_state_orange.ball.linear_velocity) / BALL_MAX_SPEED
                if self.prev_state_orange.ball.position[2] > self.min_height:
                    reward = self.height_reward * reward
                self.prev_state_orange = state
                self.prev_score_orange = score
                return reward
            else:
                self.prev_state_orange = state
                return 0.0