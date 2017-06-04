import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import pygame

class PaddleBall(object):
    def __init__(self, window_w, window_h):
        self.window_w = window_w
        self.window_h = window_h
        # Select random horizontal co-ordinate for the ball
        self.ball_x = np.random.randint(0, self.window_w-1, size=1)[0]
        # Initial vertical co-ordinate for the ball is "top"
        self.ball_y = 0
        # Select random horizontal direction to left or right for the ball
        self.ball_dir_x = np.random.choice([-1,1], size=1)[0]
        # Initial vertical direction for the ball is "down"
        self.ball_dir_y = 1
        # Select random horizontal co-ordinate for the paddle
        self.paddle_x = np.random.randint(1, self.window_w-2, size=1)[0]
        # Vertical co-ordinate for the paddle is always "bottom"
        self.paddle_y = self.window_h-1
        self.game_over = False

    def get_state(self):
        # Create array of windows size and fill with zeros
        state = np.zeros((self.window_h, self.window_w), dtype=np.int)
        # Fill ones for ball
        state[self.ball_y, self.ball_x] = 1
        # Fill ones for paddle
        state[self.paddle_y, self.paddle_x-1] = 1
        state[self.paddle_y, self.paddle_x] = 1
        state[self.paddle_y, self.paddle_x+1] = 1
        return state.reshape((1, -1))

    def take_action(self, action):
        reward = 0
        
        # Move the paddle
        # The action will be [0,1,2]. Assume 0=left, 1=right, 2=stay.
        if action == 0:  # left
            self.paddle_x = max(self.paddle_x-1, 1)
        elif action == 1:  # right
            self.paddle_x = min(self.paddle_x+1, self.window_w-2)
        
        # If paddle aligns with ball it is likely to bounce the ball.
        # Encourage this behavior by rewarding it.
        if self.ball_x >= self.paddle_x-1 and self.ball_x <= self.paddle_x+1:
            reward = 1

        # Move the ball. Check vertical limits
        if self.ball_y == self.window_h-2: # Ball is at level with paddle
            if self.ball_x >= self.paddle_x-1 and self.ball_x <= self.paddle_x+1: # Ball is on the Paddle
                self.ball_y = self.ball_y - 1 # Bounce the ball
                self.ball_dir_y = 0 # Change vertical direction of the ball to "up"
                reward = 2 # Get reward for bouncing ball off the paddle
            else: # Ball hits the ground
                self.ball_y = self.ball_y + 1 # Ball hits the ground
                reward = -1 # Punish for missing ball
                self.game_over = True
        elif self.ball_y == 0: # Ball hits top
            self.ball_y = self.ball_y + 1 # Bounce the ball
            self.ball_dir_y = 1 # Change vertical direction of the ball to "down"
        else: # Ball is in air
            if self.ball_dir_y == 1: # Ball is moving down
                self.ball_y = self.ball_y + 1 # Move ball down by 1 unit
            else: # Ball is moving up
                self.ball_y = self.ball_y - 1 # Move ball up by 1 unit
        
        # Move the ball. Check horizontal limits
        if self.ball_x == 0: # Ball hits left wall
            self.ball_x = self.ball_x + 1 # Bounce the ball
            self.ball_dir_x = 1 # Change horizontal direction of the ball to "right"
        elif self.ball_x == self.window_w-1: # Ball hits right wall
            self.ball_x = self.ball_x - 1 # Bounce the ball
            self.ball_dir_x = -1 # Change horizontal direction of the ball to "left"
        else: # Ball is in air
            if self.ball_dir_x == 1: # Ball is moving right
                self.ball_x = self.ball_x + 1 # Move ball right by 1 unit
            else: # Ball is moving left
                self.ball_x = self.ball_x - 1 # Move ball left by 1 unit
        
        return reward

    def reset(self):
        # Select random horizontal co-ordinate for the ball
        self.ball_x = np.random.randint(0, self.window_w-1, size=1)[0]
        # Initial vertical co-ordinate for the ball is "top"
        self.ball_y = 0
        # Select random horizontal direction to left or right for the ball
        self.ball_dir_x = np.random.choice([-1,1], size=1)[0]
        # Initial vertical direction for the ball is "down"
        self.ball_dir_y = 1
        # Select random horizontal co-ordinate for the paddle
        self.paddle_x = np.random.randint(1, self.window_w-2, size=1)[0]
        # Vertical co-ordinate for the paddle is always "bottom"
        self.paddle_y = self.window_h-1
        self.game_over = False


class ExperienceReplay(object):
    def __init__(self, max_memory=500, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=50):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            
            # Compute Discounted Future Reward only for this action.
            # Don't compute it for other actions. 
            # This way error for other actions will be zero and weights will not be adjusted for them.
            # Network will only learn for this action.
            targets[i] = model.predict(state_t)[0]
            
            # Predict reward for next state for this action.
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


if __name__ == "__main__":
    # Exploration probability
    epsilon = .1  
    # Number of times to play game for training
    epoch = 1000
    # Memory for Experience Replay
    max_memory = 1000
    # Batch size to train with Experience Replay
    batch_size = 100
    # Game screen width units
    window_w = 12
    # Game screen hight units
    window_h = 10
    # Hidden units in neural net
    hidden_size = 120
    # [move_left, move_right, dont_move]
    num_actions = 3

    # Build 3 layer neural net.
    # 120 input neurons, 120 hidden neurons, and 3 output neurons
    model = Sequential()
    # Activation used in Linear Rectifier for better performance.
    model.add(Dense(hidden_size, input_shape=(window_h*window_w,), activation="relu"))
    model.add(Dense(hidden_size, activation="relu"))
    model.add(Dense(num_actions))
    # Optimizer used is Stochastic Gradient Descent with loss function Mean Squared Error.
    model.compile(sgd(lr=.2), "mse")

    # To continue training a previous model, uncomment the following line.
    #model.load_weights("model.h5")

    # Define environment/game
    env = PaddleBall(window_w, window_h)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)
    
#==============================================================================
#     pygame.init()
#     scale = 40
#     display_width = window_w*scale
#     display_height = window_h*scale
#     gameDisplay = pygame.display.set_mode((display_width,display_height))
#     pygame.display.set_caption("Paddle Ball")
#     black = (0,0,0)
#     white = (255,255,255)
#     #clock = pygame.time.Clock()
#     pygame.time.set_timer(pygame.USEREVENT, 100) # Trigger event every 100 msec.
#==============================================================================

    # Train
    win_cnt = 0
    for e in range(epoch):
        loss = 0.
        env.reset()
        game_over = False
        # Get initial input
        input_t = env.get_state()
        # Save state to text file.
        #with open("canvas.txt","ab") as f_handle:
        #    np.savetxt(f_handle, input_t.reshape(window_h,window_w), fmt="%i")
        
        while not game_over:
#==============================================================================
#             for event in pygame.event.get():
#                 if event.type == pygame.MOUSEBUTTONDOWN:
#                     game_over = True
#                 
#                 if event.type == pygame.USEREVENT and not game_over:
#==============================================================================
            
                    input_tm1 = input_t
                    # Get next action
                    # The action will be [0,1,2]. Assume 0=left, 1=right, 2=stay.
                    if np.random.rand() <= epsilon:
                        action = np.random.randint(0, num_actions, size=1)
                    else:
                        q = model.predict(input_tm1)
                        action = np.argmax(q[0])
        
                    # Apply action, get rewards and new state
                    reward = env.take_action(action)
                    game_over = env.game_over
                    input_t = env.get_state()
                    #with open("canvas.txt","ab") as f_handle:
                    #    np.savetxt(f_handle, input_t.reshape(window_h,window_w), fmt="%i")
                    if reward == 1:
                        win_cnt += 1
        
                    # Store experience
                    exp_replay.remember([input_tm1, action, reward, input_t], game_over)
        
                    # Adapt model
                    inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
        
                    loss_i = model.train_on_batch(inputs, targets)
                    loss += loss_i
                    
#==============================================================================
#                     gameDisplay.fill(white)
#                     # Draw ball
#                     pygame.draw.circle(gameDisplay, black, (env.ball_x*scale + scale//2, env.ball_y*scale + scale//2), scale//2)
#                     # Draw paddle
#                     pygame.draw.rect(gameDisplay, black, [(env.paddle_x-1)*scale, (env.window_h-1)*scale, 3*scale, scale])
#                     pygame.display.update()
#                     #clock.tick(60)
#==============================================================================
        print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, win_cnt))

    # Save trained model.
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

#==============================================================================
#     pygame.quit()
#==============================================================================
