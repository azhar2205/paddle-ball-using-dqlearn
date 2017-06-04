import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import pygame
from train_paddle_ball import PaddleBall


if __name__ == "__main__":
    epoch = 1
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

    model.load_weights("model.h5")

    # Define environment/game
    env = PaddleBall(window_w, window_h)

    pygame.init()
    scale = 40
    display_width = window_w*scale
    display_height = window_h*scale
    gameDisplay = pygame.display.set_mode((display_width,display_height))
    pygame.display.set_caption("Paddle Ball")
    black = (0,0,0)
    white = (255,255,255)
    #clock = pygame.time.Clock()
    pygame.time.set_timer(pygame.USEREVENT, 100) # Trigger event every 100 msec.

    # Test
    win_cnt = 0
    for e in range(epoch):
        loss = 0.
        env.reset()
        game_over = False
        # Get initial input
        input_t = env.get_state()
        # Save state to text file.
        #with open("state.txt","ab") as f_handle:
        #    np.savetxt(f_handle, input_t.reshape(window_h,window_w), fmt="%i")
    
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    game_over = True
                
                if event.type == pygame.USEREVENT and not game_over:
            
                    input_tm1 = input_t
                    # Get next action
                    # The action will be [0,1,2]. Assume 0=left, 1=right, 2=stay.
                    q = model.predict(input_tm1)
                    action = np.argmax(q[0])
        
                    # Apply action, get rewards and new state
                    reward = env.take_action(action)
                    game_over = env.game_over
                    input_t = env.get_state()
                    #with open("state.txt","ab") as f_handle:
                    #    np.savetxt(f_handle, input_t.reshape(window_h,window_w), fmt="%i")
                    if reward == 1:
                        win_cnt += 1
                    
                    gameDisplay.fill(white)
                    # Draw ball
                    pygame.draw.circle(gameDisplay, black, (env.ball_x*scale + scale//2, env.ball_y*scale + scale//2), scale//2)
                    # Draw paddle
                    pygame.draw.rect(gameDisplay, black, [(env.paddle_x-1)*scale, (env.window_h-1)*scale, 3*scale, scale])
                    pygame.display.update()
                    #clock.tick(60)
    #print("Win count {}".format(win_cnt))

    pygame.quit()
