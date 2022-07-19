import gym
from gym import spaces
import numpy as np
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
import time
from distance import Distance

class DinoEnv(gym.Env):
    def __init__(self):

        self.action_space = spaces.Discrete(3)

        # Velocity, Distance Obstacle; Height Obstacle; Width Obstacle; Coord Y obstacle
        low_observation = np.array([
            5,
            0,
            0,
            0,
            0
        ], dtype=np.float32, )
        high_observation = np.array([
            15,
            10,
            2,
            2,
            150
        ], dtype=np.float32, )

        self.observation_space = spaces.Box(low=low_observation, high=high_observation, dtype=np.float32)

        _chrome_options = webdriver.ChromeOptions()
        _chrome_options.add_argument("--mute-audio")
        _chrome_options.add_argument("--start-maximized")

        self._driver = webdriver.Chrome(
            executable_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "chromedriver"
            ),
            options=_chrome_options
        )

        self.action_chains = ActionChains(self._driver)
        self.actions_map = [
            Keys.ARROW_RIGHT,  # do nothing
            Keys.ARROW_UP,  # jump
            Keys.ARROW_DOWN  # duck
        ]

    def reset(self):
        try:
            if self._driver.current_url == 'chrome://dino/':
                self._driver.refresh()
            else:
                self._driver.get('chrome://dino')
        except WebDriverException:
            pass
        time.sleep(0.5)
        self._driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.SPACE)
        time.sleep(0.5)
        return self.next_observation()

    def step(self, action):
        self.action_chains.key_down(self.actions_map[action]).perform()
        self.action_chains.key_up(self.actions_map[action]).perform()
        # self._driver.find_element(By.TAG_NAME, 'body') \
        #     .send_keys(self.actions_map[action])

        obs = self.next_observation()
        done = self.get_done()
        reward = .1 if not done else -1
        # time.sleep(0.015)

        return obs, reward, done

    def next_observation(self):
        velocity = self.get_velocity()
        min_dist_obstacle, height_obstacle, width_obstacle, y_coord = self.get_values_in_image()
        return velocity, min_dist_obstacle, height_obstacle, width_obstacle, y_coord

    def get_velocity(self):
        return float(self._driver.execute_script("return Runner.instance_.currentSpeed"))

    def get_values_in_image(self):
        _img = self._driver.execute_script(
            "return document.querySelector('canvas.runner-canvas').toDataURL()"
        )
        min_dist_obstacle, height_obstacle, width_obstacle, y_coord = Distance.get_distance_image(_img)
        return min_dist_obstacle, height_obstacle, width_obstacle, y_coord

    def get_done(self):
        return not self._driver.execute_script("return Runner.instance_.playing")

    def get_score(self):
        return int(''.join(
            self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        ))

