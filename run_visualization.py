"""
Launch the Pygame visualization of the SAGIN ocean environment.
Run this locally (with a display) to see the simulation in real time.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from env.ocean_env import OceanEnv


def main():
    env = OceanEnv(render_mode="human", episode_length=5000)
    env.reset(seed=42)

    print("SAGIN Ocean IoT MEC Simulation")
    print("Close the Pygame window to exit.")

    running = True
    while running:
        obs, reward, terminated, truncated, info = env.step()
        if terminated:
            env.reset()
        try:
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        except Exception:
            pass

    env.close()


if __name__ == "__main__":
    main()
