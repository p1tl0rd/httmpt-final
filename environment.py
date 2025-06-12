# environment.py

import copy
import time
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from shapely.geometry import Point
import matplotlib.animation as animation
from shapely.geometry.polygon import Polygon
import pygame

class Swarm(object):
    """
    Môi trường mô phỏng một bầy UAV, học cách giữ đội hình và đi đến các waypoint
    trong khi tránh các vật cản.
    """
    def __init__(self, v_max=2, v_min=0, safe_distance=1, render_var=False):
        # Tải cấu hình từ file utils.py
        self.N, self.N_f, self.Weight_matrix, self.WP_list = Load_files()

        # Tham số mô phỏng
        self.wp_rad = 0.5
        self.counter = 0
        self.render_var = render_var
        self.v_max = v_max
        self.v_min = v_min
        self.max_steps = 1000
        self.wp_update_var = 0
        self.safe_distance = safe_distance  # Khoảng cách an toàn giữa các agent
        self.timestep = 0.1
        self.done = False

        # Tham số vật cản
        self.num_obstacles = 10
        self.obstacle_radius = 1.5
        self.obstacle_pos = []
        self.obstacle_safe_dist = self.obstacle_radius + 2.0 # tăng vùng an toàn

        # === Hằng số cho Hàm thưởng (Reward Constants) ===
        # Phần thưởng dương
        self.formation_reward_const = 0.1
        self.goal_reward_const = 50

        # Hình phạt (số âm)
        self.time_penalty_const = -0.1
        self.agent_collision_penalty = -200 # Phạt va chạm agent
        
        # Hằng số mới cho việc phạt né vật cản theo cách tinh chỉnh
        self.obstacle_invasion_weight = -50.0  # Trọng số phạt khi xâm phạm vùng an toàn
        self.obstacle_penalty = -200  # hoặc -300 nếu muốn phạt mạnh hơn

        # Tham số môi trường và Pygame
        self.const = 30
        self.boundary_points = [(self.const,self.const),(-self.const,self.const),(-self.const,-self.const),(self.const,-self.const)]
        self.start_location = np.array([[i, np.random.randint(3)] for i in range(self.N)]).astype('float64')
        
        self.pos = copy.copy(self.start_location)
        self.discard_list = []
        self.screen_size = 600
        self.pygame_initialized = False

        # Khởi tạo vật cản lần đầu
        self._generate_obstacles()

    def _generate_obstacles(self):
        """Tạo ngẫu nhiên vị trí các vật cản, đảm bảo chúng không quá gần điểm xuất phát/đích."""
        self.obstacle_pos.clear()
        start_buffer = 5.0
        goal_buffer = 5.0
        
        while len(self.obstacle_pos) < self.num_obstacles:
            pos_x = np.random.uniform(-self.const + self.obstacle_radius, self.const - self.obstacle_radius)
            pos_y = np.random.uniform(-self.const + self.obstacle_radius, self.const - self.obstacle_radius)
            new_pos = np.array([pos_x, pos_y])

            # Kiểm tra chồng chéo với điểm xuất phát, waypoint và các vật cản khác
            if any(self.get_distance(new_pos, p) < start_buffer for p in self.start_location): continue
            if any(self.get_distance(new_pos, p) < goal_buffer for p in self.WP_list): continue
            if any(self.get_distance(new_pos, p) < 2.5 * self.obstacle_radius for p in self.obstacle_pos): continue

            self.obstacle_pos.append(new_pos)

    def get_distance(self, point1, point2):
        """Tính khoảng cách Euclidean giữa hai điểm."""
        return np.linalg.norm(point1 - point2)

    def restore_start_location(self):
        """Đặt lại trạng thái môi trường về ban đầu cho một episode mới."""
        temp_var = 19
        self.WP_list = list(np.random.permutation([[-temp_var, temp_var], [-temp_var, -temp_var],
                                                   [temp_var, -temp_var], [temp_var, temp_var]]))
        self.pos = copy.copy(self.start_location)
        self.wp_update_var = 0
        self.discard_list.clear()
        self.counter = 0
        self._generate_obstacles()

    def reset(self):
        """Reset môi trường và trả về trạng thái quan sát ban đầu."""
        self.restore_start_location()
        goal_pos = self.get_current_waypoint()
        
        # Xây dựng vector trạng thái
        state_list = []
        state_list.extend(self.pos.flatten())
        state_list.extend(goal_pos.flatten())
        if self.obstacle_pos:
            state_list.extend(np.array(self.obstacle_pos).flatten())
        
        return np.array(state_list)
	
    def init_pygame(self):
        """Khởi tạo Pygame để render."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Swarm Simulation with Obstacles")
        self.clock = pygame.time.Clock()
        self.pygame_initialized = True

    def render(self, ep):
        """Vẽ trạng thái hiện tại của môi trường."""
        if not self.pygame_initialized: self.init_pygame()

        self.screen.fill((255, 255, 255))
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.screen_size, self.screen_size), 2)

        def to_screen_coords(pos, radius=0):
            scale = self.screen_size / (2 * self.const)
            x = int((pos[0] + self.const) * scale)
            y = int((self.const - pos[1]) * scale)
            r = int(radius * scale)
            return x, y, r

        # Vẽ vật cản và vùng penalty
        for obs_pos in self.obstacle_pos:
            # Vẽ vùng penalty (vùng an toàn) - vòng tròn lớn, nét đứt hoặc màu nhạt
            ox, oy, penalty_rad = to_screen_coords(obs_pos, self.obstacle_safe_dist)
            pygame.draw.circle(self.screen, (255, 200, 200), (ox, oy), penalty_rad, width=2)  # Màu đỏ nhạt, nét mảnh

            # Vẽ obstacle thật (vòng tròn nhỏ, màu đậm)
            ox, oy, orad = to_screen_coords(obs_pos, self.obstacle_radius)
            pygame.draw.circle(self.screen, (50, 50, 50), (ox, oy), orad)

        # Vẽ UAVs
        for pos in self.pos:
            x, y, _ = to_screen_coords(pos)
            pygame.draw.circle(self.screen, (3, 96, 22), (x, y), 8)

        # Vẽ waypoint
        wap = self.get_current_waypoint()
        wx, wy, _ = to_screen_coords(wap)
        pygame.draw.circle(self.screen, (255, 0, 0), (wx, wy), 10, 3)

        pygame.display.flip()
        self.clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def get_current_waypoint(self):
        """Lấy waypoint mục tiêu hiện tại."""
        return self.WP_list[self.wp_update_var]

    def update_pos(self, v):
        """Cập nhật vị trí của các agent dựa trên vector vận tốc."""
        self.pos += v * self.timestep

    def step(self, v):
        """Thực hiện một bước trong môi trường."""
        self.counter += 1
        goal_pos = self.get_current_waypoint()
        
        # Khởi tạo các biến để log reward
        reward_formation = 0
        reward_goal = 0
        penalty_collision_agent = 0
        penalty_collision_obstacle = 0
        penalty_time = 0
        
        if self.done:
            self.done = False
            self.restore_start_location()

        if self.counter >= self.max_steps:
            print("Max-steps reached")
            self.done = True

        v = np.reshape(v, (self.N, 2))
        v = rescale_vector(v, self.v_max, self.v_min)
        self.update_pos(v)
        
        collision_detected = False
        
        for i, pos1 in enumerate(self.pos):
            
            # --- 1. Agent-Agent Collision & Formation Reward ---
            for j, pos2 in enumerate(self.pos):
                if i == j: continue
                dist = self.get_distance(pos1, pos2)
                
                if dist < self.safe_distance:
                    penalty_collision_agent += self.agent_collision_penalty
                    self.done = True
                    collision_detected = True
                    break

                if abs(dist - self.Weight_matrix[i][j]) <= 0.2:
                    reward_formation += self.formation_reward_const

            if collision_detected: break

            # --- 2. Obstacle Collision Penalty (LOGIC ĐÃ ĐƯỢC CẬP NHẬT) ---
            for obs_pos in self.obstacle_pos:
                dist_to_surface = self.get_distance(pos1, obs_pos) - self.obstacle_radius
                
                if dist_to_surface < 0: # Va chạm cứng
                    penalty_collision_obstacle += self.obstacle_collision_penalty
                    self.done = True
                    collision_detected = True
                    break
                
                safe_zone_width = self.obstacle_safe_dist - self.obstacle_radius
                if dist_to_surface < safe_zone_width: # Xâm phạm vùng an toàn
                    invasion_ratio = 1.0 - (dist_to_surface / safe_zone_width)
                    penalty_collision_obstacle += self.obstacle_invasion_weight * (invasion_ratio ** 2)

            if collision_detected: break

        # --- 3. Goal & Time Reward (Chỉ tính khi không có va chạm) ---
        if not self.done:
            for i, pos1 in enumerate(self.pos):
                goal_distance = self.get_distance(pos1, goal_pos)
                if abs(goal_distance - self.Weight_matrix[i][self.N]) <= self.wp_rad and i not in self.discard_list:
                    self.discard_list.append(i)
                    reward_goal += self.goal_reward_const
                else:
                    penalty_time += self.time_penalty_const

        # --- 4. Kiểm tra hoàn thành mục tiêu ---
        if len(self.discard_list) == self.N:
            print("GOAL REACHED", end=" ")
            self.discard_list.clear()
            self.wp_update_var += 1
            if self.wp_update_var >= len(self.WP_list):
                self.done = True
                print("FINAL GOAL", end=" ")
                self.wp_update_var -= 1

        # Tổng hợp reward
        total_reward = (
            reward_formation
            + reward_goal
            + penalty_collision_agent
            + penalty_collision_obstacle
            + penalty_time
        )

        info = {
            'reward_formation': reward_formation,
            'reward_goal': reward_goal,
            'penalty_agent_collision': penalty_collision_agent,
            'penalty_obstacle_collision': penalty_collision_obstacle,
            'penalty_time': penalty_time
        }
        
        # Xây dựng lại state để trả về
        state_list = []
        state_list.extend(self.pos.flatten())
        state_list.extend(goal_pos.flatten())
        if self.obstacle_pos:
            state_list.extend(np.array(self.obstacle_pos).flatten())

        return np.array(state_list), total_reward, self.done, info