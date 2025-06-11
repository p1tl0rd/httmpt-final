import numpy as np

def create_v_formation(num_agents, leader_id, spacing):
    """
    Tạo ma trận tọa độ cho đội hình chữ V.
    
    Args:
        num_agents (int): Tổng số agent.
        leader_id (int): ID của agent dẫn đầu (thường là 0).
        spacing (float): Khoảng cách cơ bản giữa các agent.
        
    Returns:
        numpy.ndarray: Mảng tọa độ (num_agents, 2).
    """
    if num_agents < 1:
        return np.array([])
        
    coords = np.zeros((num_agents, 2))
    
    # Vị trí của leader
    coords[leader_id] = [0, 0]
    
    left_wing_count = 0
    right_wing_count = 0
    
    for i in range(num_agents):
        if i == leader_id:
            continue
        
        if (i - leader_id) % 2 == 1:
            # Cánh phải
            right_wing_count += 1
            coords[i] = [right_wing_count * spacing, -right_wing_count * spacing]
        else:
            # Cánh trái
            left_wing_count += 1
            coords[i] = [-left_wing_count * spacing, -left_wing_count * spacing]
            
    return coords

def create_line_formation(num_agents, spacing):
    """
    Tạo ma trận tọa độ cho đội hình hàng ngang.
    """
    coords = np.zeros((num_agents, 2))
    center_offset = (num_agents - 1) * spacing / 2.0
    for i in range(num_agents):
        coords[i] = [i * spacing - center_offset, 0]
    return coords

def create_diamond_formation(num_agents, spacing):
    """
    Tạo ma trận tọa độ cho đội hình kim cương (chỉ hoạt động tốt với số agent nhất định).
    Ví dụ cho 4 agent.
    """
    if num_agents != 4:
        print("Warning: Diamond formation is optimized for 4 agents.")
        # Fallback to line formation if not 4 agents
        return create_line_formation(num_agents, spacing)

    coords = np.array([
        [0, spacing],    # Top
        [-spacing, 0],   # Left
        [spacing, 0],    # Right
        [0, -spacing]    # Bottom
    ])
    return coords

def get_formation_coords(formation_name, num_agents, leader_id, spacing):
    """
    Hàm factory để lấy tọa độ đội hình dựa trên tên.
    """
    if formation_name == 'v':
        return create_v_formation(num_agents, leader_id, spacing)
    elif formation_name == 'line':
        return create_line_formation(num_agents, spacing)
    elif formation_name == 'diamond':
        return create_diamond_formation(num_agents, spacing)
    else:
        raise ValueError(f"Unknown formation: {formation_name}. Available: 'v', 'line', 'diamond'")