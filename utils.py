import numpy as np
import pickle
from scipy.spatial import distance
from formation_factory import get_formation_coords # Import hàm factory

def generate_weight_matrix(coords, leader_id):
    """
    Tạo ma trận khoảng cách từ tọa độ của đội hình.
    Ma trận này bao gồm khoảng cách giữa các agent và khoảng cách từ mỗi agent đến leader.
    
    Args:
        coords (numpy.ndarray): Tọa độ của các agent trong đội hình.
        leader_id (int): ID của agent dẫn đầu.
        
    Returns:
        numpy.ndarray: Ma trận trọng số (N+1, N+1).
    """
    num_agents = coords.shape[0]
    
    # Tính ma trận khoảng cách Euclidean giữa tất cả các agent
    dist_matrix = distance.cdist(coords, coords, 'euclidean')
    
    # Tạo ma trận trọng số lớn hơn (N+1, N+1)
    # Cột/hàng cuối cùng sẽ lưu khoảng cách đến mục tiêu (goal)
    # Trong mô hình này, khoảng cách đến mục tiêu được xem là khoảng cách đến leader
    weight_matrix = np.zeros((num_agents + 1, num_agents + 1))
    
    # Điền khoảng cách giữa các agent
    weight_matrix[:num_agents, :num_agents] = dist_matrix
    
    # Điền khoảng cách từ mỗi agent đến leader (đây sẽ là khoảng cách mục tiêu tương đối)
    leader_distances = dist_matrix[:, leader_id]
    weight_matrix[:num_agents, num_agents] = leader_distances
    weight_matrix[num_agents, :num_agents] = leader_distances
    
    return weight_matrix


def Load_files():
    """
    Hàm này giờ sẽ định cấu hình các tham số và tạo ma trận trọng số trực tiếp.
    """
    # ==================== CONFIGURATION ====================
    # Thay đổi các tham số ở đây
    NUM_AGENTS = 5                 # Tổng số UAV
    LEADER_ID = 2                  # UAV dẫn đầu (ID bắt đầu từ 0)
    SPACING = 1.0                  # Khoảng cách cơ bản giữa các UAV
    FORMATION_TYPE = 'v'           # Chọn đội hình: 'v', 'line', 'diamond'
    
    # Danh sách các điểm tham chiếu (waypoints)
    WP_list = [[-20, 20], [20, 20], [20, -20], [-20, -20]]
    # =======================================================
    
    print(f"--- Configuration ---")
    print(f"Number of Agents: {NUM_AGENTS}")
    print(f"Formation: {FORMATION_TYPE.upper()}")
    print(f"Spacing: {SPACING}")
    print(f"Leader ID: {LEADER_ID}")
    print(f"---------------------")

    # 1. Lấy tọa độ tương đối của đội hình mong muốn
    formation_coords = get_formation_coords(FORMATION_TYPE, NUM_AGENTS, LEADER_ID, SPACING)
    
    # 2. Tạo ma trận trọng số (ma trận khoảng cách mong muốn)
    # Lưu ý: ma trận này không nhân với D_wm nữa vì spacing đã bao gồm khoảng cách
    Weight_matrix = generate_weight_matrix(formation_coords, LEADER_ID)
    
    # Chuyển đổi N_f (uav dẫn đầu) từ 1-based sang 0-based
    N_f_zero_based = LEADER_ID
    
    return NUM_AGENTS, N_f_zero_based, Weight_matrix, WP_list


def rescale_vector(v, v_max, v_min):
    # (Hàm này không thay đổi)
	v_mod = np.linalg.norm(v)
	try:
		v = (v/v_mod)*min(max(v_mod,v_min),v_max)
	except:
		return v
	return v