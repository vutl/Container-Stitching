import cv2
import numpy as np
from itertools import combinations
from collections import defaultdict

CONST_EXPAND_IMAGE = 150
HEIGHT_LIMIT = 70
# def glue_matching(left_kpts, right_kpts, matches):
#     # Chuyển đổi matches thành danh sách điểm bình thường để dễ xử lý
#     matches_points = [
#         (left_kpts[m.queryIdx].pt, right_kpts[m.trainIdx].pt) for m in matches]
#     return matches_points


def glue_matching(left_kpts, right_kpts, matches, imgshape1, imgshape2, cor1=None, cor2=None):
    # Chuyển đổi matches thành danh sách điểm bình thường để dễ xử lý
    # matches_points = [
    #     (left_kpts[m.queryIdx].pt, right_kpts[m.trainIdx].pt) for m in matches]

    matches_points = []
    # print('len original matches', len(matches))
    for m in matches:
        left_point = left_kpts[m.queryIdx].pt
        right_point = right_kpts[m.trainIdx].pt

        if left_point[1] < HEIGHT_LIMIT or right_point[1] < HEIGHT_LIMIT\
            or left_point[1] > imgshape1[0] - HEIGHT_LIMIT \
                or right_point[1] > imgshape2[0] - HEIGHT_LIMIT\
            or abs(left_point[1] - right_point[1]) > 10:
            continue

        matches_points.append((left_point, right_point))

    # print('len matches_points', len(matches_points))

    return matches_points


def cluster_keypoints_kmeans(matches, image_height=None):
    """
    Phân cụm các keypoints thành 3 cụm dựa trên tọa độ y (độ cao):
    - Cụm 0: 1/3 trên của ảnh
    - Cụm 1: 1/3 giữa của ảnh
    - Cụm 2: 1/3 dưới của ảnh

    Tham số:
    - matches: danh sách các cặp keypoints, mỗi cặp là ((x1, y1), (x2, y2))
    - image_height: chiều cao của ảnh. Nếu không cung cấp, sẽ sử dụng giá trị y lớn nhất từ các keypoints.

    Trả về:
    - clustered_points: danh sách các tuple ((x1, y1), (x2, y2), cluster_label)
    """
    if len(matches) == 0:
        return []
    coords = np.array([[(kp1[0] + kp2[0]) / 2, (kp1[1] + kp2[1]) / 2]
                      for kp1, kp2 in matches])

    if image_height is None:
        image_height = np.max(coords[:, 1])

    # Tính các ngưỡng phân chia thành 3 phần bằng nhau
    four = (image_height - CONST_EXPAND_IMAGE) / 3

    # Gán nhãn cụm dựa trên y
    labels = np.zeros(len(coords), dtype=int)
    labels[coords[:, 1] < four] = 0
    labels[(coords[:, 1] >= four) & (coords[:, 1] < 2 * four)] = 1
    labels[(coords[:, 1] >= 2*four) & (coords[:, 1] < 3 * four)] = 2
    labels[coords[:, 1] >= (image_height - CONST_EXPAND_IMAGE)] = 3
    #
    clustered_points = [(matches[i][0], matches[i][1], labels[i])
                        for i in range(len(matches))]

    return clustered_points


def determine_best_cluster(clustered_points):
    """
    Xác định cụm tốt nhất dựa trên điều kiện cụm 3 và độ tự tin của các cụm khác.

    Tham số:
    - clustered_points: danh sách các tuple ((x1, y1), (x2, y2), cluster_label)

    Trả về:
    - best_cluster_label: nhãn của cụm tốt nhất
    """
    # Tách các điểm theo cụm
    cluster_4_points = [p for p in clustered_points if p[2] == 3]

    # **Điều kiện bổ sung: Nếu tổng số điểm < 15 và cụm 3 có ít nhất 1 điểm, chọn cụm 3**
    cluster_counts = defaultdict(int)
    for _, _, label in clustered_points:
        cluster_counts[label] += 1

    # Đếm số điểm trong cụm 0-2
    clusters_0_3 = [count for label, count in cluster_counts.items() if label in [
        0, 1, 2]]

    # Kiểm tra điều kiện: tất cả các cụm 0-2 đều có số điểm dưới 5
    all_clusters_0_3_under_5 = all(count < 5 for count in clusters_0_3)

    # Kiểm tra số điểm trong cụm 3
    cluster_4_count = cluster_counts.get(3, 0)

    # Điều kiện: tất cả cụm 0-3 < 5 và cụm 3 >=1
    if all_clusters_0_3_under_5 and cluster_4_count >= 1:
        return 3

    # **Điều kiện ban đầu: Cụm 3 có từ 10 điểm trở lên và độ lệch x <= 10 pixel**
    if len(cluster_4_points) >= 10:
        # Lấy các tọa độ x trung bình của các điểm trong cụm 3
        x_coords = [(p[0][0] + p[1][0]) / 2 for p in cluster_4_points]
        max_x = max(x_coords)
        min_x = min(x_coords)
        if (max_x - min_x) <= 10:
            # Cụm 3 thoả mãn cả hai điều kiện
            return 3

    # Nếu không thoả mãn các điều kiện trên, sử dụng calculate_confidence để xác định cụm tốt nhất
    best_cluster = calculate_confidence(clustered_points)
    return best_cluster


def calculate_confidence(clustered_points):
    """
    Tính độ tự tin cho từng cụm dựa trên khoảng cách trung bình về tọa độ x,
    toạ độ y, số lượng điểm trong cụm, và loại bỏ cụm có độ lệch x khác biệt.

    Tham số:
    - clustered_points: danh sách các tuple ((x1, y1), (x2, y2), cluster_label)

    Trả về:
    - best_cluster_label: nhãn của cụm tốt nhất
    """
    clusters = {}

    # Gom các điểm theo nhãn cụm
    for point1, point2, label in clustered_points:
        if label not in clusters:
            clusters[label] = []
        clusters[label].append([point1, point2])

    # Tính độ tự tin dựa trên tổng khoảng cách x và y trung bình
    cluster_confidences = []
    for label, points in clusters.items():
        total_x_distances = sum([abs(p1[0] - p2[0]) for p1, p2 in points])
        total_y_distances = sum([abs(p1[1] - p2[1]) for p1, p2 in points])
        avg_x_distance = total_x_distances / len(points)
        avg_y_distance = total_y_distances / len(points)
        num_points = len(points)
        cluster_confidences.append(
            (label, avg_x_distance, avg_y_distance, num_points))

    # Kiểm tra độ lệch x trung bình giữa các cụm
    if len(cluster_confidences) >= 3:
        # Tìm hai cụm có độ lệch x gần nhau nhất
        min_diff = float('inf')
        clusters_to_keep = []
        for c1, c2 in combinations(cluster_confidences, 2):
            diff = abs(c1[1] - c2[1])
            if diff < min_diff:
                min_diff = diff
                clusters_to_keep = [c1, c2]

        # Tìm cụm bị loại
        all_labels = {c[0] for c in cluster_confidences}
        kept_labels = {c[0] for c in clusters_to_keep}
        removed_labels = all_labels - kept_labels

        # Nếu cụm bị loại có độ lệch x khác biệt nhiều so với hai cụm còn lại
        for label in removed_labels:
            cluster = next(c for c in cluster_confidences if c[0] == label)
            if abs(cluster[1] - clusters_to_keep[0][1]) > min_diff * 1.2:
                # Loại cụm này khỏi danh sách
                cluster_confidences = clusters_to_keep
                break

    # Sắp xếp các cụm theo độ tự tin (ưu tiên cụm có độ lệch x, y nhỏ nhất và số điểm nhiều nhất)
    cluster_confidences.sort(key=lambda x: (-x[3], x[1], x[2]))
    if not len(cluster_confidences):
        return None
    best_cluster_label = cluster_confidences[0][0]
    return best_cluster_label


def get_point_near_cluster_avg(clustered_points, best_cluster_label):
    """
    Lấy điểm trong cụm có tọa độ gần với giá trị trung bình của cụm nhất,
    chỉ lấy các điểm có giá trị x nhỏ hơn trung bình x của cụm.

    Tham số:
    - clustered_points: danh sách các tuple ((x1, y1), (x2, y2), cluster_label)
    - best_cluster_label: nhãn của cụm tốt nhất

    Trả về:
    - (point1, point2): cặp điểm gần với trung bình cụm nhất
    """
    points_in_cluster = [
        p for p in clustered_points if p[2] == best_cluster_label]

    if len(points_in_cluster) == 0:
        return None  # Hoặc xử lý phù hợp

    # Tính tọa độ x và y trung bình của cụm
    avg_x = sum([(p1[0] + p2[0]) / 2 for p1, p2,
                _ in points_in_cluster]) / len(points_in_cluster)
    avg_y = sum([(p1[1] + p2[1]) / 2 for p1, p2,
                _ in points_in_cluster]) / len(points_in_cluster)

    # Lọc các điểm có giá trị x nhỏ hơn trung bình x
    points_below_avg_x = [
        p for p in points_in_cluster
        if ((p[0][0] + p[1][0]) / 2) <= avg_x
    ]

    if len(points_below_avg_x) == 0:
        return None  # Không có điểm nào thỏa mãn điều kiện x < avg_x

    # Tìm điểm có tọa độ gần với giá trị trung bình nhất (sử dụng khoảng cách Euclid)
    closest_point = min(
        points_below_avg_x,
        key=lambda p: np.sqrt(
            (((p[0][0] + p[1][0]) / 2 - avg_x) ** 2) +
            (((p[0][1] + p[1][1]) / 2 - avg_y) ** 2)
        )
    )
    return closest_point[0], closest_point[1]


def sift_feature_matching(img1, img2, confidence_threshold):
    # Khởi tạo đối tượng SIFT
    sift = cv2.SIFT_create()

    # Phát hiện keypoints và tính toán descriptors cho cả hai ảnh
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Sử dụng BFMatcher (Brute-Force Matcher) để ghép nối các đặc trưng
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Áp dụng tỷ lệ Lowe để lọc các matches chất lượng
    good_matches = []
    for m, n in matches:
        if m.distance < confidence_threshold * n.distance:
            good_matches.append(m)

    # Lấy tọa độ của các cặp điểm tương đồng
    left_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    right_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    return left_pts, right_pts
