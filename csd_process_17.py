from collections import defaultdict
import cv2, numpy as np
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """두 박스 간의 IoU(Intersection over Union) 계산"""
    x1, y1, x2, y2 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 교집합 영역 계산
    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    
    # 각 박스 면적
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 합집합 면적
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def is_contained(box1, box2):
    """box1이 box2 안에 완전히 포함되는지 확인"""
    x1, y1, x2, y2 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    return x1 >= x1_2 and y1 >= y1_2 and x2 <= x2_2 and y2 <= y2_2

def count_nearby_slow_vehicles(tid, all_vehicle_positions, current_frame_idx, radius=150):
    """주변 반경 내에서 느리게 움직이거나 정지한 차량 수를 카운트"""
    if tid not in all_vehicle_positions or len(all_vehicle_positions[tid]) == 0:
        return 0, 0  # (느린 차량 수, 정상 차량 수)

    current_pos = all_vehicle_positions[tid][-1]
    nearby_slow = 0
    nearby_normal = 0

    for other_tid, positions in all_vehicle_positions.items():
        if other_tid == tid or len(positions) < 15:
            continue

        # 최근 프레임만 비교 (같은 시간대)
        other_pos = positions[-1]
        if abs(other_pos[2] - current_frame_idx) > 5:  # 프레임 차이가 크면 스킵
            continue

        # 거리 계산
        distance = ((current_pos[0] - other_pos[0])**2 +
                   (current_pos[1] - other_pos[1])**2)**0.5

        if distance < radius:
            # 최근 15프레임 동안의 이동거리 확인
            if len(positions) >= 15:
                recent_positions = positions[-15:]
                first_pos = recent_positions[0]
                last_pos = recent_positions[-1]
                movement = ((last_pos[0] - first_pos[0])**2 +
                           (last_pos[1] - first_pos[1])**2)**0.5

                if movement < 20:  # 20픽셀 이하 = 느린 움직임/정지
                    nearby_slow += 1
                elif movement > 40:  # 40픽셀 이상 = 정상 주행
                    nearby_normal += 1

    return nearby_slow, nearby_normal

def analyze_movement_pattern(positions, window=90):
    """움직임 패턴 분석 - 정체는 간헐적 전진, 정지는 완전 고정"""
    if len(positions) < 20:  # 최소 20프레임은 필요
        return "unknown"

    # 가능한 최대 윈도우 사용 (최대 90프레임 = 3초)
    actual_window = min(window, len(positions))
    recent = positions[-actual_window:]

    # 전체 이동거리 계산
    total_movement = ((recent[-1][0] - recent[0][0])**2 +
                     (recent[-1][1] - recent[0][1])**2)**0.5

    # 프레임당 평균 이동거리
    avg_movement_per_frame = total_movement / len(recent)

    # 구간별 움직임 분석 (10프레임씩 더 긴 구간으로)
    movements = []
    for i in range(0, len(recent) - 10, 10):
        segment_movement = ((recent[i+10][0] - recent[i][0])**2 +
                           (recent[i+10][1] - recent[i][1])**2)**0.5
        movements.append(segment_movement)

    if len(movements) == 0:
        return "unknown"

    # 움직임 분류
    moving_segments = sum(1 for m in movements if m > 8)  # 8픽셀 이상 움직임
    stationary_segments = sum(1 for m in movements if m < 3)  # 거의 정지

    # 움직임의 변화 분석 (정체는 변화가 크고, 정지차는 변화 없음)
    movement_variance = 0
    if len(movements) > 1:
        avg_movement = sum(movements) / len(movements)
        movement_variance = sum((m - avg_movement) ** 2 for m in movements) / len(movements)

    # 정지차 판단 (엄격하게):
    # 1. 전체 이동거리가 매우 작고 (10픽셀 이하)
    # 2. 정지 구간이 대부분이며 (85% 이상)
    # 3. 움직임 변화가 거의 없음 (변동성 < 5)
    if (total_movement <= 10 and
        stationary_segments / len(movements) >= 0.85 and
        movement_variance < 5):
        return "parked"  # 확실한 정지

    # 정체 판단:
    # 1. 전체 이동거리가 어느정도 있거나 (20픽셀 이상)
    # 2. 움직인 구간이 충분히 있거나 (25% 이상)
    # 3. 움직임 변화가 큼 (변동성 >= 5, 멈췄다 가고 반복)
    if (total_movement >= 20 or
        moving_segments / len(movements) >= 0.25 or
        movement_variance >= 5):
        return "congestion"  # 간헐적 전진 = 정체

    # 애매한 경우 (10~20픽셀, 낮은 변동성) = 정지로 판단
    return "parked"

def nms_by_confidence(boxes_xyxy, tids, confs, clss, iou_threshold=0.5):
    """신뢰도 기반 NMS (Non-Maximum Suppression)"""
    if len(boxes_xyxy) == 0:
        return [], [], [], []
    
    # 신뢰도 기준으로 내림차순 정렬
    indices = np.argsort(confs)[::-1]
    
    keep_indices = []
    
    for i in indices:
        keep = True
        for j in keep_indices:
            iou = calculate_iou(boxes_xyxy[i], boxes_xyxy[j])
            
            # 겹침이 심한 경우만 제거 (포함 관계는 별도 처리 가능)
            if iou > iou_threshold:
                # 클래스가 다른 경우는 겹쳐도 유지 (사람과 차량)
                if clss[i] != clss[j]:
                    continue
                
                # 포함 관계인지 확인
                if is_contained(boxes_xyxy[i], boxes_xyxy[j]) or is_contained(boxes_xyxy[j], boxes_xyxy[i]):
                    # 포함 관계라면 더 신뢰도 높은 것(이미 keep_indices에 있는 것) 유지
                    keep = False
                else:
                    # 단순 겹침이라면 제거
                    keep = False
                break
        
        if keep:
            keep_indices.append(i)
    
    return (boxes_xyxy[keep_indices], 
            [tids[i] for i in keep_indices], 
            confs[keep_indices],
            [clss[i] for i in keep_indices])

# ── 설정 ────────────────────────────────────────────────
model       = YOLO("rtdetr-l.pt")                       # 모델
video_path  = f"input\CCTV042_20250702_172343209_1_1_정지차_3차로.mp4"
cap         = cv2.VideoCapture(video_path)
track_hist  = defaultdict(list)                        # {id: [(x, y), ...]}
max_hist    = 60                                       # 궤적 길이
# 사람, 자동차, 트럭, 버스 검지 (COCO dataset class IDs)
target_classes = [0, 2, 5, 7]  # person, car, bus, truck
vehicle_classes = [2, 5, 7]    # car, bus, truck

# 차량 위치 히스토리 관리
vehicle_positions = defaultdict(list)  # {id: [(center_x, center_y, frame_idx), ...]}
vehicle_wrong_way_history = defaultdict(list)  # {id: [True/False, ...]} 역주행 판단 히스토리
vehicle_stopped_frames = defaultdict(int)  # {id: 연속 정지 프레임 수}

# 정지 감지 설정 (새로운 패러다임)
QUICK_STOP_FRAMES = 30     # 30프레임(약 1초) - 빠른 정지 상태 판단
CONFIRMED_STOP_FRAMES = 150  # 150프레임(약 5초) - 확실한 정지차로 판단
MIN_BOX_SIZE = 50          # 정지차 판단에 사용할 최소 박스 크기 (가로+세로 평균)
STOP_THRESHOLD_MIN = 12    # 정지 판단 최소 임계값 (픽셀)
STOP_THRESHOLD_RATIO = 0.06  # 박스 크기 대비 정지 임계값 비율

# 역주행 감지 설정
wrong_way_frame_count = 20  # 20프레임(약 0.67초) 이상 관찰하여 역주행 판단
wrong_way_threshold = 10  # Y좌표 변화량 임계값 (픽셀)

# 박스 색상
moving_color = (0, 255, 0)       # 움직이는 차량: 초록색
congestion_color = (0, 165, 255) # 정체 차량: 주황색
parked_color = (0, 0, 255)       # 정지 차량: 빨간색
wrong_way_color = (255, 0, 255)  # 역주행 차량: 마젠타색
person_color = (255, 0, 0)       # 사람: 파란색

# 비디오 출력 설정
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

frame_idx = 0
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1

    # ── 추적 ────────────────────────────────────────────
    res = model.track(
        frame, 
        persist=True, 
        classes=target_classes,
        device='cuda',
        tracker='my_bot.yaml',
        iou=0.8,
        conf=0.15
    )[0]

    if res.boxes and res.boxes.is_track:
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()      # (N,4) [x1,y1,x2,y2]
        tids       = res.boxes.id.int().cpu().tolist() # 트랙 ID들
        confs      = res.boxes.conf.cpu().numpy()      # 신뢰도 점수들
        clss       = res.boxes.cls.int().cpu().tolist()  # 클래스 ID들
        
        # 신뢰도 0.5 이상인 박스만 필터링
        valid_indices = confs >= 0.5
        boxes_xyxy = boxes_xyxy[valid_indices]
        tids = [tids[i] for i in range(len(tids)) if valid_indices[i]]
        clss = [clss[i] for i in range(len(clss)) if valid_indices[i]]
        confs = confs[valid_indices]
        
        # NMS 후처리로 겹치는 박스 제거 (신뢰도가 높은 것만 유지)
        boxes_xyxy, tids, confs, clss = nms_by_confidence(boxes_xyxy, tids, confs, clss, iou_threshold=0.5)

        # 1단계: 각 차량의 정지 상태 빠르게 판단
        stopped_vehicles = []  # 이번 프레임에 정지 중인 차량들

        for (x1, y1, x2, y2), tid, conf, cls in zip(boxes_xyxy, tids, confs, clss):
            if cls in vehicle_classes:
                # 바운딩 박스 하단 중심점 계산
                center_x = (x1 + x2) / 2
                center_y = y2

                # 차량 위치 히스토리 업데이트
                vehicle_positions[tid].append((center_x, center_y, frame_idx))

                # 오래된 히스토리 제거 (메모리 절약)
                if len(vehicle_positions[tid]) > max_hist:
                    vehicle_positions[tid] = vehicle_positions[tid][-max_hist:]

                # 박스 크기 계산
                box_width = x2 - x1
                box_height = y2 - y1
                box_size = (box_width + box_height) / 2

                # 빠른 정지 상태 판단 (30프레임만 체크)
                is_stopped_now = False
                if (len(vehicle_positions[tid]) >= QUICK_STOP_FRAMES and
                    box_size >= MIN_BOX_SIZE):

                    # 박스 크기에 비례한 정지 임계값
                    adaptive_threshold = max(STOP_THRESHOLD_MIN, box_size * STOP_THRESHOLD_RATIO)

                    # 최근 30프레임의 위치 변화 확인
                    recent_positions = vehicle_positions[tid][-QUICK_STOP_FRAMES:]
                    first_pos = recent_positions[0]
                    max_distance = 0

                    for pos in recent_positions[1:]:
                        distance = ((pos[0] - first_pos[0])**2 + (pos[1] - first_pos[1])**2)**0.5
                        max_distance = max(max_distance, distance)

                    # 정지 상태 판단
                    if max_distance < adaptive_threshold:
                        is_stopped_now = True
                        vehicle_stopped_frames[tid] += 1
                        stopped_vehicles.append((tid, x1, y1, x2, y2, cls))
                    else:
                        vehicle_stopped_frames[tid] = 0  # 움직이면 카운터 리셋
                else:
                    vehicle_stopped_frames[tid] = 0  # 관찰 시간 부족하면 리셋

        # 2단계: 프레임 단위 정체 판단 (정지 차량이 2대 이상이면 모두 정체)
        num_stopped = len(stopped_vehicles)

        # 3단계: 각 정지 차량에 대해 STOP vs TRAFFIC 판단 및 그리기
        for tid, x1, y1, x2, y2, cls in stopped_vehicles:
            is_parked = False
            is_congestion = False
            show_as_normal = False  # 초록색(일반)으로 표시

            # 패러다임: 정지 차량이 2대 이상이면 무조건 정체
            if num_stopped >= 2:
                is_congestion = True
                # 정체 상황에서는 정지차 확정 카운터 초기화
                vehicle_stopped_frames[tid] = 0
            # 1대만 정지 상태인 경우
            else:
                # 5초(150프레임) 이상 지속 정지 = 정지차
                if vehicle_stopped_frames[tid] >= CONFIRMED_STOP_FRAMES:
                    is_parked = True
                else:
                    # 5초 미만 = 일반 차량처럼 초록색으로 표시
                    show_as_normal = True

            # 색상 결정
            if is_parked:
                box_color = parked_color
                status_text = " (STOP)"
            elif is_congestion:
                box_color = congestion_color
                status_text = " (TRAFFIC)"
            else:
                # show_as_normal = True인 경우 포함
                box_color = moving_color
                status_text = ""

            # 바운딩 박스 그리기
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                box_color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # ID 텍스트 표시
            text = f"ID:{tid}{status_text}"
            cv2.putText(
                frame,
                text,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
                cv2.LINE_AA
            )

        # 4단계: 정지하지 않은 차량들 처리 (역주행 감지 등)
        for (x1, y1, x2, y2), tid, conf, cls in zip(boxes_xyxy, tids, confs, clss):
            if cls in vehicle_classes:
                # 이미 정지 차량으로 처리했으면 스킵
                if any(t == tid for t, _, _, _, _, _ in stopped_vehicles):
                    continue

                # 역주행 감지
                is_wrong_way = False
                box_width = x2 - x1
                box_height = y2 - y1
                box_size = (box_width + box_height) / 2

                if (len(vehicle_positions[tid]) >= wrong_way_frame_count and
                    box_size >= MIN_BOX_SIZE):

                    # 충분한 프레임 수 확보 후 역주행 판단
                    recent_positions = vehicle_positions[tid][-wrong_way_frame_count:]

                    # 시작점과 끝점의 좌표 차이로 전체적인 방향 판단
                    start_x = recent_positions[0][0]
                    start_y = recent_positions[0][1]
                    end_x = recent_positions[-1][0]
                    end_y = recent_positions[-1][1]

                    x_movement = end_x - start_x  # 음수: 왼쪽 이동, 양수: 오른쪽 이동
                    y_movement = end_y - start_y  # 양수: 아래로 이동, 음수: 위로 이동

                    # 추가적으로 중간 구간들도 확인하여 일관된 방향인지 검증
                    consistent_wrong_way = 0
                    total_segments = 0

                    # 5프레임씩 나누어 각 구간의 방향성 확인 (더 세밀하게)
                    segment_size = 5
                    for i in range(0, len(recent_positions) - segment_size, segment_size):
                        seg_start_x = recent_positions[i][0]
                        seg_start_y = recent_positions[i][1]
                        seg_end_x = recent_positions[i + segment_size][0]
                        seg_end_y = recent_positions[i + segment_size][1]

                        seg_x_movement = seg_end_x - seg_start_x
                        seg_y_movement = seg_end_y - seg_start_y

                        # 최소 움직임이 있을 때만 판단
                        movement_magnitude = (seg_x_movement**2 + seg_y_movement**2)**0.5
                        if movement_magnitude > 3:
                            total_segments += 1
                            # 역주행 조건: 아래쪽(y증가) 또는 왼쪽(x감소)으로 이동
                            if seg_y_movement > 2 or seg_x_movement < -2:
                                consistent_wrong_way += 1

                    # 전체 움직임 기준으로도 역주행 판단
                    total_movement = (x_movement**2 + y_movement**2)**0.5
                    is_moving_down = y_movement > wrong_way_threshold
                    is_moving_left = x_movement < -wrong_way_threshold

                    # 역주행 조건:
                    # 1) 충분한 움직임이 있고 (아래쪽 또는 왼쪽으로)
                    # 2) 구간별 일관성도 확인
                    if ((is_moving_down or is_moving_left) and
                        total_movement > wrong_way_threshold and
                        total_segments > 0 and
                        consistent_wrong_way / total_segments >= 0.3):
                        is_wrong_way = True

                # 역주행 판단 히스토리 업데이트 및 안정화
                vehicle_wrong_way_history[tid].append(is_wrong_way)
                if len(vehicle_wrong_way_history[tid]) > 15:  # 최근 15프레임만 유지
                    vehicle_wrong_way_history[tid] = vehicle_wrong_way_history[tid][-15:]

                # 최근 히스토리에서 과반수가 역주행이면 최종 역주행으로 판단
                if len(vehicle_wrong_way_history[tid]) >= 5:
                    recent_wrong_way_count = sum(vehicle_wrong_way_history[tid][-10:])
                    total_recent_frames = len(vehicle_wrong_way_history[tid][-10:])
                    if recent_wrong_way_count / total_recent_frames >= 0.4:
                        is_wrong_way = True
                    else:
                        is_wrong_way = False

                is_wrong_way = False

                # 색상 결정 (정지하지 않은 차량: 역주행 또는 일반)
                if is_wrong_way:
                    box_color = wrong_way_color
                    status_text = " (WRONG WAY)"
                else:
                    box_color = moving_color
                    status_text = ""

                # 바운딩 박스 그리기
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    box_color,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

                # ID 텍스트 표시
                text = f"ID:{tid}{status_text}"
                cv2.putText(
                    frame,
                    text,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                    cv2.LINE_AA
                )


    # 프레임을 출력 비디오에 저장
    out.write(frame)
    
    cv2.imshow("RT-DETR Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()