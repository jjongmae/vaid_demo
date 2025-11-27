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
print("Loading model...")
model       = YOLO("rtdetr-x.pt")                       # 모델
print("Model loaded successfully")

video_path  = r"input\본선_정지차량_야간_비_공세육교_20250804011943_20250804033237_서울방향 정지물체.avi"
print(f"Video path: {video_path}")

import os
if not os.path.exists(video_path):
    print(f"ERROR: Video file not found at: {video_path}")
    exit(1)
print("Video file exists")

cap         = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("ERROR: Cannot open video file")
    exit(1)
print(f"Video opened successfully - FPS: {cap.get(cv2.CAP_PROP_FPS)}, Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

track_hist  = defaultdict(list)                        # {id: [(x, y), ...]}
max_hist    = 30                                       # 궤적 길이
# 사람, 자동차, 트럭, 버스 검지 (COCO dataset class IDs)
target_classes = [0, 2, 5, 7]  # person, car, bus, truck
vehicle_classes = [2, 5, 7]    # car, bus, truck

# 차량 위치 히스토리 관리
vehicle_positions = defaultdict(list)  # {id: [(center_x, center_y, frame_idx), ...]}
stationary_threshold = 20  # 20픽셀 이내 움직임을 정지로 판단
stationary_frame_count = 30  # 30프레임(약 1초) 이상 정지시 정지차로 판단
min_box_size = 20  # 정지차 판단에 사용할 최소 박스 크기 (가로+세로 평균)

# 박스 색상
moving_color = (0, 255, 0)      # 움직이는 차량: 초록색
stationary_color = (0, 0, 255)  # 정지 차량: 빨간색
person_color = (255, 0, 0)      # 사람: 파란색

# 비디오 출력 설정
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# ROI 설정 - 왼쪽 40% 제외
roi_x_start = int(width * 0.4)  # 화면의 40% 지점부터 시작
print(f"ROI: Excluding left 40% of frame (x < {roi_x_start})")

frame_idx = 0
print("Starting frame processing...")
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print(f"Finished processing. Total frames: {frame_idx}")
        break
    frame_idx += 1

    if frame_idx % 30 == 0:  # 매 30프레임마다 진행상황 출력
        print(f"Processing frame {frame_idx}...")

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

        for (x1, y1, x2, y2), tid, conf, cls in zip(boxes_xyxy, tids, confs, clss):
            # 바운딩 박스 중심점 계산
            center_x = (x1 + x2) / 2

            # ROI 필터링: 왼쪽 40% 영역에 있는 객체는 제외
            if center_x < roi_x_start:
                continue

            if cls in vehicle_classes:
                # 바운딩 박스 하단 중심점 계산 (차량 위치 추적용)
                center_y = y2
                
                # 차량 위치 히스토리 업데이트
                vehicle_positions[tid].append((center_x, center_y, frame_idx))
                
                # 오래된 히스토리 제거 (메모리 절약)
                if len(vehicle_positions[tid]) > max_hist:
                    vehicle_positions[tid] = vehicle_positions[tid][-max_hist:]
                
                # 정지 차량 판단 (박스 크기 필터링 + 적응적 임계값)
                is_stationary = False
                box_width = x2 - x1
                box_height = y2 - y1
                box_size = (box_width + box_height) / 2

                # 박스가 충분히 큰 경우에만 정지차 판단 수행 (멀어지는 차량 제외)
                if (len(vehicle_positions[tid]) >= stationary_frame_count and
                    box_size >= min_box_size):

                    # 박스 크기에 비례한 정지 임계값 (최소 12픽셀, 박스 크기의 6%)
                    adaptive_threshold = max(12, box_size * 0.06)

                    # 최근 프레임의 위치 변화 확인
                    recent_positions = vehicle_positions[tid][-stationary_frame_count:]
                    first_pos = recent_positions[0]
                    max_distance = 0

                    for pos in recent_positions[1:]:
                        distance = ((pos[0] - first_pos[0])**2 + (pos[1] - first_pos[1])**2)**0.5
                        max_distance = max(max_distance, distance)

                    if max_distance < adaptive_threshold:
                        is_stationary = True

                # 색상 결정 (정지 차량 vs 일반 차량)
                if is_stationary:
                    box_color = stationary_color
                else:
                    box_color = moving_color
                
                # 바운딩 박스 그리기
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    box_color,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                
                # ID 텍스트 표시 (정지차 표시)
                if is_stationary:
                    status_text = " (STOP)"
                else:
                    status_text = ""
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
            elif cls == 0: # Person
                # 사람일 경우 파란색으로 박스 표시
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    person_color,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                text = f"ID:{tid} (Person)"
                cv2.putText(
                    frame,
                    text,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    person_color,
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