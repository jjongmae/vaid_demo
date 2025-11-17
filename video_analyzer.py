"""
영상 분석 엔진
정지차 및 사람 검출 기능
"""
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys


def get_resource_path(relative_path):
    """PyInstaller로 패키징된 경우 리소스 파일 경로 반환"""
    try:
        # PyInstaller로 패키징된 경우
        base_path = sys._MEIPASS
    except Exception:
        # 일반 Python 스크립트로 실행된 경우
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


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

            # 겹침이 심한 경우만 제거
            if iou > iou_threshold:
                # 클래스가 다른 경우는 겹쳐도 유지 (사람과 차량)
                if clss[i] != clss[j]:
                    continue

                # 포함 관계인지 확인
                if is_contained(boxes_xyxy[i], boxes_xyxy[j]) or is_contained(boxes_xyxy[j], boxes_xyxy[i]):
                    keep = False
                else:
                    keep = False
                break

        if keep:
            keep_indices.append(i)

    return (boxes_xyxy[keep_indices],
            [tids[i] for i in keep_indices],
            confs[keep_indices],
            [clss[i] for i in keep_indices])


class VideoAnalyzer:
    """영상 분석 클래스"""

    def __init__(self, model_path="rtdetr-l.pt"):
        """
        초기화

        Args:
            model_path: YOLO 모델 경로
        """
        # PyInstaller 패키징 대응
        model_path = get_resource_path(model_path)
        self.model = YOLO(model_path)

        # COCO dataset 클래스 ID
        self.target_classes = [0, 2, 5, 7]  # person, car, bus, truck
        self.vehicle_classes = [2, 5, 7]    # car, bus, truck
        self.person_class = 0

        # 색상 정의
        self.moving_color = (0, 255, 0)       # 움직이는 차량: 초록색
        self.parked_color = (0, 0, 255)       # 정지 차량: 빨간색
        self.person_color = (255, 0, 0)       # 사람: 파란색
        self.wrong_way_color = (255, 0, 255)  # 역주행 차량: 보라색

        # 기본 파라미터
        self.params = {
            'stop_frames': 150,           # 정지 판단 프레임 수
            'min_box_size': 50,           # 최소 박스 크기
            'stop_threshold_min': 12,     # 정지 판단 최소 임계값 (픽셀)
            'stop_threshold_ratio': 0.06, # 박스 크기 대비 정지 임계값 비율
            'conf_threshold': 0.5,        # 신뢰도 임계값
            'iou_threshold': 0.5,         # NMS IoU 임계값
            'track_iou': 0.8,             # 트래킹 IoU
            'detect_conf': 0.15,          # 검출 신뢰도
            # 역주행 감지 파라미터
            'use_wrong_way_detection': False,
            'wrong_way_frames': 10,
            'direction_vectors': []
        }

        # 추적 히스토리
        self.max_hist = 300
        self.reset_tracking()

    def reset_tracking(self):
        """추적 히스토리 초기화"""
        self.vehicle_positions = defaultdict(list)
        self.vehicle_stopped_frames = defaultdict(int)
        self.vehicle_wrong_way_frames = defaultdict(int)
        self.vehicle_move_vectors = defaultdict(list)  # 이동 벡터 저장을 위해 추가

    def update_params(self, **kwargs):
        """파라미터 업데이트"""
        for key, value in kwargs.items():
            # direction_vectors는 GUI의 리스트를 직접 참조하도록 설정
            if key == 'direction_vectors':
                self.params['direction_vectors'] = value
            elif key in self.params:
                self.params[key] = value

    def analyze_video(self, video_path, output_path, progress_callback=None):
        """
        영상 분석 및 결과 저장

        Args:
            video_path: 입력 영상 경로
            output_path: 출력 영상 경로
            progress_callback: 진행률 콜백 함수 (frame_idx, total_frames)

        Returns:
            bool: 성공 여부
        """
        # 결과 폴더 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        # 비디오 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 추적 히스토리 초기화
        self.reset_tracking()

        frame_idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1

            # 프레임 처리
            processed_frame = self.process_frame(frame, frame_idx)

            # 결과 저장
            out.write(processed_frame)

            # 진행률 콜백
            if progress_callback:
                progress_callback(frame_idx, total_frames)

        # 정리
        cap.release()
        out.release()

        return True

    def process_frame(self, frame, frame_idx):
        """
        프레임 처리

        Args:
            frame: 입력 프레임
            frame_idx: 프레임 인덱스

        Returns:
            처리된 프레임
        """
        # 객체 추적
        res = self.model.track(
            frame,
            persist=True,
            classes=self.target_classes,
            device='cpu',
            tracker=get_resource_path('my_bot.yaml'),
            iou=self.params['track_iou'],
            conf=self.params['detect_conf']
        )[0]

        if not (res.boxes and res.boxes.is_track):
            return frame

        # 박스 정보 추출
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        tids = res.boxes.id.int().cpu().tolist()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.int().cpu().tolist()

        # 신뢰도 필터링
        valid_indices = confs >= self.params['conf_threshold']
        boxes_xyxy = boxes_xyxy[valid_indices]
        tids = [tids[i] for i in range(len(tids)) if valid_indices[i]]
        clss = [clss[i] for i in range(len(clss)) if valid_indices[i]]
        confs = confs[valid_indices]

        # NMS 후처리
        boxes_xyxy, tids, confs, clss = nms_by_confidence(
            boxes_xyxy, tids, confs, clss,
            iou_threshold=self.params['iou_threshold']
        )

        # 정지 차량 검출
        stopped_vehicles = self._detect_stopped_vehicles(
            boxes_xyxy, tids, clss, frame_idx
        )

        # 역주행 차량 검출
        wrong_way_vehicles = []
        if self.params['use_wrong_way_detection']:
            wrong_way_vehicles = self._detect_wrong_way_vehicles(
                boxes_xyxy, tids, clss, frame_idx
            )

        # 정지 차량 그리기
        self._draw_stopped_vehicles(frame, stopped_vehicles)

        # 일반 차량, 사람, 역주행 차량 그리기
        self._draw_moving_objects(frame, boxes_xyxy, tids, clss, stopped_vehicles, wrong_way_vehicles)

        return frame

    def _detect_stopped_vehicles(self, boxes_xyxy, tids, clss, frame_idx):
        """정지 차량 검출"""
        stopped_vehicles = []

        for (x1, y1, x2, y2), tid, cls in zip(boxes_xyxy, tids, clss):
            if cls not in self.vehicle_classes:
                continue

            # 바운딩 박스 하단 중심점
            center_x = (x1 + x2) / 2
            center_y = y2

            # 위치 히스토리 업데이트
            if not self.vehicle_positions[tid] or self.vehicle_positions[tid][-1][2] != frame_idx:
                 self.vehicle_positions[tid].append((center_x, center_y, frame_idx))

            if len(self.vehicle_positions[tid]) > self.max_hist:
                self.vehicle_positions[tid] = self.vehicle_positions[tid][-self.max_hist:]

            # 박스 크기
            box_width = x2 - x1
            box_height = y2 - y1
            box_size = (box_width + box_height) / 2

            # 정지 상태 판단
            if (len(self.vehicle_positions[tid]) >= self.params['stop_frames'] and
                box_size >= self.params['min_box_size']):

                # 박스 크기에 비례한 정지 임계값
                adaptive_threshold = max(
                    self.params['stop_threshold_min'],
                    box_size * self.params['stop_threshold_ratio']
                )

                # 최근 프레임의 위치 변화 확인
                recent_positions = self.vehicle_positions[tid][-self.params['stop_frames']:]
                first_pos = recent_positions[0]
                max_distance = 0

                for pos in recent_positions[1:]:
                    distance = ((pos[0] - first_pos[0])**2 + (pos[1] - first_pos[1])**2)**0.5
                    max_distance = max(max_distance, distance)

                # 정지 상태 판단
                if max_distance < adaptive_threshold:
                    self.vehicle_stopped_frames[tid] += 1
                    stopped_vehicles.append((tid, x1, y1, x2, y2, cls))
                else:
                    self.vehicle_stopped_frames[tid] = 0
            else:
                self.vehicle_stopped_frames[tid] = 0

        return stopped_vehicles

    def _detect_wrong_way_vehicles(self, boxes_xyxy, tids, clss, frame_idx):
        """역주행 차량 검출 (전체 이동 방향 기반)"""
        wrong_way_vehicles = []
        if not self.params['direction_vectors']:
            return wrong_way_vehicles

        # 기준 방향 벡터 정규화
        ref_vectors = []
        for v in self.params['direction_vectors']:
            vec = np.array([v[2] - v[0], v[3] - v[1]], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                ref_vectors.append(vec / norm)

        if not ref_vectors:
            return wrong_way_vehicles

        for (x1, y1, x2, y2), tid, cls in zip(boxes_xyxy, tids, clss):
            if cls not in self.vehicle_classes:
                continue

            history = self.vehicle_positions[tid]

            # 같은 프레임에 대한 중복 계산 방지
            if not history or history[-1][2] != frame_idx:
                continue

            # 최소 프레임 수: 충분히 추적된 차량만 판단
            min_history_length = 10
            if len(history) < min_history_length:
                continue

            # 처음 위치부터 현재까지의 전체 이동 방향 계산
            pos_start = np.array(history[0][:2])  # 맨 처음 위치
            pos_now = np.array(history[-1][:2])   # 현재 위치

            move_vec = pos_now - pos_start
            move_dist = np.linalg.norm(move_vec)

            # 최소 이동 거리 체크: 충분히 이동한 차량만 판단 (노이즈 제거)
            min_move_distance = 15.0  # 픽셀
            if move_dist < min_move_distance:
                self.vehicle_wrong_way_frames[tid] = 0
                continue

            # 이동 방향 벡터 정규화
            move_vec_norm = move_vec / move_dist

            # 시각화를 위해 이동 벡터 저장
            self.vehicle_move_vectors[tid] = [move_vec]

            # 각 기준 벡터와 코사인 유사도 계산
            is_wrong_way = False
            for ref_vec in ref_vectors:
                cosine_similarity = np.dot(move_vec_norm, ref_vec)

                # 역주행 판단: 기준 방향과 반대 방향 (코사인 < -0.3, 약 107도 이상 차이)
                if cosine_similarity < -0.3:
                    is_wrong_way = True
                    break

            if is_wrong_way:
                self.vehicle_wrong_way_frames[tid] += 1
            else:
                # 점진적 감소 (급격한 초기화 방지)
                if self.vehicle_wrong_way_frames[tid] > 0:
                    self.vehicle_wrong_way_frames[tid] = max(0, self.vehicle_wrong_way_frames[tid] - 2)

            # 일정 프레임 이상 역주행 시 최종 판단
            if self.vehicle_wrong_way_frames[tid] >= self.params['wrong_way_frames']:
                wrong_way_vehicles.append((tid, x1, y1, x2, y2, cls))

        return wrong_way_vehicles


    def _draw_direction_vectors(self, frame):
        """기준 진행 방향 벡터를 화면에 표시"""
        for v in self.params['direction_vectors']:
            x1, y1, x2, y2 = v
            # 밝은 노란색으로 기준 방향 표시
            cv2.arrowedLine(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 255),  # 노란색
                3,
                tipLength=0.2
            )
            # "올바른 방향" 텍스트 표시
            cv2.putText(
                frame,
                "CORRECT DIR",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

    def _draw_stopped_vehicles(self, frame, stopped_vehicles):
        """정지 차량 그리기"""
        for tid, x1, y1, x2, y2, cls in stopped_vehicles:
            # 정지차는 빨간색으로 표시
            box_color = self.parked_color
            status_text = " (STOP)"

            # 바운딩 박스
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                box_color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # ID 텍스트
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

    def _draw_moving_objects(self, frame, boxes_xyxy, tids, clss, stopped_vehicles, wrong_way_vehicles):
        """일반 차량, 사람, 역주행 차량 그리기"""
        stopped_vehicle_ids = {tid for tid, _, _, _, _, _ in stopped_vehicles}
        wrong_way_vehicle_ids = {tid for tid, _, _, _, _, _ in wrong_way_vehicles}

        for (x1, y1, x2, y2), tid, cls in zip(boxes_xyxy, tids, clss):
            # 이미 처리된 차량은 스킵
            if tid in stopped_vehicle_ids:
                continue

            # 색상 및 텍스트 결정
            if tid in wrong_way_vehicle_ids:
                box_color = self.wrong_way_color
                status_text = " (WRONG WAY)"
            elif cls == self.person_class:
                box_color = self.person_color
                status_text = " (PERSON)"
            else:
                box_color = self.moving_color
                status_text = ""

            # 바운딩 박스
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                box_color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # ID 텍스트
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
