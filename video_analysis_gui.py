"""
영상 분석 GUI 애플리케이션
비개발자를 위한 직관적인 인터페이스
"""
import sys
import os
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QFileDialog, QGroupBox,
    QSpinBox, QDoubleSpinBox, QTextEdit, QGridLayout, QMessageBox, QDialog, QCheckBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QPoint
from PyQt5.QtGui import QFont, QImage, QPixmap, QPainter, QPen
from video_analyzer import VideoAnalyzer


class AnalysisThread(QThread):
    """영상 분석 작업 스레드"""
    progress = pyqtSignal(int, int)  # 현재 프레임, 전체 프레임
    finished = pyqtSignal(bool, str)  # 성공 여부, 메시지
    log = pyqtSignal(str)  # 로그 메시지

    def __init__(self, analyzer, video_path, output_path):
        super().__init__()
        self.analyzer = analyzer
        self.video_path = video_path
        self.output_path = output_path

    def run(self):
        """분석 실행"""
        try:
            self.log.emit(f"분석 시작: {os.path.basename(self.video_path)}")
            self.log.emit(f"출력 경로: {self.output_path}")

            # 분석 실행
            success = self.analyzer.analyze_video(
                self.video_path,
                self.output_path,
                progress_callback=self.progress.emit
            )

            if success:
                self.finished.emit(True, "영상 분석이 완료되었습니다!")
                self.log.emit("분석 완료!")
            else:
                self.finished.emit(False, "영상을 열 수 없습니다.")
                self.log.emit("오류: 영상을 열 수 없습니다.")

        except Exception as e:
            self.finished.emit(False, f"오류 발생: {str(e)}")
            self.log.emit(f"오류: {str(e)}")


class ImageLabel(QLabel):
    """마우스 이벤트를 처리하여 벡터를 그리는 QLabel (두 번 클릭 방식)"""
    vector_added = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_point = None
        self.current_pos = None  # 미리보기 선을 위한 현재 마우스 위치
        self.vectors = []
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)  # 마우스 버튼을 누르지 않아도 move 이벤트 발생

    def set_pixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.redraw()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.start_point is None:
                # 첫 번째 클릭: 시작점 설정
                self.start_point = event.pos()
            else:
                # 두 번째 클릭: 벡터 생성
                end_point = event.pos()
                if (end_point - self.start_point).manhattanLength() > 5:
                    new_vector = (self.start_point.x(), self.start_point.y(), end_point.x(), end_point.y())
                    self.vectors.append(new_vector)
                    self.vector_added.emit(new_vector)
                # 상태 초기화
                self.start_point = None
                self.current_pos = None
        self.redraw()

    def mouseMoveEvent(self, event):
        # 시작점이 설정된 경우에만 미리보기 선을 그림
        if self.start_point is not None:
            self.current_pos = event.pos()
            self.redraw()

    def redraw(self):
        if not hasattr(self, 'original_pixmap'):
            return
        pixmap = self.original_pixmap.copy()
        painter = QPainter(pixmap)

        # 기존 벡터들 그리기
        for vec in self.vectors:
            p1 = QPoint(vec[0], vec[1])
            p2 = QPoint(vec[2], vec[3])
            self.draw_arrow(painter, p1, p2, Qt.green)

        # 현재 그리는 벡터 미리보기
        if self.start_point and self.current_pos:
            self.draw_arrow(painter, self.start_point, self.current_pos, Qt.red)

        painter.end()
        super().setPixmap(pixmap)

    def draw_arrow(self, painter, p1, p2, color):
        """화살표 그리기"""
        pen = QPen(color, 2, Qt.SolidLine)
        painter.setPen(pen)
        painter.drawLine(p1, p2)

        # p1에서 p2를 향하는 벡터의 각도 계산
        angle = np.arctan2(p2.y() - p1.y(), p2.x() - p1.x())
        arrow_size = 10
        arrow_angle = np.pi / 6  # 30도

        # 화살촉의 양 날개 끝점 계산
        p2_x = p2.x()
        p2_y = p2.y()

        arrow_p1_x = p2_x - arrow_size * np.cos(angle - arrow_angle)
        arrow_p1_y = p2_y - arrow_size * np.sin(angle - arrow_angle)

        arrow_p2_x = p2_x - arrow_size * np.cos(angle + arrow_angle)
        arrow_p2_y = p2_y - arrow_size * np.sin(angle + arrow_angle)

        # QPoint는 정수 좌표를 사용
        arrow_p1 = QPoint(int(arrow_p1_x), int(arrow_p1_y))
        arrow_p2 = QPoint(int(arrow_p2_x), int(arrow_p2_y))

        painter.drawLine(p2, arrow_p1)
        painter.drawLine(p2, arrow_p2)

    def clear_vectors(self):
        self.vectors = []
        self.start_point = None
        self.current_pos = None
        self.redraw()


class DirectionSetter(QDialog):
    """진행 방향 설정 다이얼로그"""
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.vectors = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("진행 방향 설정")
        self.setModal(True)
        layout = QVBoxLayout()

        # 안내 문구
        info_label = QLabel("올바른 진행 방향을 마우스로 드래그하여 설정하세요. (여러 개 설정 가능)")
        layout.addWidget(info_label)

        # 영상 프레임 표시
        self.image_label = ImageLabel()
        self.image_label.vector_added.connect(lambda v: self.vectors.append(v))
        layout.addWidget(self.image_label)

        # 버튼
        button_layout = QHBoxLayout()
        clear_btn = QPushButton("초기화")
        clear_btn.clicked.connect(self.clear_all_vectors)
        button_layout.addWidget(clear_btn)

        self.ok_btn = QPushButton("확인")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.load_first_frame()

    def load_first_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QMessageBox.warning(self, "오류", "영상을 열 수 없습니다.")
            self.reject()
            return

        ret, frame = cap.read()
        cap.release()

        if not ret:
            QMessageBox.warning(self, "오류", "영상의 프레임을 읽을 수 없습니다.")
            self.reject()
            return

        # 프레임 크기 조정 (너무 크면 UI에 맞게 줄임)
        h, w, _ = frame.shape
        max_size = 800
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # OpenCV(BGR) 이미지를 PyQt(RGB) 형식으로 변환
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.set_pixmap(pixmap)

    def clear_all_vectors(self):
        self.vectors = []
        self.image_label.clear_vectors()

    def get_vectors(self):
        return self.vectors


class VideoAnalysisApp(QMainWindow):
    """영상 분석 GUI 메인 윈도우"""

    def __init__(self):
        super().__init__()
        self.analyzer = VideoAnalyzer()
        self.analysis_thread = None
        self.video_path = None  # 단일 영상 선택 시 사용
        self.video_paths = []   # 분석할 영상 목록
        self.current_video_index = 0
        self.direction_vectors = []  # 역주행 감지를 위한 진행 방향 벡터 목록
        self.output_result_dir = None # 분석 결과 폴더 경로

        self.init_ui()

        # 분석 장치 정보 로깅
        try:
            if self.analyzer.device == 'cuda' and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.add_log(f"분석 장치: GPU ({gpu_name})")
            else:
                self.add_log("분석 장치: CPU")
        except Exception as e:
            self.add_log(f"분석 장치 확인 중 오류 발생: {e}")

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("영상 분석 프로그램 - 정지차 및 사람 검출")
        self.setGeometry(100, 100, 800, 700)

        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # 제목
        title = QLabel("영상 분석 프로그램")
        title.setFont(QFont("맑은 고딕", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # 영상 선택 섹션
        video_group = self.create_video_selection_group()
        main_layout.addWidget(video_group)

        # 설정 섹션
        settings_group = self.create_settings_group()
        main_layout.addWidget(settings_group)

        # 진행 상태 섹션
        progress_group = self.create_progress_group()
        main_layout.addWidget(progress_group)

        # 로그 섹션
        log_group = self.create_log_group()
        main_layout.addWidget(log_group)

    def create_video_selection_group(self):
        """영상 또는 폴더 선택 그룹 생성"""
        group = QGroupBox("영상/폴더 선택")
        layout = QVBoxLayout()

        # 경로 표시
        path_layout = QHBoxLayout()
        self.video_path_label = QLabel("영상을 선택하거나 폴더를 선택해주세요")
        self.video_path_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        path_layout.addWidget(self.video_path_label)
        layout.addLayout(path_layout)

        # 버튼들
        button_layout = QHBoxLayout()

        self.select_btn = QPushButton("영상 선택")
        self.select_btn.clicked.connect(self.select_video)
        button_layout.addWidget(self.select_btn)

        self.select_folder_btn = QPushButton("폴더 선택")
        self.select_folder_btn.clicked.connect(self.select_folder)
        button_layout.addWidget(self.select_folder_btn)

        self.analyze_btn = QPushButton("분석 시작")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        button_layout.addWidget(self.analyze_btn)

        layout.addLayout(button_layout)
        group.setLayout(layout)
        return group

    def create_settings_group(self):
        """설정 그룹 생성"""
        group = QGroupBox("분석 설정 (고급)")
        layout = QGridLayout()

        row = 0

        # 정지 판단 프레임 수
        layout.addWidget(QLabel("정지 판단 프레임:"), row, 0)
        self.stop_frames_spin = QSpinBox()
        self.stop_frames_spin.setRange(1, 300)
        self.stop_frames_spin.setValue(150)
        self.stop_frames_spin.setSuffix(" 프레임")
        self.stop_frames_spin.setToolTip("차량이 정지차로 판단되는 프레임 수 (30fps 기준 5초)")
        layout.addWidget(self.stop_frames_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("최소 박스 크기:"), row, 0)
        self.min_box_spin = QSpinBox()
        self.min_box_spin.setRange(1, 200)
        self.min_box_spin.setValue(50)
        self.min_box_spin.setSuffix(" 픽셀")
        self.min_box_spin.setToolTip("정지 판단에 사용할 최소 박스 크기")
        layout.addWidget(self.min_box_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("정지 임계값 (최소):"), row, 0)
        self.stop_threshold_min_spin = QSpinBox()
        self.stop_threshold_min_spin.setRange(5, 50)
        self.stop_threshold_min_spin.setValue(12)
        self.stop_threshold_min_spin.setSuffix(" 픽셀")
        self.stop_threshold_min_spin.setToolTip("정지로 판단하는 최소 이동 거리")
        layout.addWidget(self.stop_threshold_min_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("정지 임계값 (비율):"), row, 0)
        self.stop_threshold_ratio_spin = QDoubleSpinBox()
        self.stop_threshold_ratio_spin.setRange(0.01, 0.2)
        self.stop_threshold_ratio_spin.setValue(0.06)
        self.stop_threshold_ratio_spin.setSingleStep(0.01)
        self.stop_threshold_ratio_spin.setDecimals(2)
        self.stop_threshold_ratio_spin.setToolTip("박스 크기 대비 정지 임계값 비율")
        layout.addWidget(self.stop_threshold_ratio_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("신뢰도 임계값:"), row, 0)
        self.conf_threshold_spin = QDoubleSpinBox()
        self.conf_threshold_spin.setRange(0.1, 0.95)
        self.conf_threshold_spin.setValue(0.5)
        self.conf_threshold_spin.setSingleStep(0.05)
        self.conf_threshold_spin.setDecimals(2)
        self.conf_threshold_spin.setToolTip("객체 검출 신뢰도 임계값")
        layout.addWidget(self.conf_threshold_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("NMS IoU 임계값:"), row, 0)
        self.iou_threshold_spin = QDoubleSpinBox()
        self.iou_threshold_spin.setRange(0.1, 0.9)
        self.iou_threshold_spin.setValue(0.5)
        self.iou_threshold_spin.setSingleStep(0.05)
        self.iou_threshold_spin.setDecimals(2)
        self.iou_threshold_spin.setToolTip("겹치는 박스 제거 IoU 임계값")
        layout.addWidget(self.iou_threshold_spin, row, 1)

        row += 1
        # 역주행 감지 섹션
        self.wrong_way_checkbox = QCheckBox("역주행 감지 사용")
        self.wrong_way_checkbox.toggled.connect(self.toggle_wrong_way_settings)
        layout.addWidget(self.wrong_way_checkbox, row, 0)

        self.set_direction_btn = QPushButton("진행 방향 설정")
        self.set_direction_btn.clicked.connect(self.open_direction_setter)
        layout.addWidget(self.set_direction_btn, row, 1)

        row += 1
        layout.addWidget(QLabel("역주행 판단 프레임:"), row, 0)
        self.wrong_way_frames_spin = QSpinBox()
        self.wrong_way_frames_spin.setRange(5, 60)
        self.wrong_way_frames_spin.setValue(10)
        self.wrong_way_frames_spin.setSuffix(" 프레임")
        self.wrong_way_frames_spin.setToolTip("역주행으로 최종 판단하기까지 필요한 최소 프레임 수. 높을수록 오검지가 줄어듭니다.")
        layout.addWidget(self.wrong_way_frames_spin, row, 1)

        # 초기에는 비활성화
        self.wrong_way_frames_spin.setEnabled(False)
        self.set_direction_btn.setEnabled(False)

        group.setLayout(layout)
        return group

    def toggle_wrong_way_settings(self, checked):
        """역주행 관련 설정 활성화/비활성화"""
        self.wrong_way_frames_spin.setEnabled(checked)
        self.set_direction_btn.setEnabled(checked)
        if checked:
            self.add_log("역주행 감지 기능이 활성화되었습니다. '진행 방향 설정'을 해주세요.")
        else:
            self.add_log("역주행 감지 기능이 비활성화되었습니다.")
            self.direction_vectors = []  # 기능 비활성화 시 벡터 초기화

    def open_direction_setter(self):
        """진행 방향 설정 창 열기"""
        # 분석할 영상이 먼저 선택되어야 함
        video_to_use = None
        if self.video_paths:
            video_to_use = self.video_paths[0]
        elif self.video_path:
            video_to_use = self.video_path

        if not video_to_use:
            QMessageBox.warning(self, "경고", "먼저 분석할 영상을 선택해주세요.")
            return

        # 방향 설정 다이얼로그 열기
        dialog = DirectionSetter(video_to_use, self)
        if dialog.exec_() == QDialog.Accepted:
            self.direction_vectors = dialog.get_vectors()
            self.add_log(f"{len(self.direction_vectors)}개의 진행 방향 벡터가 설정되었습니다.")
            if not self.direction_vectors:
                self.add_log("설정된 벡터가 없습니다. 역주행 감지가 제대로 동작하지 않을 수 있습니다.")
        else:
            self.add_log("진행 방향 설정이 취소되었습니다.")

    def create_progress_group(self):
        """진행 상태 그룹 생성"""
        group = QGroupBox("진행 상태")
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("대기 중...")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)

        group.setLayout(layout)
        return group

    def create_log_group(self):
        """로그 그룹 생성"""
        group = QGroupBox("로그")
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)

        # 로그 초기화 버튼
        clear_btn = QPushButton("로그 지우기")
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)

        group.setLayout(layout)
        return group

    def select_video(self):
        """영상 파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "영상 파일 선택",
            os.getcwd(),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )

        if file_path:
            self.video_path = file_path
            self.video_paths = [file_path]  # 분석 목록을 단일 파일로 설정
            self.video_path_label.setText(os.path.basename(file_path))
            self.analyze_btn.setEnabled(True)
            self.add_log(f"영상 선택됨: {os.path.basename(file_path)}")

    def select_folder(self):
        """영상 폴더 선택하여 분석 목록에 추가"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "영상 폴더 선택",
            os.getcwd()
        )

        if dir_path:
            self.video_paths = []
            valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
            for filename in sorted(os.listdir(dir_path)): # 파일 이름 순으로 정렬
                if filename.lower().endswith(valid_extensions):
                    self.video_paths.append(os.path.join(dir_path, filename))

            if self.video_paths:
                self.video_path = None  # 단일 영상 선택 해제
                self.video_path_label.setText(f"폴더: {os.path.basename(dir_path)} ({len(self.video_paths)}개 영상)")
                self.analyze_btn.setEnabled(True)
                self.add_log(f"폴더에서 {len(self.video_paths)}개의 영상을 분석 목록에 추가했습니다.")
            else:
                QMessageBox.warning(self, "경고", "선택한 폴더에 분석할 영상 파일이 없습니다.")
                self.video_path_label.setText("영상을 선택하거나 폴더를 선택해주세요.")
                self.analyze_btn.setEnabled(False)

    def apply_settings(self):
        """설정 적용"""
        self.analyzer.update_params(
            stop_frames=self.stop_frames_spin.value(),
            min_box_size=self.min_box_spin.value(),
            stop_threshold_min=self.stop_threshold_min_spin.value(),
            stop_threshold_ratio=self.stop_threshold_ratio_spin.value(),
            conf_threshold=self.conf_threshold_spin.value(),
            iou_threshold=self.iou_threshold_spin.value(),
            # 역주행 감지 관련 파라미터 추가
            use_wrong_way_detection=self.wrong_way_checkbox.isChecked(),
            wrong_way_frames=self.wrong_way_frames_spin.value(),
            direction_vectors=self.direction_vectors
        )
        self.add_log("설정이 적용되었습니다.")

    def start_analysis(self):
        """분석 시작 (목록에 있는 모든 영상)"""
        if not self.video_paths:
            QMessageBox.warning(self, "경고", "영상을 먼저 선택하거나 폴더를 선택해주세요.")
            return

        # 역주행 감지 설정 검증
        if self.wrong_way_checkbox.isChecked():
            if not self.direction_vectors:
                QMessageBox.critical(
                    self,
                    "역주행 방향 미설정",
                    "역주행 감지가 활성화되어 있지만, 진행 방향이 설정되지 않았습니다.\n\n"
                    "진행 방향을 설정하지 않으면 역주행 감지가 동작하지 않습니다.\n\n"
                    "'진행 방향 설정' 버튼을 클릭하여 방향을 먼저 설정해주세요."
                )
                self.add_log("분석이 취소되었습니다. '진행 방향 설정'을 먼저 진행해주세요.")
                return

        # 설정 자동 적용
        self.apply_settings()

        self.current_video_index = 0
        self.analyze_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.select_folder_btn.setEnabled(False)

        self.start_next_analysis()

    def start_next_analysis(self):
        """목록의 다음 영상 분석을 시작"""
        if self.current_video_index >= len(self.video_paths):
            # 모든 분석이 완료된 경우
            self.reset_ui_after_analysis(all_successful=True)
            QMessageBox.information(self, "완료", "모든 영상 분석이 완료되었습니다!")
            return

        video_path = self.video_paths[self.current_video_index]

        # 출력 경로 생성
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.getcwd(), "result")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_name}_analyzed.mp4")
        self.output_result_dir = output_dir # 결과 폴더 경로 저장

        # UI 업데이트
        self.progress_bar.setValue(0)
        total_videos = len(self.video_paths)
        progress_text = f"분석 준비 중 ({self.current_video_index + 1}/{total_videos}): {os.path.basename(video_path)}"
        self.progress_label.setText(progress_text)
        self.add_log(progress_text)

        # Add device log before starting analysis
        if self.analyzer.device == 'cuda':
            self.add_log(f"영상 분석을 GPU ({torch.cuda.get_device_name(0)})로 시작합니다.")
        else:
            self.add_log("영상 분석을 CPU로 시작합니다.")

        # 분석 스레드 시작
        self.analysis_thread = AnalysisThread(self.analyzer, video_path, output_path)
        self.analysis_thread.progress.connect(self.update_progress)
        self.analysis_thread.finished.connect(self.one_analysis_finished)
        self.analysis_thread.log.connect(self.add_log)
        self.analysis_thread.start()

    def update_progress(self, current, total):
        """진행률 업데이트"""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)

            total_videos = len(self.video_paths)
            if not self.video_paths: # 분석이 끝난 직후 video_paths가 비워지는 경우 방지
                return
                
            video_name = os.path.basename(self.video_paths[self.current_video_index])

            if total_videos > 1:
                progress_text = f"처리 중 ({self.current_video_index + 1}/{total_videos}: {video_name}): {current}/{total} 프레임 ({percentage}%)"
            else:
                progress_text = f"처리 중: {current}/{total} 프레임 ({percentage}%)"
            self.progress_label.setText(progress_text)

    def one_analysis_finished(self, success, message):
        """단일 영상 분석 완료 시 호출"""
        if success:
            self.add_log(f"분석 완료: {os.path.basename(self.video_paths[self.current_video_index])}")
            self.current_video_index += 1
            self.start_next_analysis()  # 다음 영상 분석 시작
        else:
            self.add_log(f"분석 실패: {os.path.basename(self.video_paths[self.current_video_index])}. 원인: {message}")
            self.reset_ui_after_analysis(all_successful=False)
            QMessageBox.critical(self, "오류", f"분석 중 오류가 발생하여 중단합니다:\n{message}")

    def reset_ui_after_analysis(self, all_successful):
        """분석 세션 종료 후 UI 초기화"""
        if all_successful:
            self.progress_label.setText("모든 분석 완료!")
            self.progress_bar.setValue(100)
            # 분석 완료 후 결과 폴더 열기
            if self.output_result_dir and os.path.exists(self.output_result_dir):
                try:
                    os.startfile(self.output_result_dir)
                    self.add_log(f"결과 폴더 열기: {self.output_result_dir}")
                except Exception as e:
                    self.add_log(f"결과 폴더 열기 실패: {e}")
        else:
            self.progress_label.setText("분석 중단됨")
            self.progress_bar.setValue(0)

        self.analyze_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.select_folder_btn.setEnabled(True)
        self.video_paths = []
        self.current_video_index = 0
        self.output_result_dir = None # 폴더 경로 초기화

    def add_log(self, message):
        """로그 추가"""
        self.log_text.append(f"[{self.get_timestamp()}] {message}")

    @staticmethod
    def get_timestamp():
        """현재 시간 문자열"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")


def main():
    """메인 함수"""
    app = QApplication(sys.argv)

    # 한글 폰트 설정
    font = QFont("맑은 고딕", 10)
    app.setFont(font)

    window = VideoAnalysisApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

