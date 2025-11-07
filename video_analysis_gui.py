"""
영상 분석 GUI 애플리케이션
비개발자를 위한 직관적인 인터페이스
"""
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QFileDialog, QGroupBox,
    QSpinBox, QDoubleSpinBox, QTextEdit, QGridLayout, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont
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


class VideoAnalysisApp(QMainWindow):
    """영상 분석 GUI 메인 윈도우"""

    def __init__(self):
        super().__init__()
        self.analyzer = VideoAnalyzer()
        self.analysis_thread = None
        self.video_path = None

        self.init_ui()

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
        """영상 선택 그룹 생성"""
        group = QGroupBox("영상 선택")
        layout = QVBoxLayout()

        # 영상 경로 표시
        path_layout = QHBoxLayout()
        self.video_path_label = QLabel("영상을 선택해주세요")
        self.video_path_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        path_layout.addWidget(self.video_path_label)
        layout.addLayout(path_layout)

        # 버튼들
        button_layout = QHBoxLayout()

        self.select_btn = QPushButton("영상 선택")
        self.select_btn.clicked.connect(self.select_video)
        button_layout.addWidget(self.select_btn)

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
        self.stop_frames_spin.setRange(10, 300)
        self.stop_frames_spin.setValue(30)
        self.stop_frames_spin.setSuffix(" 프레임")
        self.stop_frames_spin.setToolTip("차량이 정지차로 판단되는 프레임 수 (30fps 기준 1초)")
        layout.addWidget(self.stop_frames_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("최소 박스 크기:"), row, 0)
        self.min_box_spin = QSpinBox()
        self.min_box_spin.setRange(20, 200)
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

        group.setLayout(layout)
        return group

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
            self.video_path_label.setText(os.path.basename(file_path))
            self.analyze_btn.setEnabled(True)
            self.add_log(f"영상 선택됨: {os.path.basename(file_path)}")

    def apply_settings(self):
        """설정 적용"""
        self.analyzer.update_params(
            stop_frames=self.stop_frames_spin.value(),
            min_box_size=self.min_box_spin.value(),
            stop_threshold_min=self.stop_threshold_min_spin.value(),
            stop_threshold_ratio=self.stop_threshold_ratio_spin.value(),
            conf_threshold=self.conf_threshold_spin.value(),
            iou_threshold=self.iou_threshold_spin.value()
        )
        self.add_log("설정이 적용되었습니다.")

    def start_analysis(self):
        """분석 시작"""
        if not self.video_path:
            QMessageBox.warning(self, "경고", "영상을 먼저 선택해주세요.")
            return

        # 설정 자동 적용
        self.apply_settings()

        # 출력 경로 생성 (프로그램 실행 디렉토리 기준)
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_dir = os.path.join(os.getcwd(), "result")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_name}_analyzed.mp4")

        # UI 업데이트
        self.analyze_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("분석 중...")

        # 분석 스레드 시작
        self.analysis_thread = AnalysisThread(self.analyzer, self.video_path, output_path)
        self.analysis_thread.progress.connect(self.update_progress)
        self.analysis_thread.finished.connect(self.analysis_finished)
        self.analysis_thread.log.connect(self.add_log)
        self.analysis_thread.start()

    def update_progress(self, current, total):
        """진행률 업데이트"""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
            self.progress_label.setText(f"처리 중: {current}/{total} 프레임 ({percentage}%)")

    def analysis_finished(self, success, message):
        """분석 완료"""
        self.analyze_btn.setEnabled(True)
        self.select_btn.setEnabled(True)

        if success:
            self.progress_bar.setValue(100)
            self.progress_label.setText("완료!")
            QMessageBox.information(self, "완료", message)
        else:
            self.progress_label.setText("실패")
            QMessageBox.critical(self, "오류", message)

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
