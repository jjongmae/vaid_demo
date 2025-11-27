@echo off
chcp 65001 >nul
echo ====================================
echo 영상 분석 프로그램 EXE 빌드
echo ====================================
echo.

echo [중요] 빌드 전 확인 사항:
echo 1. rtdetr-l.pt 모델 파일이 현재 폴더에 있는지 확인
echo 2. my_bot.yaml 파일이 현재 폴더에 있는지 확인
echo.
pause

REM 파일 존재 확인
if not exist "rtdetr-l.pt" (
    echo [오류] rtdetr-l.pt 모델 파일이 없습니다!
    echo 모델을 먼저 다운로드하거나 복사해주세요.
    pause
    exit /b 1
)

if not exist "my_bot.yaml" (
    echo [오류] my_bot.yaml 파일이 없습니다!
    pause
    exit /b 1
)

REM PyInstaller 설치 확인
echo [1/4] PyInstaller 설치 확인 중...
pip show pyinstaller >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo PyInstaller가 설치되어 있지 않습니다. 설치 중...
    pip install pyinstaller
)

echo [2/4] 이전 빌드 파일 정리 중...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist "영상분석프로그램.spec" del "영상분석프로그램.spec"

echo [3/4] EXE 파일 빌드 중... (시간이 오래 걸립니다, 10-15분 소요, GPU 버전은 용량이 큽니다)
echo.
pyinstaller --clean ^
    --onefile ^
    --windowed ^
    --name "영상분석프로그램" ^
    --add-data "my_bot.yaml;." ^
    --add-data "rtdetr-l.pt;." ^
    --hidden-import=ultralytics ^
    --hidden-import=torch ^
    --hidden-import=torch.cuda ^
    --hidden-import=torchvision ^
    --hidden-import=cv2 ^
    --hidden-import=numpy ^
    --hidden-import=PyQt5 ^
    --collect-all ultralytics ^
    --collect-all torch ^
    --collect-all torchvision ^
    --copy-metadata ultralytics ^
    --copy-metadata torch ^
    --noconfirm ^
    video_analysis_gui.py

echo.
echo [4/4] 빌드 완료!
echo.
echo ====================================
echo EXE 파일 위치: dist\영상분석프로그램.exe
echo ====================================
echo.
echo 사용 방법:
echo 1. dist 폴더의 영상분석프로그램.exe를 복사
echo 2. 다른 PC에 붙여넣기
echo 3. 더블클릭으로 실행
echo.
echo 참고사항:
echo - Python 설치 불필요
echo - 모델 파일 포함됨
echo - GPU/CPU 자동 선택 (GPU 있으면 GPU 사용, 없으면 CPU 사용)
echo.

pause
