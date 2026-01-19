# Neuro-Inspired Charades Agent (Team Project)

> **2026 한국계산뇌과학회 겨울학교 팀 프로젝트**
> **주제:** 뇌 기전(Brain Mechanism)을 모방한 스피드 게임(몸으로 말해요) 에이전트 개발

## 프로젝트 소개
이 프로젝트는 인간의 시각 정보 처리 경로인 **Two-Stream Hypothesis (Ventral & Dorsal Stream)**를 모방하여 개발된 AI 에이전트입니다.
단순히 정지된 이미지를 인식하는 것을 넘어, **시간의 흐름에 따른 동작(Motion)**과 **맥락(Context)**을 파악하여 사용자의 제스처를 추론합니다.

##  핵심 기술 (Core Technology)

### 1. Dual-Stream Architecture (Main Brain)
* **Ventral Stream (What):** `Qwen2.5-VL-7B-Instruct` (VLM)을 사용하여 시각적 맥락과 객체의 형태를 정밀하게 분석합니다.
* **Dorsal Stream (Where/How):** `Sliding Window` 기법과 시각적 유동(Optical Flow) 분석을 통해 움직임의 궤적과 속도, 변화량을 포착합니다.
* **Prefrontal Cortex (Decision):** 두 경로의 정보를 통합하고, 메모리 버퍼(Memory Buffer)를 통해 시간적 맥락을 고려하여 최종 정답을 판단합니다.

### 2. Real-time Hand Gesture Recognition (Interaction)
> **역할:** 키보드나 마우스 없이, 손동작(숫자 1~9)만으로 게임 카테고리를 선택하거나 시스템을 제어하는 **비접촉 인터페이스(NUI)**를 구현했습니다.

* **Skeletal Analysis (MediaPipe):**
    * 웹캠 영상에서 실시간으로 손의 **21개 관절 랜드마크(Keypoints)**를 추출합니다.
    * 배경 노이즈에 강건하도록 `x, y, z` 좌표를 정규화(Normalization)하여 처리합니다.
* **Custom CNN Classifier:**
    * 추출된 63차원 좌표 벡터(21개 점 × 3축)를 입력으로 받는 경량화된 **CNN(Convolutional Neural Network)** 모델을 자체 적용했습니다.
    * **Temporal Smoothing:** `deque(maxlen=7)`를 이용한 이동 평균 필터를 적용하여, 손떨림이나 일시적인 오인식(Jittering)을 방지하고 예측의 안정성을 확보했습니다.
* **Visual Feedback:**
    * 인식된 숫자를 별도의 빈 캔버스(Black Canvas)에 텍스트로 출력하여, 사용자가 직관적으로 인식 결과를 확인할 수 있도록 UX를 설계했습니다.

### 3. Infrastructure
* **Model:** Qwen/Qwen2.5-VL-7B-Instruct (bfloat16)
* **Interface:** Gradio (Real-time Streaming)
* **Framework:** PyTorch (CNN), MediaPipe (Pose/Hand), Gradio (Web UI), OpenCV (Image Processing)

## 설치 및 실행 (Installation)

```bash
# 1. 환경 설정
conda create -n speed_game python=3.10
conda activate speed_game

# 2. 필수 라이브러리 설치
pip install torch transformers qwen_vl_utils gradio

# 3. 실행
python app.py
