# Neuro-Inspired Charades Agent (Team Project)

> **2026 한국계산뇌과학회 겨울학교 팀 프로젝트**
> **주제:** 뇌 기전(Brain Mechanism)을 모방한 스피드 게임(몸으로 말해요) 에이전트 개발

## 프로젝트 소개
이 프로젝트는 인간의 시각 정보 처리 경로인 **Two-Stream Hypothesis (Ventral & Dorsal Stream)**를 모방하여 개발된 AI 에이전트입니다.
단순히 정지된 이미지를 인식하는 것을 넘어, **시간의 흐름에 따른 동작(Motion)**과 **맥락(Context)**을 파악하여 사용자의 제스처를 추론합니다.

## 핵심 기술 (Core Technology)

### 1. Dual-Stream Architecture
* **Dorsal Stream (Where/How):** * `Sliding Window` 기법과 `MediaPipe/Motion Analysis`를 통해 움직임의 궤적과 속도를 처리합니다.
    * 배경을 제거(Center Crop)하고 순수 동작에 집중합니다.
* **Ventral Stream (What):** * `Qwen2.5-VL-7B-Instruct` (VLM)을 사용하여 객체의 형태와 미세한 특징을 분석합니다.
* **Prefrontal Cortex (Decision):** * 두 경로의 정보를 통합하고, 이전 문맥(Memory Buffer)을 고려하여 최종 정답을 판단합니다.

### 2. Infrastructure
* **Model:** Qwen/Qwen2.5-VL-7B-Instruct (bfloat16)
* **Interface:** Gradio (Real-time Streaming)

## 🛠️ 설치 및 실행 (Installation)

```bash
# 1. 환경 설정
conda create -n speed_game python=3.10
conda activate speed_game

# 2. 필수 라이브러리 설치
pip install torch transformers qwen_vl_utils gradio

# 3. 실행
python app.py
