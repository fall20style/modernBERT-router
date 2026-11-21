# Installation & Deployment Guide

저사양 Linux 서버(CPU 전용, 가용 RAM 1.4GB 이하 환경)에서 메모리 부족(OOM) 없이 **ONNX Semantic Tool Router**를 설치하고 배포하는 상세 매뉴얼입니다.

---

## 1. 시스템 요구사항 (Prerequisites)

* **OS:** Linux (Ubuntu 20.04 / 22.04 / 24.04 권장)
* **Python:** Python 3.10 이상
* **Hardware:** CPU 2 Cores 이상, 가용 RAM 512MB 이상 (GPU 불필요)
* **필수 유틸리티:** `unzip` (`sudo apt update && sudo apt install -y unzip`)

---

## 2. 프로젝트 디렉토리 및 가상환경 세팅

1. **작업 디렉토리 이동 및 가상환경 생성**
   ```bash
   mkdir -p ~/work/modernBERT_router
   cd ~/work/modernBERT_router

   python3 -m venv router_env
   source router_env/bin/activate
   ```

2. **pip 최신화**
   ```bash
   pip install --upgrade pip
   ```

---

## 3. 경량 의존성 패키지 설치

추론 서버에는 용량이 큰 `torch`(PyTorch)를 설치하지 않고, CPU 추론에 최적화된 `onnxruntime`만 설치합니다.

1. **`requirements.txt` 확인**
   ```text
   numpy>=1.24.0
   transformers>=4.40.0
   onnxruntime>=1.17.0
   ```

2. **캐시 없이 패키지 설치 (디스크 및 RAM 절약)**
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```

---

## 4. 학습된 ONNX 모델 배포 (`onnx_router.zip`)

Google Colab에서 파인튜닝 및 변환을 마친 `onnx_router.zip` 파일을 서버의 `~/work/modernBERT_router/` 경로로 업로드한 뒤 압축을 해제합니다.

```bash
cd ~/work/modernBERT_router
unzip -o onnx_router.zip
```

### 압축 해제 후 필수 파일 확인
아래 명령어로 `onnx_router/` 폴더 내부에 모델과 토크나이저 파일이 정상적으로 위치하는지 확인합니다.

```bash
ls -lh ./onnx_router/
```
* **필수 존재 파일:**
  * `model.onnx` (약 420MB)
  * `tokenizer.json`
  * `tokenizer_config.json`

---

## 5. 설치 검증 및 리소스 확인

1. **정상 로드 테스트**
   ```bash
   python -c "import onnxruntime as ort; print('ONNX Runtime Providers:', ort.get_available_providers())"
   ```
   * 출력에 `'CPUExecutionProvider'`가 포함되면 정상입니다.

2. **실행 중 시스템 RAM 점유율 모니터링**
   ```bash
   free -h
   ```
   * `available` 메모리가 안정적으로 유지되는지 확인합니다.
