# Lightweight Semantic Tool Router (ONNX & Cross-Encoder)

저사양 CPU 서버(가용 RAM 1.4GB 이하) 환경에서 초고속·고정밀로 사용자 발화의 의도를 분류하고 적절한 외부 도구(Function/API)를 라우팅하기 위해 구축한 **ONNX Cross-Encoder 기반 Semantic Tool Router** 프로젝트입니다.

https://github.com/zune2/colab-notebooks/blob/main/2026-11-21-Qwen2.5-7B-Instruct-Cross-Encoder-LoRA.ipynb

---

## 1. 프로젝트 개요 (Overview)

* **개발 배경:** GPU가 없는 저사양 Linux 서버에서는 7B 이상의 Local LLM 구동이 불가능하며, 매 요청마다 수십 개의 도구 스키마를 외부 LLM에 전달하면 토큰 비용과 응답 지연 시간(Latency)이 급증합니다.
* **핵심 아키텍처:**
  1. **경량 Cross-Encoder 라우팅:** `klue/roberta-base` 기반 파인튜닝 모델이 `[질문] + [도구 명세]` 쌍의 문맥적 연관성을 직접 채점하여 호출할 도구만 필터링합니다.
  2. **ONNX Runtime 단독 추론:** 추론 서버에는 무거운 PyTorch를 제거하고 순수 `onnxruntime`만 사용하여 메모리 점유율(약 250~300MB)과 CPU 추론 시간을 최적화했습니다.
  3. **다중 도구 및 일반 대화 분기:** 독립 스코어링(0.0~1.0) 방식을 사용하여 복합 질문 시 여러 도구를 동시에 선택하거나, 임계값(`0.55`) 미만일 경우 일반 대화(`None`)로 안전하게 분기합니다.

---

## 2. 디렉토리 구조 (Project Structure)

```text
modernBERT_router/
├── README.md                  # 프로젝트 개요 및 빠른 시작 가이드
├── INSTALL.md                 # Linux 서버 환경 구축 및 모델 배포 상세 매뉴얼
├── TEST.md                    # 단위 테스트, 벤치마크 검증 및 트러블슈팅 가이드
├── requirements.txt           # 추론 서버용 경량 Python 의존성 목록
├── router.py                  # ONNX Cross-Encoder 라우터 핵심 엔진 클래스
├── tools.py                   # 실행할 Python 함수 및 도구 스키마 레지스트리
├── main.py                    # 배치 테스트 및 실시간 CLI 대화형 실행기
└── onnx_router/               # Colab에서 학습 후 변환된 ONNX 모델 디렉토리
    ├── model.onnx
    ├── tokenizer.json
    └── tokenizer_config.json
```

---

## 3. 빠른 시작 (Quick Start)

1. **환경 설치 및 모델 압축 해제** (상세 내용은 [INSTALL.md](./INSTALL.md) 참조)
   ```bash
   python3 -m venv router_env
   source router_env/bin/activate
   pip install --no-cache-dir -r requirements.txt
   unzip -o onnx_router.zip
   ```

2. **라우터 실행 및 테스트** (상세 내용은 [TEST.md](./TEST.md) 참조)
   ```bash
   python main.py
   ```

---

## 4. 성능 요약 (Benchmark Summary)

* **1단계 함수 라우팅 정확도 (Tool Accuracy):** `92.86%` (미학습 평가셋 기준, Threshold `0.55`)
* **2단계 함수+인자 최종 정확도 (E2E Accuracy):** `71.43%` (경량 LLM 인자 추출 결합 기준)
* **추론 서버 메모리 점유율:** 약 `280MB` (PyTorch 미설치, `onnxruntime` CPU 단독 구동 기준)
