# Testing & Verification Guide

구축된 **ONNX Semantic Tool Router**의 기능 동작 검증, 벤치마크 시나리오, 그리고 운영 중 발생할 수 있는 이슈에 대한 트러블슈팅 가이드입니다.

---

## 1. 기본 동작 및 CLI 대화형 테스트 (`main.py`)

프로젝트 루트 경로에서 `main.py`를 실행하면 **① 사전 정의된 5개 핵심 시나리오 자동 테스트**가 먼저 실행된 후, **② 실시간 터미널 입력 모드**로 전환됩니다.

```bash
cd ~/work/modernBERT_router
source router_env/bin/activate
python main.py
```

### 정상 출력 예시
```text
⏳ ONNX Tool Router 로딩 중...
✅ 총 3개 도구 등록 완료!

[질문] 정자동 지금 우산 챙겨야 해? (142.35ms)
 -> 선택된 도구: ['get_weather']
 -> 상세 점수  : [{'name': 'get_weather', 'score': 1.0}, {'name': 'search_order', 'score': 0.0}, {'name': 'cancel_subscription', 'score': 0.0}]

[질문] 내일 서울 날씨 알려주고 내 택배 어디 왔는지도 봐줘 (158.12ms)
 -> 선택된 도구: ['get_weather', 'search_order']
 -> 상세 점수  : [{'name': 'get_weather', 'score': 0.9912}, {'name': 'search_order', 'score': 0.9845}, {'name': 'cancel_subscription', 'score': 0.0012}]

[질문] 오늘 점심 메뉴 뭐 먹을까? (138.90ms)
 -> 선택된 도구: 없음 (일반 대화)
 -> 상세 점수  : [{'name': 'get_weather', 'score': 0.0}, {'name': 'search_order', 'score': 0.0}, {'name': 'cancel_subscription', 'score': 0.0}]
```

---

## 2. 테스트 시나리오 체크리스트 (Test Cases)

새로운 도구를 추가하거나 모델 재학습 후 배포할 때 아래 4가지 유형의 테스트 케이스를 반드시 검증하세요.

| 테스트 유형 | 테스트 질문 예시 | 기대 결과 (Expected) | 검증 목적 |
| :--- | :--- | :--- | :--- |
| **단일 도구 (명시적)** | `"주문번호 987654 배송 상태 조회해 줘"` | `['search_order']` (`score >= 0.85`) | 핵심 키워드 매칭 정확도 |
| **단일 도구 (간접/구어체)** | `"내일 출근길에 패딩 입어야 할 정도로 추워?"` | `['get_weather']` (`score >= 0.80`) | 문맥 기반 일반화(Zero-shot 변주) 성능 |
| **다중 도구 (복합 의도)** | `"날씨 알려주고 멤버십 해지도 해줘"` | `['get_weather', 'cancel_subscription']` | 독립 스코어링 기반 Multi-Tool 동시 포착 |
| **일반 대화 (Negative)** | `"파이썬 딕셔너리 정렬하는 법 알려줘"` | `[]` (`모든 도구 score < 0.55`) | 오탐(False Positive) 방지 및 일반 대화 분기 |

---

## 3. 임계값(Threshold) 튜닝 가이드

`router.py` 초기화 시 `threshold` 값을 조정하여 서비스 성격에 맞게 민감도를 제어할 수 있습니다.

```python
router = OnnxToolRouter(model_dir="./onnx_router", threshold=0.55)
```

* **`threshold = 0.70 ~ 0.80` (보수적 운영):** 확실한 요청일 때만 도구를 호출하고, 조금이라도 애매하면 일반 LLM 대화로 넘기고 싶을 때 권장합니다.
* **`threshold = 0.50 ~ 0.55` (적극적 포착 - 기본 권장값):** *"결제한 책 출발했나요?"*처럼 두 도구의 키워드가 섞인 경계선 질문까지 놓치지 않고 도구로 연결할 때 적합합니다.

---

## 4. 트러블슈팅 (Troubleshooting)

| 증상 / 에러 메시지 | 원인 | 해결 방법 |
| :--- | :--- | :--- |
| `No such file or directory: './onnx_router/model.onnx'` | `onnx_router.zip` 압축이 풀리지 않았거나 하위 폴더 경로가 다름 | `ls -R`로 `model.onnx` 위치를 확인하고 `./onnx_router/` 직하위로 이동 |
| `IndexError: index out of range in self` | 토크나이저가 RoBERTa에서 지원하지 않는 `token_type_ids=1`을 생성함 | `router.py`의 `self.tokenizer(...)` 호출부에 `return_token_type_ids=False` 옵션 유지 |
| 추론 속도(Latency)가 500ms 이상으로 느림 | CPU 스레드 경합 또는 Swap 메모리 사용 발생 | `free -h`로 가용 RAM 확인 후, `router.py`의 `sess_options.intra_op_num_threads`를 서버 물리 코어 수(1 또는 2)에 맞게 조정 |
| 특정 질문에서 모든 도구 점수가 `0.55` 미만으로 나옴 | 학습 데이터에 해당 표현 유형(예: `"결제"` + `"배송"` 조합)이 부족함 | Colab 학습 노트북의 `synthetic_dataset`에 해당 표현 유형 10~20건을 추가해 재학습 |
