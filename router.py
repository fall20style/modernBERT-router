import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class OnnxToolRouter:
    def __init__(self, model_dir: str = "./onnx_router", threshold: float = 0.55):
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # CPU 스레드 최적화 및 메모리 절약 세션 옵션
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        
        self.session = ort.InferenceSession(
            f"{model_dir}/model.onnx",
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.tools = {}

    def register_tool(self, name: str, description: str):
        self.tools[name] = f"Tool: {name} | Desc: {description}"

    def route(self, query: str) -> dict:
        if not self.tools:
            raise ValueError("등록된 도구가 없습니다. register_tool()로 도구를 먼저 등록하세요.")

        start_time = time.perf_counter()
        
        tool_names = list(self.tools.keys())
        tool_texts = list(self.tools.values())

        # 질문과 등록된 모든 도구를 1:N 배치로 묶어 토크나이징
        inputs = self.tokenizer(
            [query] * len(tool_texts),
            tool_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="np",
            return_token_type_ids=False  # RoBERTa 인덱스 초과 방지
        )

        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }

        # ONNX 추론 및 [0.0, 1.0] 확률 스케일링
        raw_logits, = self.session.run(["logits"], ort_inputs)
        scores = np.clip(raw_logits.squeeze(-1), 0.0, 1.0)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # 점수 내림차순 정렬
        all_scores = [
            {"name": name, "score": round(float(score), 4)}
            for name, score in zip(tool_names, scores)
        ]
        all_scores.sort(key=lambda x: x["score"], reverse=True)

        # 임계값(threshold) 이상인 도구만 필터링 (다중 도구 지원)
        selected = [item for item in all_scores if item["score"] >= self.threshold]

        return {
            "query": query,
            "selected_tools": selected,
            "all_scores": all_scores,
            "latency_ms": elapsed_ms
        }
