import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class LightweightToolRouter:
    def __init__(self, model_name="xgen-ai/KoModernBERT-base-mlm-ecs-simcse"):
        print(f"[{model_name}] 모델 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        self.tools = []
        self.tool_embeddings = None

    def _get_embedding(self, texts: list[str]) -> torch.Tensor:
        # 메모리 절약을 위해 max_length를 256으로 제한
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        with torch.inference_mode():
            outputs = self.model(**inputs)
            # Mean Pooling (문장 전체의 의미 추출)
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            return F.normalize(embeddings, p=2, dim=1)

    def register_tools(self, tool_list: list[dict]):
        self.tools = tool_list
        # 긴 설명 대신 사용자 발화와 형태가 비슷한 '예시 문장'만 묶어서 임베딩
        tool_texts = [
            " / ".join(t['examples']) 
            for t in tool_list
        ]
        self.tool_embeddings = self._get_embedding(tool_texts)
        print(f"총 {len(self.tools)}개의 도구 등록 완료!")

    def route(self, query: str, threshold: float = 0.55):
        start_time = time.perf_counter()
        
        query_emb = self._get_embedding([query])
        # 코사인 유사도 계산 (내적)
        scores = torch.matmul(query_emb, self.tool_embeddings.T).squeeze(0).tolist()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        results = []
        for tool, score in zip(self.tools, scores):
            results.append({"name": tool["name"], "score": round(score, 4)})
            
        # 점수순 정렬
        results.sort(key=lambda x: x["score"], reverse=True)
        selected = [r for r in results if r["score"] >= threshold]
        
        return selected, results, round(elapsed_ms, 2)

# --- 실행 테스트 ---
if __name__ == "__main__":
    router = LightweightToolRouter()

    # 1. 도구 정의 및 등록
    sample_tools = [
        {
            "name": "get_weather",
            "description": "특정 지역이나 도시의 오늘, 내일, 주간 날씨와 기온, 강수 확률을 조회합니다.",
            "examples": ["서울 날씨 어때?", "내일 판교 비 와?", "주말 기온 알려줘"]
        },
        {
            "name": "search_order",
            "description": "사용자가 구매한 상품의 주문 내역, 배송 상태, 택배 위치를 조회합니다.",
            "examples": ["내 택배 어디쯤 왔어?", "어제 시킨 물건 배송 조회", "주문 내역 확인해줘"]
        },
        {
            "name": "cancel_subscription",
            "description": "유료 멤버십이나 정기 구독 서비스를 해지하고 결제를 취소합니다.",
            "examples": ["멤버십 해지하고 싶어요", "정기결제 취소해줘", "구독 끊는 법"]
        }
    ]
    router.register_tools(sample_tools)

    # 2. 테스트 쿼리 실행
    test_queries = [
        "분당 지금 우산 챙겨야 해?",
        "지난주에 주문한 운동화 아직 안 왔는데 확인 좀",
        "다음 달부터 돈 안 빠져나가게 해지해 줘",
        "오늘 점심 메뉴 뭐 먹을까?"  # 매칭되는 도구가 없어야 함
    ]

    for q in test_queries:
        selected, all_scores, latency = router.route(q)
        print(f"\n[질문] {q} (소요시간: {latency}ms)")
        print(f" -> 선택된 도구: {selected if selected else '없음 (일반 대화)'}")
        print(f" -> 전체 점수: {all_scores}")

