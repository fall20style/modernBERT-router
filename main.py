from router import OnnxToolRouter
from tools import TOOL_REGISTRY


def main():
    print("⏳ ONNX Tool Router 로딩 중...")
    router = OnnxToolRouter(model_dir="./onnx_router", threshold=0.55)

    # 레지스트리의 모든 도구를 라우터에 등록
    for name, info in TOOL_REGISTRY.items():
        router.register_tool(name, info["description"])
    print(f"✅ 총 {len(TOOL_REGISTRY)}개 도구 등록 완료!\n")

    # 1. 배치 테스트 실행
    sample_queries = [
        "정자동 지금 우산 챙겨야 해?",
        "어제 시킨 운동화 배송 어디쯤 왔어?",
        "다음 달부터 멤버십 자동결제 해지해 줘",
        "내일 서울 날씨 알려주고 내 택배 어디 왔는지도 봐줘",  # 다중 도구 테스트
        "오늘 점심 메뉴 뭐 먹을까?"  # 일반 대화 테스트
    ]

    for q in sample_queries:
        res = router.route(q)
        selected_names = [t["name"] for t in res["selected_tools"]]
        print(f"[질문] {q} ({res['latency_ms']}ms)")
        print(f" -> 선택된 도구: {selected_names if selected_names else '없음 (일반 대화)'}")
        print(f" -> 상세 점수  : {res['all_scores']}\n")

    # 2. 실시간 터미널 입력 테스트 모드
    print("=" * 60)
    print("💬 실시간 테스트 모드 (종료하려면 'q' 또는 'exit' 입력)")
    print("=" * 60)
    while True:
        user_input = input("\n질문 입력 > ").strip()
        if user_input.lower() in ["q", "exit", "quit"]:
            break
        if not user_input:
            continue

        res = router.route(user_input)
        print(f"⏱️ 소요시간: {res['latency_ms']}ms")
        if not res["selected_tools"]:
            print("🤖 판단: 일반 대화 (호출할 도구 없음)")
        else:
            for tool in res["selected_tools"]:
                t_name = tool["name"]
                print(f"🎯 선택 도구: {t_name} (신뢰도: {tool['score']})")
                # 기본 인자로 함수 실행 예시 (추후 인자 추출기 연결 지점)
                print(f"   실행 결과: {TOOL_REGISTRY[t_name]['func']()}")


if __name__ == "__main__":
    main()
