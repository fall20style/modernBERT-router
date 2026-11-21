def get_weather(location: str = "분당 정자동", date: str = "오늘") -> str:
    return f"🌤️ [{date}] {location} 날씨는 맑고, 현재 기온은 8°C입니다."

def search_order(item_name: str = None, order_id: str = None) -> str:
    target = f"주문번호 {order_id}" if order_id else f"상품 '{item_name}'"
    return f"📦 [{target}] 현재 성남 물류터미널에서 배송 중입니다."

def cancel_subscription(plan_name: str = "전체", immediate: bool = False) -> str:
    mode = "즉시 해지" if immediate else "다음 결제일 만료 해지"
    return f"✅ [{plan_name}] 구독 해지 접수 완료 ({mode})"


# 라우터 등록 및 함수 매핑용 레지스트리
TOOL_REGISTRY = {
    "get_weather": {
        "description": "특정 지역의 날짜별 날씨와 기온, 강수 확률을 조회합니다.",
        "func": get_weather
    },
    "search_order": {
        "description": "사용자가 구매한 상품의 주문 내역, 배송 상태, 택배 위치를 조회합니다.",
        "func": search_order
    },
    "cancel_subscription": {
        "description": "유료 멤버십이나 정기 구독 서비스를 해지하고 결제를 취소합니다.",
        "func": cancel_subscription
    }
}
