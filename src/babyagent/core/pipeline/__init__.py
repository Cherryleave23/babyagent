"""智能管线模块

意图分类 → 路由分发 → 产品推荐硬管线 / 健康膳食 RAG 管线。

提供的公共接口:
  - classify_intent / classify_intent_sync: 意图分类
  - route_by_intent: 路由分发
  - recommend_products: 产品推荐四步硬管线
  - consult_health_diet: 健康膳食 RAG 管线
  - generate_rejection: 拒绝回复生成
  - is_emergency_situation: 紧急症状检测
  - MessageIntent / IntentResult: 数据模型
  - ProductRecommendInput / ProductRecommendOutput: 产品管线输入输出
  - extract_baby_needs / map_category_to_db / search_products_in_db / explain_recommendations: 单步工具
  - handle_empty_results: 空结果处理
"""

from .intent import (
    MessageIntent,
    IntentResult,
    classify_intent,
    classify_intent_sync,
    route_by_intent,
)

from .rejection import (
    generate_rejection,
    is_emergency_situation,
    get_rejection_message_by_type,
)

from .product_recommend import (
    ProductRecommendInput,
    ProductRecommendOutput,
    extract_baby_needs,
    map_category_to_db,
    search_products_in_db,
    explain_recommendations,
    handle_empty_results,
    recommend_products,
)

from .health_diet import (
    consult_health_diet,
)

__all__ = [
    # intent
    "MessageIntent",
    "IntentResult",
    "classify_intent",
    "classify_intent_sync",
    "route_by_intent",
    # rejection
    "generate_rejection",
    "is_emergency_situation",
    "get_rejection_message_by_type",
    # product_recommend
    "ProductRecommendInput",
    "ProductRecommendOutput",
    "extract_baby_needs",
    "map_category_to_db",
    "search_products_in_db",
    "explain_recommendations",
    "handle_empty_results",
    "recommend_products",
    # health_diet
    "consult_health_diet",
]
