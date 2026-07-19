"""产品推荐硬管线 — 四步防幻觉管线

严格遵循 I1 不变量：推荐的产品名称必须全部来自数据库，LLM 仅负责
需求提取和推荐理由解释。管线分为四个不可跳过的步骤（R2-R4-R3-解释）：

  步骤 1: extract_baby_needs       — LLM 提取结构化需求（R2）
  步骤 2: map_category_to_db        — 品类映射到 DB 分类（R4）
  步骤 3: search_products_in_db     — 纯 DB 产品搜索（R3）
  步骤 4: explain_recommendations   — LLM 生成推荐理由

空结果处理（R40）：返回提示+可提供健康膳食建议。
"""

from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class ProductRecommendInput(BaseModel):
    """产品推荐管线输入

    Attributes:
        baby_profile: 宝宝档案信息，含 age_months, allergies, dietary_restrictions 等。
        employee_message: 员工原始消息文本。
        model_config: LLM 模型配置字典（api_key, base_url, model_name 等）。
    """

    baby_profile: dict[str, Any] = Field(default_factory=dict, description="宝宝档案")
    employee_message: str = Field(default="", description="员工消息")
    llm_config: dict[str, Any] = Field(default_factory=dict, description="模型配置")


class ProductRecommendOutput(BaseModel):
    """产品推荐管线输出

    Attributes:
        extracted_needs: LLM 提取的结构化需求。
        matched_category: 映射后的 DB 分类名称。
        recommended_products: 来自 DB 的推荐产品列表（非 LLM 生成）。
        explanation: LLM 生成的推荐理由文本。
    """

    extracted_needs: dict[str, Any] = Field(default_factory=dict, description="提取的需求")
    matched_category: str = Field(default="", description="DB 分类")
    recommended_products: list[dict[str, Any]] = Field(default_factory=list, description="推荐产品")
    explanation: str = Field(default="", description="推荐理由")


# ---------------------------------------------------------------------------
# LLM 提示词
# ---------------------------------------------------------------------------

_NEED_EXTRACTION_SYSTEM_PROMPT = """你是一个母婴产品需求分析专家。请根据用户的消息和宝宝档案信息，提取结构化的产品需求。

请分析以下信息并以 JSON 格式回复：
- baby_age_months: 宝宝当前月龄（整数）
- concern: 用户关心的核心问题，如"腹泻"、"湿疹"、"红屁股"、"便秘"、"胀气"等
- dietary_restrictions: 饮食限制列表（结合宝宝档案中已有的 + 消息中新提及的）
- product_category_needed: 你认为需要哪类产品（中文品类名，如"益生菌"、"护臀膏"、"湿疹膏"、"奶粉"、"辅食"等）
- urgency_note: 如果用户描述了紧急/严重症状则填写，否则为 null

注意：
1. 产品品类必须是母婴领域常见品类
2. dietary_restrictions 需要合并档案中已有的和新发现的两部分
3. 如果用户消息不足以判断，请给出最合理的推断
"""

_EXPLANATION_SYSTEM_PROMPT = """你是一个专业的母婴健康顾问。请根据宝宝的需求和推荐的产品列表，生成一段中文推荐说明。

要求：
1. 说明为什么推荐这些产品（结合宝宝的具体问题和月龄）
2. 简要说明每款产品的适用场景
3. 语气温暖专业，像母婴店导购在给顾客真诚建议
4. 不要编造产品名称或功效——只能使用提供的产品信息
5. 控制在 200-400 字以内
6. 末尾加上使用建议的免责提醒
"""


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _build_model_client(model_config: dict[str, Any]) -> Any:
    """构建 OpenAI SDK 兼容客户端。"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai 包: pip install openai")

    return OpenAI(
        api_key=model_config.get("api_key", ""),
        base_url=model_config.get("base_url", ""),
    )


async def _call_llm_json(
    system_prompt: str,
    user_prompt: str,
    model_config: dict[str, Any],
    temperature: float = 0.3,
) -> dict[str, Any]:
    """异步调用 LLM 并返回 JSON 解析结果。

    Args:
        system_prompt: 系统提示。
        user_prompt: 用户提示。
        model_config: 模型配置。
        temperature: 温度参数。

    Returns:
        解析后的 JSON 字典。

    Raises:
        ValueError: LLM 返回无效 JSON 或空内容。
    """
    client = _build_model_client(model_config)

    response = client.chat.completions.create(
        model=model_config.get("model_name", "deepseek-chat"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=model_config.get("max_tokens", 1000),
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("LLM 返回空内容")

    return json.loads(raw)


async def _call_llm_text(
    system_prompt: str,
    user_prompt: str,
    model_config: dict[str, Any],
    temperature: float = 0.7,
) -> str:
    """异步调用 LLM 并返回纯文本。

    Args:
        system_prompt: 系统提示。
        user_prompt: 用户提示。
        model_config: 模型配置。
        temperature: 温度参数。

    Returns:
        LLM 生成的文本。

    Raises:
        ValueError: LLM 返回空内容。
    """
    client = _build_model_client(model_config)

    response = client.chat.completions.create(
        model=model_config.get("model_name", "deepseek-chat"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=model_config.get("max_tokens", 800),
    )

    text = response.choices[0].message.content
    if not text:
        raise ValueError("LLM 返回空内容")

    return text


# ---------------------------------------------------------------------------
# 步骤 1: 需求提取（R2）
# ---------------------------------------------------------------------------


async def extract_baby_needs(
    baby_profile: dict[str, Any],
    employee_message: str,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """调用 LLM 从用户消息中提取结构化产品需求（R2）。

    提取内容包括：宝宝月龄、核心问题、饮食限制、所需品类、紧急备注。
    饮食限制会合并宝宝档案中已有的记录。

    Args:
        baby_profile: 宝宝档案字典，须包含 age_months, dietary_restrictions 等。
        employee_message: 员工消息文本。
        model_config: LLM 模型配置。

    Returns:
        结构化需求字典，包含:
          - baby_age_months: int
          - concern: str
          - dietary_restrictions: list[str]
          - product_category_needed: str
          - urgency_note: str | None
    """
    # 从档案中提取已有信息作为上下文
    age_months = baby_profile.get("age_months", 0)
    existing_restrictions = baby_profile.get("dietary_restrictions", [])
    allergies = baby_profile.get("allergies", [])
    allergy_names = [
        a.get("allergen", "") if isinstance(a, dict) else str(a)
        for a in (allergies or [])
    ]

    user_prompt = (
        f"宝宝月龄: {age_months}个月\n"
        f"已知饮食限制: {', '.join(existing_restrictions) if existing_restrictions else '无'}\n"
        f"已知过敏原: {', '.join(allergy_names) if allergy_names else '无'}\n"
        f"员工消息: {employee_message}\n"
    )

    try:
        parsed = await _call_llm_json(
            _NEED_EXTRACTION_SYSTEM_PROMPT,
            user_prompt,
            model_config,
            temperature=0.2,
        )

        # 合并饮食限制：档案已有 + LLM 新发现
        llm_restrictions = parsed.get("dietary_restrictions", [])
        if not isinstance(llm_restrictions, list):
            llm_restrictions = []
        merged_restrictions = list(set(existing_restrictions + llm_restrictions))

        needs = {
            "baby_age_months": int(parsed.get("baby_age_months", age_months)),
            "concern": parsed.get("concern", ""),
            "dietary_restrictions": merged_restrictions,
            "product_category_needed": parsed.get("product_category_needed", ""),
            "urgency_note": parsed.get("urgency_note") or None,
        }

        logger.info(
            "需求提取完成: concern=%s, category=%s, age=%d, restrictions=%s",
            needs["concern"], needs["product_category_needed"],
            needs["baby_age_months"], needs["dietary_restrictions"],
        )

        return needs

    except Exception as exc:
        logger.error("需求提取 LLM 调用失败: %s", exc)

        # 降级：从消息中做简单的关键词推断
        fallback_category = _keyword_category_guess(employee_message)
        return {
            "baby_age_months": age_months,
            "concern": employee_message[:50],
            "dietary_restrictions": existing_restrictions,
            "product_category_needed": fallback_category,
            "urgency_note": None,
        }


def _keyword_category_guess(message: str) -> str:
    """基于关键词的品类降级推测（LLM 不可用时使用）。"""
    keywords_map = {
        "红屁股": "护臀膏",
        "红屁屁": "护臀膏",
        "尿布疹": "护臀膏",
        "湿疹": "湿疹膏",
        "腹泻": "益生菌",
        "拉肚子": "益生菌",
        "便秘": "益生菌",
        "胀气": "益生菌",
        "奶粉": "奶粉",
        "辅食": "辅食",
        "米粉": "辅食",
        "奶瓶": "奶瓶",
        "尿布": "尿不湿",
        "尿不湿": "尿不湿",
        "湿巾": "湿巾",
        "沐浴": "洗护用品",
        "洗发": "洗护用品",
        "护肤": "护肤霜",
        "防晒": "防晒霜",
    }
    for keyword, category in keywords_map.items():
        if keyword in message:
            return category
    return "母婴用品"


# ---------------------------------------------------------------------------
# 步骤 2: 品类映射到 DB 分类（R4）
# ---------------------------------------------------------------------------


def _string_similarity(a: str, b: str) -> float:
    """计算两个字符串的相似度（0-1）。

    使用 SequenceMatcher 做字符级相似度计算。

    Args:
        a: 字符串 A。
        b: 字符串 B。

    Returns:
        相似度（0-1）。
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _substring_match(llm_category: str, db_category: str) -> bool:
    """检查 LLM 品类与 DB 分类是否存在包含关系。

    Args:
        llm_category: LLM 输出的品类名。
        db_category: DB 中的分类名。

    Returns:
        True 如果任一是另一的子串。
    """
    llm_lower = llm_category.lower()
    db_lower = db_category.lower()
    return llm_lower in db_lower or db_lower in llm_lower


def map_category_to_db(
    llm_category: str,
    db_categories: list[str],
) -> str:
    """将 LLM 输出的品类名映射到数据库中实际存在的分类（R4）。

    映射策略（优先级从高到低）：
      1. 精确匹配（忽略大小写）
      2. 子串包含匹配
      3. 相似度最高匹配（threshold = 0.3）

    均不匹配时返回第一条 DB 分类作为保底。

    Args:
        llm_category: LLM 输出的品类名称。
        db_categories: DB 中所有可用的分类名称列表。

    Returns:
        映射后的 DB 分类名称。
    """
    if not db_categories:
        logger.warning("DB 分类列表为空，无法进行品类映射")
        return llm_category

    if not llm_category:
        logger.warning("LLM 品类为空，返回第一条 DB 分类作为保底")
        return db_categories[0]

    # 1) 精确匹配
    llm_lower = llm_category.lower()
    for cat in db_categories:
        if cat.lower() == llm_lower:
            logger.info("品类精确匹配: '%s' → '%s'", llm_category, cat)
            return cat

    # 2) 子串匹配
    for cat in db_categories:
        if _substring_match(llm_category, cat):
            logger.info("品类子串匹配: '%s' → '%s'", llm_category, cat)
            return cat

    # 3) 相似度匹配
    best_score = 0.0
    best_cat = db_categories[0]
    for cat in db_categories:
        score = _string_similarity(llm_category, cat)
        if score > best_score:
            best_score = score
            best_cat = cat

    if best_score >= 0.3:
        logger.info("品类相似度匹配: '%s' → '%s' (score=%.3f)", llm_category, best_cat, best_score)
    else:
        logger.warning(
            "品类匹配置信度过低: '%s', best='%s' (score=%.3f)，使用保底分类",
            llm_category, best_cat, best_score,
        )

    return best_cat


# ---------------------------------------------------------------------------
# 步骤 3: DB 产品搜索（R3）— 纯数据库，无 LLM 参与
# ---------------------------------------------------------------------------


async def search_products_in_db(
    unified_store: Any,
    matched_category: str,
    baby_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """在数据库中按分类和宝宝月龄搜索产品（R3 纯 DB 操作）。

    此步骤绝不涉及 LLM，仅通过 unified_store 进行数据查询。
    使用向量搜索以提高匹配相关性，同时传递宝宝档案进行月龄过滤。

    Args:
        unified_store: UnifiedStore 实例，须提供 search_products() 方法。
        matched_category: 映射后的 DB 分类名称。
        baby_profile: 宝宝档案（用于月龄过滤）。

    Returns:
        匹配的产品字典列表。
    """
    try:
        # 优先使用向量搜索（按分类查询描述文本）
        query = f"{matched_category} {baby_profile.get('concern', '')}"
        products = unified_store.search_products(
            query=query,
            top_k=5,
            baby_profile=baby_profile,
        )

        if products:
            logger.info(
                "DB 产品搜索: category='%s', 找到 %d 个产品",
                matched_category, len(products),
            )
            return products

        # 向量搜索无结果时，回退到按分类名精确查询
        logger.info("向量搜索无结果，尝试分类精确查询: '%s'", matched_category)
        products = unified_store.get_products_by_category(category=matched_category)

        if products:
            # 手动过滤月龄
            age = baby_profile.get("age_months")
            if age is not None:
                filtered = [
                    p for p in products
                    if (
                        (p.get("suitable_age_min_months") is None or age >= p["suitable_age_min_months"])
                        and (p.get("suitable_age_max_months") is None or age <= p["suitable_age_max_months"])
                    )
                ]
                logger.info("分类精确查询+月龄过滤: %d → %d 个产品", len(products), len(filtered))
                return filtered[:5]
            return products[:5]

        logger.info("DB 产品搜索: category='%s', 无匹配产品", matched_category)
        return []

    except Exception as exc:
        logger.error("DB 产品搜索失败: %s", exc)
        return []


# ---------------------------------------------------------------------------
# 步骤 4: LLM 生成推荐理由
# ---------------------------------------------------------------------------


async def explain_recommendations(
    products: list[dict[str, Any]],
    baby_needs: dict[str, Any],
    model_config: dict[str, Any],
) -> str:
    """调用 LLM 生成推荐理由文本（仅解释，不创造产品名）。

    产品名称和数据全部来自 DB 查询结果，LLM 只负责用自然语言
    解释为什么这些产品适合该宝宝。

    Args:
        products: DB 查询返回的产品列表。
        baby_needs: 步骤 1 提取的结构化需求。
        model_config: LLM 模型配置。

    Returns:
        推荐理由文本。
    """
    if not products:
        return "未找到匹配的产品。"

    # 构建产品摘要（只传必要信息，避免 token 浪费）
    product_lines = []
    for i, p in enumerate(products, 1):
        name = p.get("name", "未知产品")
        desc = p.get("description", "")[:120]
        age_range = ""
        min_age = p.get("suitable_age_min_months")
        max_age = p.get("suitable_age_max_months")
        if min_age is not None or max_age is not None:
            age_range = f"，适用{min_age or 0}-{max_age or '以上'}个月"
        product_lines.append(f"{i}. {name}: {desc}{age_range}")

    product_summary = "\n".join(product_lines)

    user_prompt = (
        f"宝宝问题: {baby_needs.get('concern', '未知')}\n"
        f"宝宝月龄: {baby_needs.get('baby_age_months', 0)}个月\n"
        f"饮食限制: {', '.join(baby_needs.get('dietary_restrictions', [])) or '无'}\n"
        f"\n推荐产品:\n{product_summary}\n"
    )

    try:
        explanation = await _call_llm_text(
            _EXPLANATION_SYSTEM_PROMPT,
            user_prompt,
            model_config,
            temperature=0.6,
        )
        logger.info("推荐理由生成完成，长度: %d 字", len(explanation))
        return explanation

    except Exception as exc:
        logger.error("推荐理由生成 LLM 调用失败: %s", exc)
        # 降级：返回简单的模板化说明
        product_names = [p.get("name", "未知产品") for p in products[:3]]
        return (
            f"根据宝宝的月龄和需求，为您推荐以下产品：{'、'.join(product_names)}。"
            "请根据产品说明选择适合宝宝的款式。如有疑问欢迎继续咨询。"
        )


# ---------------------------------------------------------------------------
# 空结果处理（R40）
# ---------------------------------------------------------------------------


async def handle_empty_results(
    baby_needs: dict[str, Any],
    model_config: dict[str, Any],
) -> str:
    """处理产品搜索无结果的情况（R40）。

    返回提示信息说明暂无匹配产品，但可以从知识库提供健康/膳食建议。
    绝不虚构产品名称。

    Args:
        baby_needs: 提取的结构化需求。
        model_config: 模型配置（用于生成友好提示）。

    Returns:
        空结果提示文本。
    """
    concern = baby_needs.get("concern", "您关心的问题")
    category = baby_needs.get("product_category_needed", "相关产品")

    base_message = (
        f"关于「{concern}」，目前门店暂时没有匹配的{category}产品。\n\n"
        "不过，我可以为您提供关于这个问题的健康护理建议和膳食指导。"
        "您想了解更多关于「{concern}」的专业知识吗？"
    )

    # 如果模型可用，让 LLM 润色一下语气
    if model_config and model_config.get("api_key"):
        try:
            polish_prompt = """你是一个母婴健康助手。请将以下内容润色得更温暖、更有人情味。
保持原意不变，不要添加虚构的产品信息。直接返回润色后的文本。"""

            polished = await _call_llm_text(
                polish_prompt,
                base_message,
                model_config,
                temperature=0.5,
            )
            return polished.strip()
        except Exception as exc:
            logger.warning("空结果提示润色失败，使用模板: %s", exc)

    return base_message


# ---------------------------------------------------------------------------
# 主管线入口
# ---------------------------------------------------------------------------


async def recommend_products(
    input: ProductRecommendInput,
    unified_store: Any,
    model_config: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, Any]], str]:
    """产品推荐主管线 — 四步硬管线。

    严格按照 R2 → R4 → R3 → 解释 的顺序执行（I1 不变量保证）。
    所有推荐的产品名称必须来自数据库。

    Args:
        input: ProductRecommendInput，含宝宝档案、员工消息、模型配置。
        unified_store: UnifiedStore 实例。
        model_config: LLM 模型配置（可选，覆盖 input 中的配置）。

    Returns:
        (products_list, explanation_text) 元组。
        - products_list: 来自 DB 的产品列表
        - explanation_text: LLM 生成的推荐理由
    """
    cfg = model_config or input.llm_config

    logger.info("产品推荐管线启动: message='%s...'", input.employee_message[:60])

    # ---- 步骤 1: 需求提取（R2） ----
    baby_needs = await extract_baby_needs(
        input.baby_profile,
        input.employee_message,
        cfg,
    )

    # ---- 步骤 2: 品类映射（R4） ----
    # 获取 DB 中所有可用分类
    db_categories: list[str] = []
    try:
        if hasattr(unified_store, "conn") and unified_store.conn:
            rows = unified_store.conn.execute(
                "SELECT DISTINCT category FROM products WHERE category != '' ORDER BY category"
            ).fetchall()
            db_categories = [r["category"] for r in rows]
    except Exception as exc:
        logger.warning("获取 DB 分类失败: %s", exc)

    llm_category = baby_needs.get("product_category_needed", "")
    matched_category = map_category_to_db(llm_category, db_categories)

    # ---- 步骤 3: DB 产品搜索（R3） ----
    products = await search_products_in_db(
        unified_store,
        matched_category,
        input.baby_profile,
    )

    # 空结果处理（R40）
    if not products:
        logger.info("产品搜索无结果，返回空结果提示")
        empty_msg = await handle_empty_results(baby_needs, cfg)
        return [], empty_msg

    # ---- 步骤 4: LLM 生成推荐理由 ----
    explanation = await explain_recommendations(products, baby_needs, cfg)

    logger.info(
        "产品推荐管线完成: 推荐 %d 个产品, category='%s'",
        len(products), matched_category,
    )
    return products, explanation
