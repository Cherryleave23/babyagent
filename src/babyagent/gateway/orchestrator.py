"""主消息处理编排器 — 全管线串联

将微信消息处理的全链路编织为一条异步流水线：

    接收消息 → 会话隔离 → 指令处理 → 宝宝切换 → 意图分类
    → 路由分发 → 生成回复 → 历史管理 → 压缩检查 → 返回

R7/R8: @Agent 指令为纯规则驱动，不调用 LLM。
R20-R23: 隐式+显式宝宝切换检测。
R30-R36: 上下文压缩（含档案更新提取）。
R37/R38: 首次接触宝宝时注入档案摘要。
R39: 压缩优先级：档案更新先于会话摘要。
R40: 产品搜索空结果时的温和降级回复。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any, Optional

from babyagent.config.loader import AppConfig
from babyagent.core.baby.compression import (
    BabyProfileUpdate,
    CompressionOutput,
    generate_compression,
    build_empty_output,
)
from babyagent.core.baby.profile import BabyProfileManager
from babyagent.core.db.unified_store import UnifiedStore
from babyagent.core.pipeline.health_diet import consult_health_diet
from babyagent.core.pipeline.intent import (
    IntentResult,
    MessageIntent,
    classify_intent,
)
from babyagent.core.pipeline.product_recommend import (
    ProductRecommendInput,
    recommend_products,
)
from babyagent.core.pipeline.rejection import (
    generate_rejection,
    is_emergency_situation,
    get_rejection_message_by_type,
)
from babyagent.gateway.baby_switch import detect_baby_switch
from babyagent.gateway.command_parser import parse_command, CommandResult
from babyagent.gateway.session_store import SessionStore, SessionContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 压缩触发阈值：历史token数达到模型上下文窗口的该比例时触发
_COMPRESSION_THRESHOLD_RATIO = 0.75

# 估算 token 数的粗略系数（中文字符约 1.5 token/字）
_TOKEN_ESTIMATE_RATIO = 1.5


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _config_to_model_dict(config: AppConfig) -> dict[str, Any]:
    """将 AppConfig 转换为管线模块需要的模型配置字典。

    Args:
        config: AppConfig 实例。

    Returns:
        包含 api_key, base_url, model_name, provider, max_tokens, temperature 的字典。
    """
    return {
        "provider": config.model.provider,
        "model_name": config.model.model_name,
        "api_key": config.model.api_key,
        "base_url": config.model.base_url,
        "max_tokens": config.model.max_tokens,
        "temperature": config.model.temperature,
    }


def _config_to_aux_model_dict(config: AppConfig) -> dict[str, Any]:
    """将 AppConfig 转换为辅助模型配置字典。

    Args:
        config: AppConfig 实例。

    Returns:
        辅助模型配置字典。
    """
    return {
        "provider": config.aux_model.provider,
        "model_name": config.aux_model.model_name,
        "api_key": config.aux_model.api_key,
        "base_url": config.aux_model.base_url,
    }


def _estimate_tokens(text: str) -> int:
    """粗略估算文本的 token 数。

    中文按 1.5 token/字估算，英文按 0.25 token/字。

    Args:
        text: 输入文本。

    Returns:
        估算 token 数。
    """
    if not text:
        return 0
    return int(len(text) * _TOKEN_ESTIMATE_RATIO)


def _estimate_history_tokens(history: list[tuple[str, str]]) -> int:
    """估算对话历史的 token 总数。

    Args:
        history: 对话历史列表。

    Returns:
        估算 token 数。
    """
    total = 0
    for role, content in history:
        total += _estimate_tokens(content) + 10  # 10 tokens for role overhead
    return total


def _session_history_to_text(history: list[tuple[str, str]]) -> str:
    """将对话历史转换为压缩输入格式的文本。

    Args:
        history: [(role, content), ...] 列表。

    Returns:
        格式化的对话文本。
    """
    lines = []
    for role, content in history:
        role_cn = "用户" if role == "user" else "助手"
        lines.append(f"[{role_cn}] {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 指令处理（R7/R8）
# ---------------------------------------------------------------------------


async def _handle_command(
    command: CommandResult,
    session: SessionContext,
    profile_manager: BabyProfileManager,
    unified_store: UnifiedStore,
    session_store: SessionStore,
    config: AppConfig,
) -> str:
    """处理 @Agent 结构化指令。

    所有指令均为规则驱动（R7/R8），不调用 LLM。

    Args:
        command: 解析后的指令结果。
        session: 当前会话上下文。
        profile_manager: 档案管理器。
        unified_store: 统一存储。
        session_store: 会话存储。
        config: 应用配置。

    Returns:
        指令执行的确认/错误消息。
    """
    store_id = session.store_id or config.enterprise.store_id

    if command.cmd == "switch_baby":
        # 显式切换宝宝（R21）
        name = command.args.get("name", "")
        if not name:
            return command.error_message or "请指定要切换的宝宝名字。"

        # 复用 detect_baby_switch 的处理逻辑
        result = detect_baby_switch(
            message=f"@切换宝宝 {name}",
            current_baby_name=session.active_baby_name,
            profile_manager=profile_manager,
            store_id=store_id,
        )

        if result["switched"] and result["new_baby_id"] is not None:
            session_store.set_active_baby(
                session.session_key,
                result["new_baby_id"],
                result["new_baby_name"] or name,
            )
            return result["confirmation_message"] or f"已切换到「{name}」的档案。"

        if result["conflict_babies"]:
            return result["confirmation_message"] or "存在重名冲突，请确认。"

        return result["confirmation_message"] or f"未找到「{name}」的档案。"

    elif command.cmd == "create_profile":
        # R7: 建档指令
        args = command.args
        try:
            profile = profile_manager.create_profile(
                store_id=store_id,
                name=args["name"],
                gender=args["gender"],
                birth_date=args["birth_date"],
                notes_free_text=args.get("notes_free_text", ""),
            )
            return (
                f"宝宝档案创建成功！\n"
                f"姓名：{profile.name}\n"
                f"性别：{'男' if profile.gender == 'male' else '女'}\n"
                f"出生日期：{profile.birth_date}\n"
                f"月龄：{profile.age_months}个月\n\n"
                f"现在您可以使用 @切换宝宝 {profile.name} 来切换到该宝宝的档案。"
            )
        except Exception as exc:
            logger.error("建档失败: %s", exc)
            return f"建档失败：{exc}\n请检查指令格式：@Agent 建档 {{姓名}} {{性别(男/女)}} {{出生日期(YYYY-MM-DD)}}"

    elif command.cmd == "add_allergy":
        # R7: 过敏添加指令
        baby_name = command.args.get("name", "")
        allergen = command.args.get("allergen", "")

        if not baby_name:
            return "请指定宝宝名字，例如：@Agent 小明 过敏史添加 牛奶蛋白"

        # 按名查找宝宝
        candidates = profile_manager.resolve_baby_by_name(baby_name, store_id)
        if not candidates:
            return f"未找到名为「{baby_name}」的宝宝档案。请先用 @Agent 建档 创建。"

        if len(candidates) > 1:
            return f"找到 {len(candidates)} 个叫「{baby_name}」的宝宝，请确认后重试。"

        baby = candidates[0]
        try:
            profile_manager.update_allergy(
                baby_id=baby.id,
                allergen=allergen,
                operation="append",
                source="human",
                session_id=session.session_key,
            )
            return f"已为「{baby_name}」记录过敏原：{allergen}。"
        except Exception as exc:
            logger.error("过敏添加失败: %s", exc)
            return f"过敏记录添加失败：{exc}"

    elif command.cmd == "delete_allergy":
        # R7: 过敏删除指令
        baby_name = command.args.get("name", "")
        allergen = command.args.get("allergen", "")

        if not baby_name:
            return "请指定宝宝名字，例如：@Agent 小明 过敏史删除 牛奶蛋白"

        candidates = profile_manager.resolve_baby_by_name(baby_name, store_id)
        if not candidates:
            return f"未找到名为「{baby_name}」的宝宝档案。"

        if len(candidates) > 1:
            return f"找到 {len(candidates)} 个叫「{baby_name}」的宝宝，请确认后重试。"

        baby = candidates[0]
        try:
            profile_manager.update_allergy(
                baby_id=baby.id,
                allergen=allergen,
                operation="delete",
                source="human",
                session_id=session.session_key,
            )
            return f"已从「{baby_name}」移除过敏原：{allergen}。"
        except Exception as exc:
            logger.error("过敏删除失败: %s", exc)
            return f"过敏删除失败：{exc}"

    elif command.cmd == "add_growth":
        # R7: 生长记录指令
        baby_name = command.args.get("name", "")
        record_date = command.args.get("record_date", "")
        height_cm = command.args.get("height_cm")
        weight_kg = command.args.get("weight_kg")

        if not baby_name:
            return "请指定宝宝名字，例如：@Agent 小明 生长记录 2024-07-15 65cm 7.5kg"

        candidates = profile_manager.resolve_baby_by_name(baby_name, store_id)
        if not candidates:
            return f"未找到名为「{baby_name}」的宝宝档案。"

        if len(candidates) > 1:
            return f"找到 {len(candidates)} 个叫「{baby_name}」的宝宝，请确认后重试。"

        baby = candidates[0]
        try:
            profile_manager.add_growth_record(
                baby_id=baby.id,
                record_date=record_date,
                height_cm=height_cm,
                weight_kg=weight_kg,
            )

            parts = [f"已为「{baby_name}」记录生长数据（{record_date}）："]
            if height_cm is not None:
                parts.append(f"身高：{height_cm}cm")
            if weight_kg is not None:
                parts.append(f"体重：{weight_kg}kg")
            return "\n".join(parts)
        except Exception as exc:
            logger.error("生长记录添加失败: %s", exc)
            return f"生长记录添加失败：{exc}"

    elif command.cmd == "unknown":
        return command.error_message or "无法识别的 @Agent 指令。"

    return "未知指令类型。"


# ---------------------------------------------------------------------------
# 上下文压缩（R30-R36, R39）
# ---------------------------------------------------------------------------


async def _compress_if_needed(
    session: SessionContext,
    session_store: SessionStore,
    profile_manager: BabyProfileManager,
    config: AppConfig,
) -> None:
    """检查是否需要压缩，若需要则执行上下文压缩。

    R39 压缩顺序：
      1. 提取档案更新（baby_profile_updates）
      2. 高置信度更新自动合入档案（merge_baby_updates）
      3. 生成 session_summary 替代原始历史

    R30a: session_summary 仅留在当前 session 中，不对外暴露。

    Args:
        session: 当前会话上下文。
        session_store: 会话存储。
        profile_manager: 档案管理器。
        config: 应用配置。
    """
    history = session.conversation_history
    if not history:
        return

    estimated_tokens = _estimate_history_tokens(history)
    threshold_tokens = int(config.compression.max_context_tokens * _COMPRESSION_THRESHOLD_RATIO)

    if estimated_tokens < threshold_tokens:
        logger.debug("历史 token=%d < 阈值=%d，跳过压缩", estimated_tokens, threshold_tokens)
        return

    logger.info(
        "触发上下文压缩: token=%d >= 阈值=%d, history_rounds=%d",
        estimated_tokens, threshold_tokens, len(history),
    )

    # ---- 获取宝宝档案快照 ----
    profile_snapshot = ""
    if session.active_baby_id is not None:
        try:
            baby = profile_manager.get_profile(str(session.active_baby_id))
            if baby:
                profile_snapshot = baby.to_context_summary()
        except Exception as exc:
            logger.warning("获取宝宝档案快照失败: %s", exc)

    # ---- 构建对话历史文本 ----
    session_text = _session_history_to_text(history)

    # ---- 调用压缩（R31） ----
    try:
        aux_config = _config_to_aux_model_dict(config)
        compression_output: CompressionOutput = generate_compression(
            session_history=session_text,
            current_baby_profile_snapshot=profile_snapshot,
            aux_model_config=aux_config,
        )
    except Exception as exc:
        logger.error("压缩生成失败: %s", exc)
        compression_output = build_empty_output()
        compression_output.session_summary = f"[压缩失败] {session_text[:300]}..."

    # ---- R39 步骤 1: 档案更新优先 ----
    actionable_updates = compression_output.actionable_updates()
    if actionable_updates and session.active_baby_id is not None:
        try:
            updates_for_merge = [u.model_dump() for u in actionable_updates]
            result = profile_manager.merge_baby_updates(
                baby_id=str(session.active_baby_id),
                updates=updates_for_merge,
            )
            logger.info(
                "档案更新合入完成: applied=%d, pending=%d",
                len(result["applied"]), len(result["pending"]),
            )
        except Exception as exc:
            logger.error("档案更新合入失败: %s", exc)

    # ---- R39 步骤 2: 用 session_summary 替换历史（R30a） ----
    new_summary = compression_output.session_summary
    if new_summary:
        session_store.replace_history(
            session.session_key,
            [("system", f"[会话摘要] {new_summary}")],
        )
        session.conversation_history = [("system", f"[会话摘要] {new_summary}")]
        logger.info("上下文压缩完成: session=%s, summary_len=%d", session.session_key, len(new_summary))


# ---------------------------------------------------------------------------
# 主编排器
# ---------------------------------------------------------------------------


async def process_message(
    employee_wxid: str,
    message_text: str,
    context_token: str,
    unified_store: UnifiedStore,
    profile_manager: BabyProfileManager,
    session_store: SessionStore,
    config: AppConfig,
) -> str:
    """处理单条微信消息的完整管线。

    这是 BabyAgent 消息处理的最高层入口，串联了从会话分配到回复生成的
    全部步骤。遵循 SPEC 中定义的管道顺序，并确保 R37-R40 的边界条件处理。

    管线流程：
      1. 获取/创建 Session → 按 employee_wxid 隔离
      2. 更新 context_token → 微信回复关联
      3. 指令检查 → @Agent 规则驱动指令（R7/R8）
      4. 宝宝切换检测 → 隐式/显式检测（R20-R23）
      5. 紧急症状检测 → 直接返回就医转介（R6）
      6. 意图分类 → LLM 分类（R1）
      7. 路由分发 → product/health_diet/reject
      8. 首次宝宝上下文注入 → 切换后首条消息注入摘要（R37/R38）
      9. 生成回复 → 调用下游管线
      10. 追加历史 → 记录 (user, assistant) 轮次
      11. 压缩检查 → 若超出阈值则压缩（R30-R36, R39）
      12. 返回回复

    Args:
        employee_wxid: 微信用户 ID。
        message_text: 消息文本。
        context_token: 微信上下文令牌。
        unified_store: 统一存储实例。
        profile_manager: 宝宝档案管理器。
        session_store: 会话存储。
        config: 应用配置。

    Returns:
        完成的中文回复文本。
    """
    response = ""
    model_config = _config_to_model_dict(config)
    store_id = config.enterprise.store_id

    # ========================================================================
    # 步骤 1: 获取/创建 Session（R41）
    # ========================================================================
    session = session_store.get_or_create_session(
        employee_wxid=employee_wxid,
        store_id=store_id,
    )

    # ========================================================================
    # 步骤 2: 更新 context_token
    # ========================================================================
    session_store.update_token(session.session_key, context_token)

    try:
        # ====================================================================
        # 步骤 3: @Agent 指令检查（R7/R8）
        # ====================================================================
        command = parse_command(message_text)

        if command.cmd != "unknown":
            # 结构化指令 → 规则驱动处理，不调 LLM
            logger.info("检测到 @Agent 指令: cmd=%s", command.cmd)
            response = await _handle_command(
                command=command,
                session=session,
                profile_manager=profile_manager,
                unified_store=unified_store,
                session_store=session_store,
                config=config,
            )
            # 指令消息也不计入对话历史
            return response

        # ====================================================================
        # 步骤 4: 宝宝切换检测（R20-R23）
        # ====================================================================
        switch_result = detect_baby_switch(
            message=message_text,
            current_baby_name=session.active_baby_name,
            profile_manager=profile_manager,
            store_id=store_id,
        )

        baby_switch_confirmation: Optional[str] = None

        if switch_result["switched"] and switch_result["new_baby_id"] is not None:
            session_store.set_active_baby(
                session.session_key,
                switch_result["new_baby_id"],
                switch_result["new_baby_name"] or "",
            )
            session.active_baby_id = switch_result["new_baby_id"]
            session.active_baby_name = switch_result["new_baby_name"]

            if switch_result["confirmation_message"]:
                baby_switch_confirmation = switch_result["confirmation_message"]

        elif switch_result["conflict_babies"]:
            # R23: 多匹配冲突 → 返回候选列表
            response = switch_result["confirmation_message"] or "存在重名冲突。"
            return response

        elif switch_result["confirmation_message"] and not switch_result["switched"]:
            # 切换失败的错误提示（如"未找到档案"）
            response = switch_result["confirmation_message"]
            return response

        # ====================================================================
        # 步骤 5: 紧急症状检测（R6）— 在意图分类前拦截
        # ====================================================================
        if is_emergency_situation(message_text):
            response = generate_rejection(message_text=message_text)
            # 紧急症状不计入对话历史
            if baby_switch_confirmation:
                response = baby_switch_confirmation + "\n\n" + response
            return response

        # ====================================================================
        # 步骤 6: 意图分类（R1）
        # ====================================================================
        intent_result: IntentResult = await classify_intent(
            message_text=message_text,
            active_baby_name=session.active_baby_name,
            model_config=model_config,
        )

        # 如果 LLM 提取了宝宝名，且当前无活跃宝宝，尝试切换
        if (
            intent_result.baby_name_hint
            and session.active_baby_id is None
            and not switch_result["switched"]
        ):
            implicit_switch = detect_baby_switch(
                message=message_text,
                current_baby_name=None,
                profile_manager=profile_manager,
                store_id=store_id,
            )
            if implicit_switch["switched"] and implicit_switch["new_baby_id"] is not None:
                session_store.set_active_baby(
                    session.session_key,
                    implicit_switch["new_baby_id"],
                    implicit_switch["new_baby_name"] or "",
                )
                session.active_baby_id = implicit_switch["new_baby_id"]
                session.active_baby_name = implicit_switch["new_baby_name"]
                if implicit_switch["confirmation_message"]:
                    baby_switch_confirmation = implicit_switch["confirmation_message"]

        # ====================================================================
        # 步骤 7: 路由分发
        # ====================================================================
        if intent_result.intent == MessageIntent.OUT_OF_SCOPE:
            response = generate_rejection(intent_result=intent_result, message_text=message_text)

        elif intent_result.intent == MessageIntent.PRODUCT_RECOMMEND:
            # ---- 产品推荐管线（R2-R4） ----
            response = await _run_product_pipeline(
                session=session,
                message_text=message_text,
                profile_manager=profile_manager,
                unified_store=unified_store,
                model_config=model_config,
                session_store=session_store,
            )

        else:
            # ---- 健康膳食管线 ----
            response = await _run_health_pipeline(
                session=session,
                message_text=message_text,
                profile_manager=profile_manager,
                unified_store=unified_store,
                model_config=model_config,
                session_store=session_store,
            )

        # ====================================================================
        # 步骤 8: 追加切换确认消息
        # ====================================================================
        if baby_switch_confirmation:
            response = baby_switch_confirmation + "\n\n" + response

        # ====================================================================
        # 步骤 9: 追加到对话历史
        # ====================================================================
        session_store.append_to_history(session.session_key, "user", message_text)
        session_store.append_to_history(session.session_key, "assistant", response)
        session.conversation_history.append(("user", message_text))
        session.conversation_history.append(("assistant", response))

        # ====================================================================
        # 步骤 10: 上下文压缩检查（R30-R36, R39）
        # ====================================================================
        await _compress_if_needed(
            session=session,
            session_store=session_store,
            profile_manager=profile_manager,
            config=config,
        )

        return response

    except Exception as exc:
        # ====================================================================
        # 全局降级：任何步骤失败时返回安全回复
        # ====================================================================
        logger.exception("消息处理异常: employee=%s, error=%s", employee_wxid, exc)

        # 尽力保存用户消息到历史，避免丢失上下文
        try:
            session_store.append_to_history(session.session_key, "user", message_text)
        except Exception:
            pass

        fallback = (
            "抱歉，系统当前遇到了一些问题，暂时无法处理您的请求。\n"
            "请稍后重试，或直接咨询门店营业员获取帮助。"
        )

        if baby_switch_confirmation:
            fallback = baby_switch_confirmation + "\n\n" + fallback

        return fallback


# ---------------------------------------------------------------------------
# 产品推荐子管线
# ---------------------------------------------------------------------------


async def _run_product_pipeline(
    session: SessionContext,
    message_text: str,
    profile_manager: BabyProfileManager,
    unified_store: UnifiedStore,
    model_config: dict[str, Any],
    session_store: SessionStore,
) -> str:
    """执行产品推荐子管线。

    Args:
        session: 会话上下文。
        message_text: 用户消息。
        profile_manager: 档案管理器。
        unified_store: 统一存储。
        model_config: 模型配置。
        session_store: 会话存储。

    Returns:
        推荐回复文本。
    """
    # 构建宝宝档案字典（兼容管线接口）
    baby_profile_dict: dict[str, Any] = {}
    baby_context_prefix = ""

    if session.active_baby_id is not None:
        try:
            baby = profile_manager.get_profile(str(session.active_baby_id))
            if baby:
                baby_profile_dict = _baby_profile_to_pipeline_dict(baby)

                # R37/R38: 首次接触此宝宝时注入档案摘要
                if session_store.is_first_baby_message(session.session_key):
                    baby_context_prefix = f"\U0001f4cb 当前档案：{baby.to_context_summary()}\n\n"
                    session_store.mark_first_baby_message_sent(session.session_key)
        except Exception as exc:
            logger.warning("获取宝宝档案失败: %s", exc)

    input_data = ProductRecommendInput(
        baby_profile=baby_profile_dict,
        employee_message=message_text,
        model_config=model_config,
    )

    products, explanation = await recommend_products(
        input=input_data,
        unified_store=unified_store,
        model_config=model_config,
    )

    if products:
        # 有产品结果 → 格式化推荐
        product_lines = []
        for i, p in enumerate(products, 1):
            name = p.get("name", "未知产品")
            desc = p.get("description", "")[:100]
            product_lines.append(f"{i}. {name}：{desc}")
        product_text = "\n".join(product_lines)
        response = f"{explanation}\n\n推荐产品：\n{product_text}"
    else:
        # R40: 空结果 → 产品搜索返回空，提供健康建议降级
        response = explanation  # handle_empty_results 的返回值已经在 explanation 中

    if baby_context_prefix:
        response = baby_context_prefix + response

    return response


# ---------------------------------------------------------------------------
# 健康膳食子管线
# ---------------------------------------------------------------------------


async def _run_health_pipeline(
    session: SessionContext,
    message_text: str,
    profile_manager: BabyProfileManager,
    unified_store: UnifiedStore,
    model_config: dict[str, Any],
    session_store: SessionStore,
) -> str:
    """执行健康膳食咨询子管线。

    Args:
        session: 会话上下文。
        message_text: 用户消息。
        profile_manager: 档案管理器。
        unified_store: 统一存储。
        model_config: 模型配置。
        session_store: 会话存储。

    Returns:
        健康建议回复文本。
    """
    # 构建宝宝档案字典
    baby_profile_dict: dict[str, Any] = {}
    baby_context_prefix = ""

    if session.active_baby_id is not None:
        try:
            baby = profile_manager.get_profile(str(session.active_baby_id))
            if baby:
                baby_profile_dict = _baby_profile_to_pipeline_dict(baby)

                # R37/R38: 首次接触此宝宝时注入档案摘要
                if session_store.is_first_baby_message(session.session_key):
                    baby_context_prefix = f"\U0001f4cb 当前档案：{baby.to_context_summary()}\n\n"
                    session_store.mark_first_baby_message_sent(session.session_key)
        except Exception as exc:
            logger.warning("获取宝宝档案失败: %s", exc)

    response = await consult_health_diet(
        employee_message=message_text,
        baby_profile=baby_profile_dict,
        unified_store=unified_store,
        model_config=model_config,
    )

    if baby_context_prefix:
        response = baby_context_prefix + response

    return response


# ---------------------------------------------------------------------------
# 辅助：BabyProfile → 管线字典
# ---------------------------------------------------------------------------


def _baby_profile_to_pipeline_dict(baby: Any) -> dict[str, Any]:
    """将 BabyProfile Pydantic 模型转换为管线兼容的字典。

    兼容 consult_health_diet 和 recommend_products 等管线函数期望的
    字典格式（含 age_months, dietary_restrictions, allergies 等字段）。

    Args:
        baby: BabyProfile 实例。

    Returns:
        管线兼容的字典。
    """
    if hasattr(baby, "model_dump"):
        return baby.model_dump()
    elif hasattr(baby, "dict"):
        return baby.dict()
    else:
        return dict(baby) if isinstance(baby, dict) else {}
