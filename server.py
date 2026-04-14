import asyncio
import json
import os
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from graph.graph1 import graph, make_config, GRAPH_NODES, GRAPH_EDGES
from llm_models.all_llm import llm
from graph.doc_processor import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="RAG Visual Console", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/health")
def health():
    return {
        "ok": True,
        "time": datetime.now().isoformat(timespec="seconds"),
        "graph_nodes": GRAPH_NODES,
        "graph_edges": GRAPH_EDGES,
    }


# =====================
# 新增：Word 文档处理接口
# =====================

@app.post("/upload-doc")
async def upload_doc(file: UploadFile = File(...)):
    """
    上传 Word 文档，解析占位符
    返回：文件ID、占位符列表
    """
    if not file.filename.endswith(('.docx', '.doc')):
        raise HTTPException(400, "只支持 .docx 或 .doc 文件")

    # 生成唯一文件ID
    file_id = str(uuid.uuid4())[:8]
    original_name = file.filename
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{original_name}")

    # 保存文件
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # 解析占位符
    try:
        placeholders = extract_placeholders(save_path)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(500, f"解析文档失败: {e}")

    # 获取每个占位符的上下文
    placeholder_info = []
    for p in placeholders:
        ctx = get_placeholder_context(save_path, p)
        placeholder_info.append({
            "name": p,
            "context": ctx[:200] if ctx else ""
        })

    return JSONResponse({
        "file_id": file_id,
        "filename": original_name,
        "placeholders": placeholder_info,
        "message": f"找到 {len(placeholders)} 个占位符"
    })


@app.post("/fill-doc")
async def fill_doc(
        file_id: str = Form(...),
        instruction: str = Form(""),
        use_rag: bool = Form(True)
):
    """
    填充文档占位符
    - file_id: 上传时返回的文件ID
    - instruction: 用户的填充要求
    - use_rag: 是否使用 RAG 检索知识库
    """
    # 找到上传的文件
    upload_files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(file_id)]
    if not upload_files:
        raise HTTPException(404, "文件不存在或已过期")

    upload_path = os.path.join(UPLOAD_DIR, upload_files[0])
    original_name = upload_files[0].split("_", 1)[1] if "_" in upload_files[0] else upload_files[0]

    # 解析占位符
    placeholders = extract_placeholders(upload_path)
    if not placeholders:
        raise HTTPException(400, "文档中没有找到占位符 {{xxx}}")

    # 为每个占位符生成内容
    replacements = {}

    for placeholder in placeholders:
        context = get_placeholder_context(upload_path, placeholder)

        if use_rag:
            # 使用 RAG 工作流获取相关知识
            rag_query = f"{placeholder} {instruction}" if instruction else placeholder

            try:
                # 简化调用：直接用 LLM + 检索结果
                cfg = make_config()
                inputs = {
                    "messages": [("user", rag_query)],
                    "rewrite_count": 0,
                    "step_count": 0,
                    "root_question": rag_query,
                }

                # 运行工作流获取上下文
                final_state = None
                for event in graph.stream(inputs, config=cfg, stream_mode="values"):
                    final_state = event

                # 从最终状态提取知识
                rag_context = ""
                if final_state and "messages" in final_state:
                    for msg in final_state["messages"]:
                        if hasattr(msg, "content") and msg.content:
                            rag_context += msg.content + "\n"

                # 构建带 RAG 上下文的 prompt
                prompt = f"""你需要为一个 Word 文档填充内容。

占位符名称: {placeholder}
占位符所在上下文: {context}
相关知识库内容:
{rag_context[:2000]}

用户要求: {instruction if instruction else "根据上下文合理填充"}

请生成合适的填充内容。只输出内容本身，不要解释。

填充内容："""

                response = llm.invoke(prompt)
                replacements[placeholder] = response.content.strip()

            except Exception as e:
                # RAG 失败，降级到纯 LLM
                prompt = build_fill_prompt(placeholder, context, instruction)
                response = llm.invoke(prompt)
                replacements[placeholder] = response.content.strip()
        else:
            # 纯 LLM 生成
            prompt = build_fill_prompt(placeholder, context, instruction)
            response = llm.invoke(prompt)
            replacements[placeholder] = response.content.strip()

    # 填充文档
    output_name = f"filled_{file_id}_{original_name}"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    success, message = fill_placeholders(upload_path, replacements, output_path)

    if not success:
        raise HTTPException(500, f"填充失败: {message}")

    return JSONResponse({
        "success": True,
        "download_url": f"/download/{output_name}",
        "filename": output_name,
        "filled_placeholders": list(replacements.keys()),
        "message": message
    })


@app.get("/download/{filename}")
async def download_file(filename: str):
    """下载填充后的文档"""
    file_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(404, "文件不存在")

    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename
    )


# =====================
# 原有 WebSocket 逻辑
# =====================

def _extract_role_content(m: Any) -> (str, str):
    """兼容多种 message 格式"""
    role = getattr(m, "type", None) or getattr(m, "role", None) or "assistant"
    content = getattr(m, "content", "") or ""

    if isinstance(m, (list, tuple)) and len(m) >= 2:
        role = str(m[0])
        content = str(m[1])

    if isinstance(m, dict):
        role = m.get("type") or m.get("role") or "assistant"
        content = m.get("content") or ""

    if isinstance(m, str):
        role = "assistant"
        content = m

    return role, content


def _truncate(text: str, max_len: int = 80) -> str:
    """截断文本用于日志显示"""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    session_thread_id = str(uuid.uuid4())

    async def send(payload: Dict[str, Any]):
        await ws.send_text(json.dumps(payload, ensure_ascii=False))

    async def log(message: str, level: str = "info"):
        """发送日志消息，level: info/ok/warn/err"""
        await send({"type": "log", "message": message, "level": level})

    # 初始化
    await send({
        "type": "state",
        "thread_id": session_thread_id,
        "graph_loaded": True,
        "graph_nodes": GRAPH_NODES,
        "graph_edges": GRAPH_EDGES,
        "ts": datetime.now().isoformat(timespec="seconds"),
    })

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await send({"type": "error", "message": "Invalid JSON"})
                continue

            mtype = msg.get("type")

            if mtype == "init":
                tid = (msg.get("thread_id") or "").strip()
                if tid:
                    session_thread_id = tid
                await send({
                    "type": "state",
                    "thread_id": session_thread_id,
                    "graph_loaded": True,
                    "graph_nodes": GRAPH_NODES,
                    "graph_edges": GRAPH_EDGES,
                    "ts": datetime.now().isoformat(timespec="seconds"),
                })
                continue

            if mtype != "user_message":
                await send({"type": "error", "message": f"Unknown type: {mtype}"})
                continue

            text = (msg.get("text") or "").strip()
            if not text:
                await send({"type": "error", "message": "Empty message"})
                continue

            tid = (msg.get("thread_id") or session_thread_id).strip()
            if tid:
                session_thread_id = tid

            # 用户消息回显
            await send({
                "type": "message",
                "role": "user",
                "content": text,
                "ts": datetime.now().isoformat(timespec="seconds"),
            })

            # ========== 增强日志 ==========
            await log(f"📩 收到用户问题: {_truncate(text, 50)}")
            await log(f"🔗 thread_id: {session_thread_id[:8]}...")
            await log("🚀 开始执行工作流...")

            cfg = make_config(session_thread_id)
            inputs = {
                "messages": [("user", text)],
                "rewrite_count": 0,
                "step_count": 0,
                "root_question": text,
            }

            # -----------------------------
            # 流式执行 + 逐token推送
            # -----------------------------
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            # 统计信息
            start_time = datetime.now()
            token_count = 0

            def _producer():
                from graph.generate_node import set_stream_callback

                def token_callback(token: str):
                    asyncio.run_coroutine_threadsafe(
                        queue.put({"_type": "token", "token": token}),
                        loop
                    )

                set_stream_callback(token_callback)

                try:
                    for ev in graph.stream(inputs, config=cfg, stream_mode="values"):
                        asyncio.run_coroutine_threadsafe(queue.put(ev), loop)
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(queue.put(e), loop)
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                finally:
                    set_stream_callback(None)

            loop.run_in_executor(None, _producer)

            last_messages: Optional[list] = None
            streaming_started = False
            streamed_content = ""
            last_node = None
            node_start_time = None
            is_first_event = True  # 第一个event是历史state快照，current_node是残留

            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item

                    # -----------------------------
                    # 处理逐token流式输出
                    # -----------------------------
                    if isinstance(item, dict) and item.get("_type") == "token":
                        token = item.get("token", "")
                        if token:
                            if not streaming_started:
                                streaming_started = True
                                # 关键：流式生成开始时，立即通知前端切换到 generate 节点
                                # 因为 generate 节点的 current_node 要等整个函数返回后才会出现在 event 中
                                # 但 token 已经在推了，所以需要提前通知
                                if last_node != "generate":
                                    if last_node and node_start_time:
                                        elapsed = (datetime.now() - node_start_time).total_seconds()
                                        await log(f"   └─ {last_node} 完成 ({elapsed:.2f}s)")
                                    last_node = "generate"
                                    node_start_time = datetime.now()
                                    await send({"type": "node", "name": "generate"})
                                    await log("💬 进入节点: generate", "warn")

                                await send({
                                    "type": "stream_start",
                                    "role": "assistant",
                                    "ts": datetime.now().isoformat(timespec="seconds"),
                                })
                                await log("✍️ 开始生成回答...", "ok")

                            streamed_content += token
                            token_count += 1
                            await send({
                                "type": "stream_token",
                                "token": token,
                            })
                        continue

                    ev = item
                    if not isinstance(ev, dict):
                        continue

                    # ========== 节点日志增强 ==========
                    node = ev.get("current_node")

                    # 跳过第一个event的残留current_node
                    if is_first_event:
                        is_first_event = False
                        node = None

                    if node and node != last_node:
                        # 记录上一个节点的耗时
                        if last_node and node_start_time:
                            elapsed = (datetime.now() - node_start_time).total_seconds()
                            await log(f"   └─ {last_node} 完成 ({elapsed:.2f}s)")

                        # 记录新节点
                        node_start_time = datetime.now()
                        last_node = node

                        await send({"type": "node", "name": node})

                        # 节点特定日志
                        node_emoji = {
                            "agent": "🤖",
                            "retrieve": "🔍",
                            "rewrite": "✏️",
                            "web_search": "🌐",
                            "generate": "💬",
                        }
                        emoji = node_emoji.get(node, "📍")
                        await log(f"{emoji} 进入节点: {node}", "warn")

                    # ========== 状态详细日志 ==========
                    # 重写次数
                    rewrite_count = ev.get("rewrite_count")
                    if rewrite_count is not None and rewrite_count > 0:
                        await log(f"   🔄 重写次数: {rewrite_count}/2")

                    # 步数
                    step_count = ev.get("step_count")
                    if step_count is not None and step_count > 0 and step_count % 3 == 0:
                        await log(f"   📊 当前步数: {step_count}")

                    # force_web_search
                    if ev.get("force_web_search"):
                        await log("   ⚠️ 触发强制联网搜索", "warn")

                    # messages 增量推送
                    if "messages" in ev and isinstance(ev.get("messages"), list):
                        msgs = ev["messages"]
                        msg_count = len(msgs)

                        if last_messages is None:
                            # 首次：记录基线，不推送（可能包含历史消息）
                            last_messages = msgs
                            if msg_count > 0:
                                await log(f"   📝 消息队列初始化: {msg_count} 条")
                        else:
                            new_part = msgs[len(last_messages):] if len(msgs) >= len(last_messages) else msgs
                            last_messages = msgs

                            for m in new_part:
                                role, content = _extract_role_content(m)

                                # 跳过空内容
                                if not content or not content.strip():
                                    continue

                                # 跳过 agent 节点产生的 AI 回复
                                # agent 的职责是"决策是否调用工具"，不是回答用户
                                # 最终回答统一由 generate 节点通过流式输出
                                if role in ("ai", "assistant") and not streaming_started:
                                    # 还没开始流式输出，说明当前还在 agent/retrieve 阶段
                                    # agent 的 AI 回复不应该展示给用户
                                    if last_node in (None, "agent", "START"):
                                        await log(f"   ⏭️ 跳过 agent 决策回复 (不展示)")
                                        continue

                                # 跳过已流式输出的内容
                                if streaming_started and role in ("assistant", "ai") and content == streamed_content:
                                    continue

                                # 日志记录
                                if role == "tool":
                                    preview = _truncate(content, 60)
                                    await log(f"   📦 检索结果: {preview}")
                                elif "联网搜索" in content:
                                    preview = _truncate(content, 60)
                                    await log(f"   🌐 联网结果: {preview}")

                                await send({
                                    "type": "message",
                                    "role": role,
                                    "content": content,
                                    "ts": datetime.now().isoformat(timespec="seconds"),
                                })

                    # 状态快照
                    await send({
                        "type": "state",
                        "thread_id": session_thread_id,
                        "graph_loaded": True,
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "state_keys": list(ev.keys()),
                        "messages_count": len(ev.get("messages", [])) if isinstance(ev.get("messages", None),
                                                                                    list) else None,
                    })

                # ========== 结束统计 ==========
                total_time = (datetime.now() - start_time).total_seconds()

                if streaming_started:
                    await send({
                        "type": "stream_end",
                        "ts": datetime.now().isoformat(timespec="seconds"),
                    })
                    await log(f"✅ 生成完成: {token_count} tokens", "ok")

                # 最后一个节点耗时
                if last_node and node_start_time:
                    elapsed = (datetime.now() - node_start_time).total_seconds()
                    await log(f"   └─ {last_node} 完成 ({elapsed:.2f}s)")

                await log(f"🏁 工作流完成 (总耗时: {total_time:.2f}s)", "ok")
                await send({"type": "done"})

            except Exception as e:
                await log(f"❌ 执行失败: {type(e).__name__}: {e}", "err")
                await send({
                    "type": "error",
                    "message": f"Execution failed: {type(e).__name__}: {e}",
                    "trace": traceback.format_exc(),
                })
                await send({"type": "done"})

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await send({
                "type": "error",
                "message": f"WS server error: {type(e).__name__}: {e}",
                "trace": traceback.format_exc(),
            })
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )