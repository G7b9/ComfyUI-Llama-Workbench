import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CHAT_NODE = "LlamaWorkbench_Chat";
const HIDDEN_WIDGETS = [
    "lwb_history_json",
    "lwb_flow_state_json",
    "lwb_request_id",
    // Kept in the backend schema so existing workflow JSON retains its
    // values, but Chat Settings is the single visible owner of these three
    // generation controls.
    "max_tokens",
    "seed",
    "use_cache",
];
const MIN_WIDTH = 500;
const MIN_HEIGHT = 640;
const CHAT_LAYOUT_VERSION = 7;
const PANEL_LEFT = 10;
const PANEL_TOP = 108;
const PANEL_BOTTOM_GAP = 10;
const LINE_HEIGHT = 17;
const PANEL_HEADER_HEIGHT = 46;
const PANEL_MIN_HEIGHT = 190;
const MESSAGE_RIGHT_GUTTER = 18;

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function clearsContextBeforeRun(node) {
    return Boolean(findWidget(node, "clear_context_before_run")?.value);
}

function setWidgetHidden(widget) {
    if (!widget) return;

    // Use only the supported visibility fields. In particular, do not replace
    // widget.value/type/computeSize: widget hider extensions can own them.
    widget.hidden = true;
    if (widget.options) widget.options.hidden = true;
}

function outputValue(output, key, fallback = null) {
    const value = output?.[key];
    return Array.isArray(value) ? value[0] ?? fallback : value ?? fallback;
}

function parseHistory(raw) {
    try {
        const history = JSON.parse(String(raw || "[]"));
        if (!Array.isArray(history)) return [];
        return history
            .filter((item) => item && (item.role === "user" || item.role === "assistant"))
            .map((item) => ({ role: item.role, content: contentToText(item.content) }));
    } catch (_) {
        return [];
    }
}

function contentToText(value) {
    if (typeof value === "string") return value;
    if (Array.isArray(value)) {
        return value
            .filter((item) => item && typeof item === "object" && ["text", "output_text"].includes(item.type))
            .map((item) => String(item.text || ""))
            .join("");
    }
    return value == null ? "" : String(value);
}

function widgetHeight(widget, width) {
    const computed = widget.computeSize?.(width);
    const height = Number(computed?.[1]);
    return Math.max(20, Number.isFinite(height) ? height : 20) + 4;
}

function visibleWidgetHeight(node, width) {
    return (node.widgets || [])
        .filter((widget) => !widget.hidden && !widget.options?.hidden)
        .reduce((total, widget) => total + widgetHeight(widget, width), 8);
}

function nativeWidgetBottom(node, width) {
    const widgetTop = Math.max(PANEL_TOP, Number(node.widgets_start_y) || PANEL_TOP);
    const manuallyMeasuredBottom = widgetTop + visibleWidgetHeight(node, width);
    // Modern ComfyUI widgets expose their actual canvas position after layout
    // in `y`/`last_y`. This is more reliable than computeSize(): some DOM
    // widgets report the whole node height there, which would leave a zero
    // height conversation panel until an input gets connected.
    const positionedBottom = (node.widgets || [])
        .filter((widget) => !widget.hidden && !widget.options?.hidden)
        .reduce((bottom, widget) => {
            const y = Number(widget.last_y ?? widget.y);
            if (!Number.isFinite(y) || y <= 0) return bottom;
            return Math.max(bottom, y + widgetHeight(widget, width));
        }, 0);
    return Math.max(manuallyMeasuredBottom, positionedBottom);
}

function updateChatLayout(node) {
    // Intentionally do not set node.size or widgets_start_y here. Both are
    // managed by ComfyUI's native widget layout, and writing either from an
    // onResize callback can feed the calculated widget extent back into the
    // node's height indefinitely. The Conversation viewport derives its own
    // bounds in panelGeometry instead.
    node.__lwbChatMessageCache = undefined;
}

function initializeChatSize(node) {
    node.properties ||= {};
    if (node.properties.lwb_chat_layout_version === CHAT_LAYOUT_VERSION) return;

    // One-time migration for nodes saved by the previous resizing loop. New
    // nodes and migrated nodes start at a useful default, then remain freely
    // resizable through the normal ComfyUI resize handle.
    node.properties.lwb_chat_layout_version = CHAT_LAYOUT_VERSION;
    const width = Math.max(Number(node.size?.[0]) || 0, MIN_WIDTH);
    const nativeBottom = nativeWidgetBottom(node, width);
    if (typeof node.setSize === "function") {
        node.setSize([width, Math.max(MIN_HEIGHT, nativeBottom + PANEL_MIN_HEIGHT + PANEL_BOTTOM_GAP)]);
    }
}

function applySafeChatLayout(node) {
    for (const name of HIDDEN_WIDGETS) setWidgetHidden(findWidget(node, name));

    const message = findWidget(node, "lwb_user_message");
    if (message) {
        message.label = "输入消息";
        if (message.options) message.options.placeholder = "输入消息后点击上方“发送”";
    }
    updateChatLayout(node);
    node.setDirtyCanvas?.(true, true);
}

function panelGeometry(node) {
    // MIN_WIDTH/MIN_HEIGHT are creation defaults only. Do not force them
    // during drawing: users must be able to resize this like a normal node
    // without the canvas panel drawing beyond its bottom edge.
    const width = Math.max(80, Number(node.size?.[0]) || MIN_WIDTH) - PANEL_LEFT * 2;
    const height = Math.max(0, Number(node.size?.[1]) || MIN_HEIGHT);
    // Native ComfyUI widgets retain their normal placement at the top of the
    // node. Put the canvas-only conversation below them, rather than moving
    // them with widgets_start_y.
    const widgetBottom = nativeWidgetBottom(node, width);
    const y = Math.max(PANEL_TOP, widgetBottom + PANEL_BOTTOM_GAP);
    return {
        x: PANEL_LEFT,
        y,
        width,
        height: Math.max(0, height - y - PANEL_BOTTOM_GAP),
    };
}

function wrapText(ctx, text, width) {
    const lines = [];
    for (const sourceLine of String(text || "").replace(/\r/g, "").split("\n")) {
        if (!sourceLine) {
            lines.push("");
            continue;
        }
        let line = "";
        for (const character of sourceLine) {
            const candidate = line + character;
            if (line && ctx.measureText(candidate).width > width) {
                lines.push(line);
                line = character;
            } else {
                line = candidate;
            }
        }
        if (line) lines.push(line);
    }
    return lines.length ? lines : ["（空消息）"];
}

function cachedMessages(node, ctx, textWidth) {
    const raw = String(node.__lwbChatDisplayHistoryJson ?? findWidget(node, "lwb_history_json")?.value ?? "[]");
    const safeTextWidth = Math.max(24, Number(textWidth) || 24);
    const cacheKey = `${Math.floor(safeTextWidth)}\u0000${raw}`;
    if (node.__lwbChatMessageCache?.key === cacheKey) return node.__lwbChatMessageCache.items;

    ctx.font = "13px sans-serif";
    const items = parseHistory(raw).map((message) => {
        const lines = wrapText(ctx, message.content, safeTextWidth);
        return { ...message, lines, height: 28 + lines.length * LINE_HEIGHT };
    });
    node.__lwbChatMessageCache = { key: cacheKey, items };
    return items;
}

function actionRects(panel) {
    const definitions = [
        { id: "send", label: "发送", width: 46, color: "#356d50" },
        { id: "clear_context", label: "清空上下文", width: 78, color: "#65494a" },
        { id: "clear_input", label: "清空输入", width: 66, color: "#4d5661" },
        { id: "scroll_down", label: "▼", width: 22, color: "#4d5661" },
        { id: "scroll_up", label: "▲", width: 22, color: "#4d5661" },
    ];
    let right = panel.x + panel.width - 7;
    return definitions.map((definition) => {
        right -= definition.width;
        const rectangle = { ...definition, x: right, y: panel.y + 6, height: 20 };
        right -= 5;
        return rectangle;
    });
}

function shortStatus(node) {
    const text = String(node.__lwbChatStatus || "准备就绪");
    return text.length > 28 ? `${text.slice(0, 27)}…` : text;
}

function parseContextState(value) {
    try {
        const state = typeof value === "string" ? JSON.parse(value) : value;
        return state && typeof state === "object" ? state : {};
    } catch (_) {
        return {};
    }
}

function formatTokenCount(value) {
    const count = Math.max(0, Number(value) || 0);
    if (count >= 1000000) return `${(count / 1000000).toFixed(1)}M`;
    if (count >= 1000) return `${(count / 1000).toFixed(count >= 10000 ? 0 : 1)}k`;
    return String(Math.round(count));
}

function contextSizeFromLinkedBackend(node) {
    const input = node.inputs?.find((item) => item.name === "backend");
    if (input?.link == null) return 0;
    const links = node.graph?.links;
    const link = links?.[input.link] ?? links?.get?.(input.link);
    const sourceId = link?.origin_id ?? link?.originId ?? link?.[1];
    const source = node.graph?.getNodeById?.(sourceId);
    const value = Number(findWidget(source, "context_size")?.value);
    return Number.isFinite(value) && value > 0 ? value : 0;
}

function contextLimit(node) {
    const reported = Number(node.__lwbContextState?.context_size);
    return Number.isFinite(reported) && reported > 0 ? reported : contextSizeFromLinkedBackend(node);
}

function contextSummary(node) {
    const state = node.__lwbContextState || {};
    const used = Math.max(0, Number(state.estimated_tokens) || 0);
    const limit = contextLimit(node);
    if (!limit) return `上下文 ~${formatTokenCount(used)} / ?`;
    const reportedPercent = Number(state.usage_percent);
    const percent = Number.isFinite(reportedPercent) ? Math.max(0, reportedPercent) : used / limit * 100;
    return `上下文 ~${formatTokenCount(used)}/${formatTokenCount(limit)} ${percent.toFixed(0)}%`;
}

function contextDetail(node) {
    const state = node.__lwbContextState || {};
    const limit = contextLimit(node);
    const images = Math.max(0, Number(state.image_count) || 0);
    const base = limit
        ? `预计剩余 ~${formatTokenCount(Math.max(0, limit - (Number(state.estimated_tokens) || 0)))} tokens`
        : "上限未知（外接服务器可在 Connection 设置 context_size）";
    return images ? `${base} · ${images} 张图片视觉 token 未计入` : `${base} · 文本估算`;
}

function drawAction(ctx, action, disabled) {
    const blocked = disabled && ["send", "clear_context"].includes(action.id);
    ctx.fillStyle = blocked ? "#40454c" : action.color;
    ctx.fillRect(action.x, action.y, action.width, action.height);
    ctx.strokeStyle = "#7c8794";
    ctx.strokeRect(action.x + 0.5, action.y + 0.5, action.width - 1, action.height - 1);
    ctx.font = "11px sans-serif";
    ctx.fillStyle = blocked ? "#9aa2ad" : "#edf2f7";
    ctx.fillText(action.label, action.x + 6, action.y + 14);
}

function drawConversation(node, ctx) {
    if (node.flags?.collapsed) return;
    const panel = panelGeometry(node);
    if (panel.width <= 40 || panel.height < PANEL_HEADER_HEIGHT + 24) return;
    const actions = actionRects(panel);
    const busy = Boolean(node.__lwbChatBusy);
    ctx.fillStyle = "#1f2329";
    ctx.fillRect(panel.x, panel.y, panel.width, panel.height);
    ctx.strokeStyle = "#59636f";
    ctx.strokeRect(panel.x + 0.5, panel.y + 0.5, panel.width - 1, panel.height - 1);
    ctx.fillStyle = "#c8d0da";
    ctx.font = "600 13px sans-serif";
    ctx.fillText("Conversation", panel.x + 9, panel.y + 20);
    ctx.font = "11px sans-serif";
    const contextState = node.__lwbContextState || {};
    const usage = Number(contextState.usage_percent);
    ctx.fillStyle = Number.isFinite(usage) && usage >= 90 ? "#e38080" : Number.isFinite(usage) && usage >= 75 ? "#e6c36a" : "#92c8a3";
    ctx.fillText(contextSummary(node), panel.x + 95, panel.y + 20);
    ctx.font = "10px sans-serif";
    ctx.fillStyle = node.__lwbChatStatusKind === "error" ? "#e38080" : busy ? "#e6c36a" : "#8f9dab";
    const detail = busy || node.__lwbChatStatusKind === "error" ? shortStatus(node) : contextDetail(node);
    ctx.fillText(detail, panel.x + 9, panel.y + 37);
    for (const action of actions) drawAction(ctx, action, busy);

    const innerTop = panel.y + PANEL_HEADER_HEIGHT;
    const innerHeight = panel.height - PANEL_HEADER_HEIGHT - 4;
    // All three values below must come from the same geometry. Previously
    // text was measured against a wider area than its bubble, so long lines
    // leaked through the right edge or outside a right-aligned user bubble.
    // Keep every bubble clear of the scrollbar track. Right-aligned user
    // bubbles used to end under the track, hiding their right edge.
    const bubbleWidth = Math.max(40, panel.width - PANEL_LEFT - MESSAGE_RIGHT_GUTTER);
    const bubbleTextWidth = Math.max(24, bubbleWidth - 18);
    const messages = cachedMessages(node, ctx, bubbleTextWidth);
    const contentHeight = messages.reduce((total, item) => total + item.height + 7, 0);
    const maximumScroll = Math.max(0, contentHeight - innerHeight);
    if (node.__lwbChatScrollToBottom) {
        node.__lwbChatScroll = maximumScroll;
        node.__lwbChatScrollToBottom = false;
    }
    const scroll = Math.min(maximumScroll, Math.max(0, Number(node.__lwbChatScroll) || 0));
    node.__lwbChatScroll = scroll;
    node.__lwbChatScrollMetrics = null;

    ctx.save();
    ctx.beginPath();
    ctx.rect(panel.x + 1, innerTop, panel.width - 2, innerHeight);
    ctx.clip();
    if (!messages.length) {
        ctx.fillStyle = "#8e98a5";
        ctx.font = "13px sans-serif";
        ctx.fillText("输入消息后点击“发送”；长对话可用滚轮或 ▲/▼ 查看。", panel.x + 12, innerTop + 28);
    } else {
        let cursor = innerTop + 6 - scroll;
        for (const item of messages) {
            const isUser = item.role === "user";
            const bubbleX = isUser
                ? panel.x + panel.width - MESSAGE_RIGHT_GUTTER - bubbleWidth
                : panel.x + 9;
            ctx.fillStyle = isUser ? "#29493a" : "#2c3139";
            ctx.fillRect(bubbleX, cursor, bubbleWidth, item.height);
            ctx.strokeStyle = isUser ? "#538467" : "#5b6572";
            ctx.strokeRect(bubbleX + 0.5, cursor + 0.5, bubbleWidth - 1, item.height - 1);
            ctx.font = "600 11px sans-serif";
            ctx.fillStyle = isUser ? "#a8d8ba" : "#c7d0dc";
            ctx.fillText(isUser ? "你" : "助手", bubbleX + 9, cursor + 16);
            ctx.font = "13px sans-serif";
            ctx.fillStyle = "#edf1f5";
            item.lines.forEach((line, index) => {
                ctx.fillText(line, bubbleX + 9, cursor + 34 + index * LINE_HEIGHT);
            });
            cursor += item.height + 7;
        }
    }
    ctx.restore();

    if (maximumScroll > 0) {
        const trackWidth = 7;
        const trackX = panel.x + panel.width - trackWidth - 3;
        const trackY = innerTop + 3;
        const trackHeight = innerHeight - 6;
        const thumbHeight = Math.max(20, trackHeight * (innerHeight / contentHeight));
        const thumbY = trackY + (trackHeight - thumbHeight) * (scroll / maximumScroll);
        ctx.fillStyle = "#56606d";
        ctx.fillRect(trackX, trackY, trackWidth, trackHeight);
        ctx.fillStyle = "#a5b2bf";
        ctx.fillRect(trackX, thumbY, trackWidth, thumbHeight);
        // Persist the exact drawn geometry so click/drag handling does not
        // need to guess at wrapped message heights outside a draw pass.
        node.__lwbChatScrollMetrics = {
            trackX,
            trackY,
            trackWidth,
            trackHeight,
            thumbHeight,
            maximumScroll,
        };
    }
}

function localPointer(node, event, args) {
    // LiteGraph supplies `pos` to node mouse callbacks in node-local
    // coordinates. Some newer frontend wrappers instead expose graph_mouse;
    // handle both without relying on DOM widgets.
    for (const argument of args) {
        if (Array.isArray(argument) && argument.length >= 2 && Number.isFinite(argument[0]) && Number.isFinite(argument[1])) {
            return argument;
        }
        if (Array.isArray(argument?.graph_mouse) && argument.graph_mouse.length >= 2) {
            return [argument.graph_mouse[0] - node.pos[0], argument.graph_mouse[1] - node.pos[1]];
        }
    }
    if (Array.isArray(event?.graph_mouse) && event.graph_mouse.length >= 2) {
        return [event.graph_mouse[0] - node.pos[0], event.graph_mouse[1] - node.pos[1]];
    }
    const canvasMouse = app.canvas?.graph_mouse;
    if (Array.isArray(canvasMouse) && canvasMouse.length >= 2) {
        return [canvasMouse[0] - node.pos[0], canvasMouse[1] - node.pos[1]];
    }
    return null;
}

function setScrollFromTrack(node, pointerY, metrics) {
    if (!metrics || metrics.maximumScroll <= 0) return false;
    const movableHeight = Math.max(1, metrics.trackHeight - metrics.thumbHeight);
    const thumbTop = Math.min(
        metrics.trackY + movableHeight,
        Math.max(metrics.trackY, pointerY - metrics.thumbHeight / 2),
    );
    node.__lwbChatScrollToBottom = false;
    node.__lwbChatScroll = ((thumbTop - metrics.trackY) / movableHeight) * metrics.maximumScroll;
    node.setDirtyCanvas?.(true, true);
    return true;
}

function isOnScrollTrack(pointer, metrics) {
    return Boolean(
        metrics
        && pointer[0] >= metrics.trackX - 3
        && pointer[0] <= metrics.trackX + metrics.trackWidth + 3
        && pointer[1] >= metrics.trackY
        && pointer[1] <= metrics.trackY + metrics.trackHeight,
    );
}

function setStatus(node, status, kind = "idle") {
    node.__lwbChatStatus = status;
    node.__lwbChatStatusKind = kind;
    node.setDirtyCanvas?.(true, true);
}

function setBusy(node, busy, status = busy ? "正在加入队列…" : "准备就绪", kind = busy ? "busy" : "idle") {
    node.__lwbChatBusy = busy;
    setStatus(node, status, kind);
}

function isPromptLink(value, output) {
    if (!Array.isArray(value) || value.length !== 2) return false;
    const sourceId = value[0];
    const outputSlot = value[1];
    const validSource = typeof sourceId === "number" || (typeof sourceId === "string" && /^\d+$/.test(sourceId));
    return validSource && typeof outputSlot === "number" && Number.isFinite(outputSlot) && Boolean(output?.[String(sourceId)] ?? output?.[Number(sourceId)]);
}

function collectPromptLinks(value, output, result = new Set()) {
    if (isPromptLink(value, output)) {
        result.add(String(value[0]));
        return result;
    }
    if (Array.isArray(value)) {
        for (const item of value) collectPromptLinks(item, output, result);
    } else if (value && typeof value === "object") {
        for (const item of Object.values(value)) collectPromptLinks(item, output, result);
    }
    return result;
}

async function buildChatOnlyPrompt(node) {
    const prompt = await app.graphToPrompt();
    const output = prompt?.output;
    const targetId = String(node.id);
    const target = output?.[targetId] ?? output?.[Number(targetId)];
    if (!target) throw new Error("当前 Chat 节点不在可执行提示中，请检查 backend 连接。");

    const keep = new Set();
    const addWithAncestors = (nodeId) => {
        const id = String(nodeId);
        if (keep.has(id)) return;
        const apiNode = output[id] ?? output[Number(id)];
        if (!apiNode) return;
        keep.add(id);
        for (const sourceId of collectPromptLinks(apiNode.inputs || {}, output)) addWithAncestors(sourceId);
    };
    addWithAncestors(targetId);

    const scopedOutput = {};
    for (const [id, apiNode] of Object.entries(output)) {
        if (keep.has(String(id))) scopedOutput[id] = apiNode;
    }
    prompt.output = scopedOutput;
    return prompt;
}

async function sendMessage(node) {
    const message = findWidget(node, "lwb_user_message");
    const request = findWidget(node, "lwb_request_id");
    const text = String(message?.value || "").trim();
    const messageInput = node.inputs?.find((input) => input.name === "lwb_user_message");
    const hasUpstreamMessage = messageInput?.link != null;
    if (node.__lwbChatBusy) return;
    if (!text && !hasUpstreamMessage) {
        setStatus(node, "请先输入消息。", "error");
        return;
    }
    if (!message || !request) {
        setStatus(node, "Chat 状态控件未初始化，请重新添加节点。", "error");
        return;
    }

    request.value = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setBusy(node, true);
    node.graph?.setDirtyCanvas?.(true, true);
    try {
        const prompt = await buildChatOnlyPrompt(node);
        await api.queuePrompt(0, prompt);
        setStatus(node, "已加入队列，等待回复…", "busy");
    } catch (error) {
        setBusy(node, false, `加入队列失败：${error?.message || error}`, "error");
    }
}

function clearContext(node) {
    if (node.__lwbChatBusy) return;
    const history = findWidget(node, "lwb_history_json");
    const flow = findWidget(node, "lwb_flow_state_json");
    const request = findWidget(node, "lwb_request_id");
    if (history) history.value = "[]";
    if (flow) flow.value = "{}";
    if (request) request.value = `${Date.now()}-clear`;
    node.__lwbChatDisplayHistoryJson = "[]";
    node.__lwbChatMessageCache = undefined;
    node.__lwbContextState = {};
    if (node.properties) delete node.properties.lwb_context_state;
    node.__lwbChatScroll = 0;
    node.__lwbChatScrollToBottom = false;
    setStatus(node, "上下文已清空。", "idle");
}

function clearInput(node) {
    if (node.__lwbChatBusy) return;
    const message = findWidget(node, "lwb_user_message");
    if (message) message.value = "";
    setStatus(node, "输入已清空。", "idle");
}

function handlePanelClick(node, event, args) {
    const pointer = localPointer(node, event, args);
    if (!pointer) return false;
    const panel = panelGeometry(node);
    const metrics = node.__lwbChatScrollMetrics;
    if (isOnScrollTrack(pointer, metrics)) {
        node.__lwbChatScrollDragging = true;
        setScrollFromTrack(node, pointer[1], metrics);
        event?.preventDefault?.();
        event?.stopPropagation?.();
        return true;
    }
    const action = actionRects(panel).find((item) => (
        pointer[0] >= item.x && pointer[0] <= item.x + item.width && pointer[1] >= item.y && pointer[1] <= item.y + item.height
    ));
    if (!action) return false;
    if (action.id === "send") void sendMessage(node);
    if (action.id === "clear_context") clearContext(node);
    if (action.id === "clear_input") clearInput(node);
    if (action.id === "scroll_up") {
        node.__lwbChatScrollToBottom = false;
        node.__lwbChatScroll = Math.max(0, (Number(node.__lwbChatScroll) || 0) - 160);
        node.setDirtyCanvas?.(true, true);
    }
    if (action.id === "scroll_down") {
        node.__lwbChatScrollToBottom = false;
        node.__lwbChatScroll = (Number(node.__lwbChatScroll) || 0) + 160;
        node.setDirtyCanvas?.(true, true);
    }
    event?.preventDefault?.();
    event?.stopPropagation?.();
    return true;
}

function handlePanelScroll(node, event, args) {
    const pointer = localPointer(node, event, args);
    if (!pointer) return false;
    const panel = panelGeometry(node);
    const inPanel = pointer[0] >= panel.x && pointer[0] <= panel.x + panel.width && pointer[1] >= panel.y + PANEL_HEADER_HEIGHT && pointer[1] <= panel.y + panel.height;
    if (!inPanel) return false;
    const rawDelta = Number(event?.deltaY ?? (event?.wheelDelta ? -event.wheelDelta : 0));
    if (!rawDelta) return false;
    node.__lwbChatScrollToBottom = false;
    node.__lwbChatScroll = Math.max(0, (Number(node.__lwbChatScroll) || 0) + Math.sign(rawDelta) * 42);
    node.setDirtyCanvas?.(true, true);
    event?.preventDefault?.();
    event?.stopPropagation?.();
    return true;
}

function handlePanelDrag(node, event, args) {
    if (!node.__lwbChatScrollDragging) return false;
    const pointer = localPointer(node, event, args);
    if (!pointer) return false;
    setScrollFromTrack(node, pointer[1], node.__lwbChatScrollMetrics);
    event?.preventDefault?.();
    event?.stopPropagation?.();
    return true;
}

function finishPanelDrag(node) {
    if (!node.__lwbChatScrollDragging) return false;
    node.__lwbChatScrollDragging = false;
    return true;
}

function onChatExecuted(node, output) {
    const history = outputValue(output, "history_json");
    const flow = outputValue(output, "flow_state_json");
    const sent = Boolean(outputValue(output, "sent", false));
    const contextState = parseContextState(outputValue(output, "context_state_json", "{}"));
    const clearBeforeRun = clearsContextBeforeRun(node);
    if (history !== null) {
        const widget = findWidget(node, "lwb_history_json");
        node.__lwbChatDisplayHistoryJson = String(history);
        if (widget) widget.value = clearBeforeRun ? "[]" : String(history);
    }
    if (flow !== null) {
        const widget = findWidget(node, "lwb_flow_state_json");
        if (widget) widget.value = clearBeforeRun ? "{}" : String(flow);
    }
    node.__lwbContextState = contextState;
    node.properties ||= {};
    node.properties.lwb_context_state = contextState;
    if (sent) {
        const message = findWidget(node, "lwb_user_message");
        if (message) message.value = "";
        node.__lwbChatScrollToBottom = true;
        setBusy(node, false, "准备就绪。", "idle");
    }
    node.__lwbChatMessageCache = undefined;
    applySafeChatLayout(node);
}

function markBusyChatsFailed() {
    for (const node of app.graph?._nodes || []) {
        if (node.comfyClass === CHAT_NODE && node.__lwbChatBusy) {
            setBusy(node, false, "生成失败，请查看 ComfyUI 日志。", "error");
        }
    }
}

app.registerExtension({
    name: "LlamaWorkbench.InteractiveChat",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== CHAT_NODE) return;

        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            this.__lwbChatScroll = 0;
            this.__lwbChatScrollDragging = false;
            this.__lwbChatScrollToBottom = true;
            this.__lwbChatStatus = "准备就绪。";
            this.__lwbChatStatusKind = "idle";
            this.__lwbContextState = parseContextState(this.properties?.lwb_context_state);
            this.__lwbChatDisplayHistoryJson = clearsContextBeforeRun(this) ? "[]" : undefined;
            initializeChatSize(this);
            applySafeChatLayout(this);
            return result;
        };

        const resized = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            const result = resized?.apply(this, arguments);
            updateChatLayout(this);
            return result;
        };

        const drawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const result = drawForeground?.apply(this, arguments);
            drawConversation(this, ctx);
            return result;
        };

        const mouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (event, ...args) {
            if (handlePanelClick(this, event, args)) return true;
            return mouseDown?.apply(this, arguments);
        };

        const mouseWheel = nodeType.prototype.onMouseWheel;
        nodeType.prototype.onMouseWheel = function (event, ...args) {
            if (handlePanelScroll(this, event, args)) return true;
            return mouseWheel?.apply(this, arguments);
        };

        const mouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function (event, ...args) {
            if (handlePanelDrag(this, event, args)) return true;
            return mouseMove?.apply(this, arguments);
        };

        const mouseUp = nodeType.prototype.onMouseUp;
        nodeType.prototype.onMouseUp = function (event, ...args) {
            const consumed = finishPanelDrag(this);
            const result = mouseUp?.apply(this, arguments);
            return consumed || result;
        };
    },
    setup() {
        api.addEventListener("executed", (event) => {
            const detail = event.detail || {};
            const node = app.graph?.getNodeById?.(detail.node);
            if (!node || node.comfyClass !== CHAT_NODE) return;
            onChatExecuted(node, detail.output || {});
        });
        api.addEventListener("execution_error", markBusyChatsFailed);
        api.addEventListener("execution_interrupted", markBusyChatsFailed);
    },
});
