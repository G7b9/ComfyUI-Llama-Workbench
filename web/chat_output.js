import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DISPLAY_NODE = "LlamaWorkbench_ChatDisplay";
const MIN_WIDTH = 480;
const MIN_HEIGHT = 440;
const PADDING = 10;
const LINE_HEIGHT = 17;

function outputValue(output, key, fallback = "") {
    const value = output?.[key];
    return Array.isArray(value) ? value[0] ?? fallback : value ?? fallback;
}

function ensureMinimumSize(node) {
    if (typeof node.setSize !== "function") return;
    const current = node.size || [0, 0];
    node.setSize([Math.max(current[0] || 0, MIN_WIDTH), Math.max(current[1] || 0, MIN_HEIGHT)]);
}

function wrappedLines(ctx, value, width) {
    const lines = [];
    for (const sourceLine of String(value || "").replace(/\r/g, "").split("\n")) {
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
    if (!lines.length) lines.push("（无内容）");
    return lines;
}

function cachedLines(node, key, ctx, content, width, emptyMessage) {
    const text = String(content || emptyMessage);
    const cacheKey = `${Math.floor(width)}\u0000${text}`;
    node.__lwbWrappedLines ||= {};
    if (node.__lwbWrappedLines[key]?.cacheKey !== cacheKey) {
        node.__lwbWrappedLines[key] = { cacheKey, lines: wrappedLines(ctx, text, width) };
    }
    return node.__lwbWrappedLines[key].lines;
}

function drawSection(node, ctx, key, x, y, width, height, title, content, color, emptyMessage) {
    node.__lwbOutputScrollMetrics ||= {};
    node.__lwbOutputScrollMetrics[key] = null;
    ctx.fillStyle = "#20242b";
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, width - 1, height - 1);

    ctx.font = "600 13px sans-serif";
    ctx.fillStyle = color;
    ctx.fillText(title, x + PADDING, y + 21);
    ctx.font = "12px sans-serif";
    ctx.fillStyle = "#d8dce3";
    const maximum = Math.max(1, Math.floor((height - 48) / LINE_HEIGHT));
    const lines = cachedLines(node, key, ctx, content, width - PADDING * 2 - 10, emptyMessage);
    const maximumOffset = Math.max(0, lines.length - maximum);
    const scrollKey = key === "thinking" ? "__lwbThinkingScroll" : "__lwbAssistantScroll";
    const offset = Math.min(maximumOffset, Math.max(0, Number(node[scrollKey]) || 0));
    node[scrollKey] = offset;

    ctx.save();
    ctx.beginPath();
    ctx.rect(x + 1, y + 32, width - 2, height - 34);
    ctx.clip();
    for (let index = 0; index < maximum && offset + index < lines.length; index += 1) {
        ctx.fillText(lines[offset + index], x + PADDING, y + 42 + index * LINE_HEIGHT);
    }
    ctx.restore();

    if (maximumOffset > 0) {
        const trackWidth = 7;
        const trackX = x + width - trackWidth - 3;
        const trackY = y + 32;
        const trackHeight = height - 38;
        const thumbHeight = Math.max(20, trackHeight * (maximum / lines.length));
        const thumbY = trackY + (trackHeight - thumbHeight) * (offset / maximumOffset);
        ctx.fillStyle = "#67707d";
        ctx.fillRect(trackX, trackY, trackWidth, trackHeight);
        ctx.fillStyle = color;
        ctx.fillRect(trackX, thumbY, trackWidth, thumbHeight);
        ctx.font = "11px sans-serif";
        ctx.fillStyle = "#9da5b1";
        ctx.fillText("滚动查看", x + width - 58, y + 21);
        // Keep the actual drawn track for pointer handling. This avoids
        // recalculating wrapped text during mouse events.
        node.__lwbOutputScrollMetrics[key] = {
            scrollKey,
            trackX,
            trackY,
            trackWidth,
            trackHeight,
            thumbHeight,
            maximumOffset,
        };
    }
}

function panelGeometry(node) {
    const width = Math.max(MIN_WIDTH, node.size?.[0] || MIN_WIDTH) - PADDING * 2;
    const top = 68;
    const totalHeight = Math.max(260, (node.size?.[1] || MIN_HEIGHT) - top - PADDING);
    const panelHeight = Math.floor((totalHeight - PADDING) / 2);
    return { width, top, panelHeight };
}

function drawOutput(node, ctx) {
    if (node.flags?.collapsed) return;
    const { width, top, panelHeight } = panelGeometry(node);
    drawSection(node, ctx, "thinking", PADDING, top, width, panelHeight, "Thinking", node.__lwbThinking, "#c59b50", "（无思考过程）");
    drawSection(
        node,
        ctx,
        "assistant",
        PADDING,
        top + panelHeight + PADDING,
        width,
        panelHeight,
        "Final answer",
        node.__lwbAssistantMessage,
        "#7dbb8a",
        "（模型未返回正式回答）",
    );
}

function localMousePosition(node, event, args) {
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

function isOnOutputTrack(position, metrics) {
    return Boolean(
        metrics
        && position[0] >= metrics.trackX - 3
        && position[0] <= metrics.trackX + metrics.trackWidth + 3
        && position[1] >= metrics.trackY
        && position[1] <= metrics.trackY + metrics.trackHeight,
    );
}

function setOutputScrollFromTrack(node, pointerY, metrics) {
    if (!metrics || metrics.maximumOffset <= 0) return false;
    const movableHeight = Math.max(1, metrics.trackHeight - metrics.thumbHeight);
    const thumbTop = Math.min(
        metrics.trackY + movableHeight,
        Math.max(metrics.trackY, pointerY - metrics.thumbHeight / 2),
    );
    node[metrics.scrollKey] = Math.round(((thumbTop - metrics.trackY) / movableHeight) * metrics.maximumOffset);
    node.setDirtyCanvas?.(true, true);
    return true;
}

function startOutputDrag(node, event, args) {
    const position = localMousePosition(node, event, args);
    if (!position) return false;
    const entry = Object.entries(node.__lwbOutputScrollMetrics || {}).find(([, metrics]) => isOnOutputTrack(position, metrics));
    if (!entry) return false;
    const [key, metrics] = entry;
    node.__lwbOutputScrollDragging = key;
    setOutputScrollFromTrack(node, position[1], metrics);
    event?.preventDefault?.();
    event?.stopPropagation?.();
    return true;
}

function dragOutput(node, event, args) {
    const key = node.__lwbOutputScrollDragging;
    if (!key) return false;
    const position = localMousePosition(node, event, args);
    const metrics = node.__lwbOutputScrollMetrics?.[key];
    if (!position || !metrics) return false;
    setOutputScrollFromTrack(node, position[1], metrics);
    event?.preventDefault?.();
    event?.stopPropagation?.();
    return true;
}

function finishOutputDrag(node) {
    if (!node.__lwbOutputScrollDragging) return false;
    node.__lwbOutputScrollDragging = "";
    return true;
}

function scrollPanel(node, event, args) {
    const position = localMousePosition(node, event, args);
    if (!position) return false;
    const localX = position[0];
    const localY = position[1];
    const { width, top, panelHeight } = panelGeometry(node);
    if (localX < PADDING || localX > PADDING + width) return false;
    const thinkingPanel = localY >= top && localY <= top + panelHeight;
    const assistantTop = top + panelHeight + PADDING;
    const assistantPanel = localY >= assistantTop && localY <= assistantTop + panelHeight;
    if (!thinkingPanel && !assistantPanel) return false;

    const rawDelta = Number(event?.deltaY ?? (event?.wheelDelta ? -event.wheelDelta : 0));
    if (!rawDelta) return false;
    const key = thinkingPanel ? "__lwbThinkingScroll" : "__lwbAssistantScroll";
    node[key] = Math.max(0, (Number(node[key]) || 0) + Math.sign(rawDelta) * 3);
    node.setDirtyCanvas?.(true, true);
    event?.preventDefault?.();
    event?.stopPropagation?.();
    return true;
}

app.registerExtension({
    name: "LlamaWorkbench.ChatOutputDisplay",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== DISPLAY_NODE) return;

        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            ensureMinimumSize(this);
            this.__lwbThinkingScroll = 0;
            this.__lwbAssistantScroll = 0;
            this.__lwbOutputScrollDragging = "";
            return result;
        };

        const drawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const result = drawForeground?.apply(this, arguments);
            drawOutput(this, ctx);
            return result;
        };

        const mouseWheel = nodeType.prototype.onMouseWheel;
        nodeType.prototype.onMouseWheel = function (event, ...args) {
            if (scrollPanel(this, event, args)) return true;
            return mouseWheel?.apply(this, arguments);
        };

        const mouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (event, ...args) {
            if (startOutputDrag(this, event, args)) return true;
            return mouseDown?.apply(this, arguments);
        };

        const mouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function (event, ...args) {
            if (dragOutput(this, event, args)) return true;
            return mouseMove?.apply(this, arguments);
        };

        const mouseUp = nodeType.prototype.onMouseUp;
        nodeType.prototype.onMouseUp = function () {
            const consumed = finishOutputDrag(this);
            const result = mouseUp?.apply(this, arguments);
            return consumed || result;
        };
    },
    setup() {
        api.addEventListener("executed", (event) => {
            const detail = event.detail || {};
            const node = app.graph?.getNodeById?.(detail.node);
            if (!node || node.comfyClass !== DISPLAY_NODE) return;
            const output = detail.output || {};
            node.__lwbThinking = String(outputValue(output, "thinking"));
            node.__lwbAssistantMessage = String(outputValue(output, "assistant_message"));
            node.__lwbWrappedLines = undefined;
            node.__lwbThinkingScroll = 0;
            node.__lwbAssistantScroll = 0;
            node.setDirtyCanvas?.(true, true);
        });
    },
});
