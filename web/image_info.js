import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DISPLAY_NODE = "LlamaWorkbench_ImageInfoDisplay";
const MIN_WIDTH = 420;
const MIN_HEIGHT = 460;
const PADDING = 10;
const HEADER_HEIGHT = 34;
const LINE_HEIGHT = 18;

function outputValue(output, key, fallback = "") {
    const value = output?.[key];
    return Array.isArray(value) ? value[0] ?? fallback : value ?? fallback;
}

function ensureMinimumSize(node) {
    if (typeof node.setSize !== "function") return;
    const current = node.size || [0, 0];
    node.setSize([
        Math.max(current[0] || 0, MIN_WIDTH),
        Math.max(current[1] || 0, MIN_HEIGHT),
    ]);
}

function widgetBottom(node) {
    return (node.widgets || []).reduce((bottom, widget) => {
        const y = Number(widget.last_y ?? widget.y);
        const computed = Number(widget.computeSize?.(node.size?.[0])?.[1]);
        const height = Number.isFinite(computed) ? Math.max(20, computed) : 24;
        return Number.isFinite(y) && y > 0 ? Math.max(bottom, y + height) : bottom;
    }, 84);
}

function panelGeometry(node) {
    const x = PADDING;
    const y = Math.max(92, widgetBottom(node) + PADDING);
    const width = Math.max(120, (node.size?.[0] || MIN_WIDTH) - PADDING * 2);
    const height = Math.max(100, (node.size?.[1] || MIN_HEIGHT) - y - PADDING);
    return { x, y, width, height };
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
    return lines.length ? lines : ["No image data"];
}

function displayLines(node, ctx, text, width) {
    const cacheKey = `${Math.floor(width)}\u0000${text}`;
    if (node.__lwbImageInfoLineCache?.key !== cacheKey) {
        node.__lwbImageInfoLineCache = {
            key: cacheKey,
            lines: wrappedLines(ctx, text, width),
        };
    }
    return node.__lwbImageInfoLineCache.lines;
}

function drawOutput(node, ctx) {
    if (node.flags?.collapsed) return;
    const { x, y, width, height } = panelGeometry(node);

    ctx.fillStyle = "#20242b";
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = "#58a6a6";
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, width - 1, height - 1);

    ctx.font = "600 13px sans-serif";
    ctx.fillStyle = "#70c1bd";
    ctx.fillText(node.__lwbImageInfoTitle || "Image Information", x + PADDING, y + 22);

    ctx.font = "12px monospace";
    ctx.fillStyle = "#d8dce3";
    const lines = displayLines(
        node,
        ctx,
        node.__lwbImageInfoText || "No image data",
        width - PADDING * 2 - 10,
    );
    const visibleLines = Math.max(1, Math.floor((height - HEADER_HEIGHT - PADDING) / LINE_HEIGHT));
    const maximumOffset = Math.max(0, lines.length - visibleLines);
    const offset = Math.min(maximumOffset, Math.max(0, Number(node.__lwbImageInfoScroll) || 0));
    node.__lwbImageInfoScroll = offset;

    ctx.save();
    ctx.beginPath();
    ctx.rect(x + 1, y + HEADER_HEIGHT, width - 2, height - HEADER_HEIGHT - 1);
    ctx.clip();
    for (let index = 0; index < visibleLines && offset + index < lines.length; index += 1) {
        ctx.fillText(lines[offset + index], x + PADDING, y + HEADER_HEIGHT + 16 + index * LINE_HEIGHT);
    }
    ctx.restore();

    if (maximumOffset > 0) {
        const trackX = x + width - 9;
        const trackY = y + HEADER_HEIGHT + 4;
        const trackHeight = height - HEADER_HEIGHT - 8;
        const thumbHeight = Math.max(20, trackHeight * (visibleLines / lines.length));
        const thumbY = trackY + (trackHeight - thumbHeight) * (offset / maximumOffset);
        ctx.fillStyle = "#59616d";
        ctx.fillRect(trackX, trackY, 5, trackHeight);
        ctx.fillStyle = "#70c1bd";
        ctx.fillRect(trackX, thumbY, 5, thumbHeight);
    }
}

function localMousePosition(node, event, args) {
    for (const argument of args) {
        if (Array.isArray(argument) && argument.length >= 2) return argument;
        if (Array.isArray(argument?.graph_mouse)) {
            return [argument.graph_mouse[0] - node.pos[0], argument.graph_mouse[1] - node.pos[1]];
        }
    }
    if (Array.isArray(event?.graph_mouse)) {
        return [event.graph_mouse[0] - node.pos[0], event.graph_mouse[1] - node.pos[1]];
    }
    const canvasMouse = app.canvas?.graph_mouse;
    return Array.isArray(canvasMouse)
        ? [canvasMouse[0] - node.pos[0], canvasMouse[1] - node.pos[1]]
        : null;
}

function scrollOutput(node, event, args) {
    const position = localMousePosition(node, event, args);
    if (!position) return false;
    const { x, y, width, height } = panelGeometry(node);
    if (
        position[0] < x ||
        position[0] > x + width ||
        position[1] < y ||
        position[1] > y + height
    ) {
        return false;
    }
    const rawDelta = Number(event?.deltaY ?? (event?.wheelDelta ? -event.wheelDelta : 0));
    if (!rawDelta) return false;
    node.__lwbImageInfoScroll = Math.max(
        0,
        (Number(node.__lwbImageInfoScroll) || 0) + Math.sign(rawDelta) * 3,
    );
    node.setDirtyCanvas?.(true, true);
    event?.preventDefault?.();
    event?.stopPropagation?.();
    return true;
}

app.registerExtension({
    name: "LlamaWorkbench.ImageInfoDisplay",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== DISPLAY_NODE) return;

        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            ensureMinimumSize(this);
            this.__lwbImageInfoScroll = 0;
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
            if (scrollOutput(this, event, args)) return true;
            return mouseWheel?.apply(this, arguments);
        };
    },
    setup() {
        api.addEventListener("executed", (event) => {
            const detail = event.detail || {};
            const node = app.graph?.getNodeById?.(detail.node);
            if (!node || node.comfyClass !== DISPLAY_NODE) return;
            const output = detail.output || {};
            node.__lwbImageInfoText = String(outputValue(output, "formatted_info"));
            node.__lwbImageInfoTitle = String(outputValue(output, "title", "Image Information"));
            node.__lwbImageInfoLineCache = undefined;
            node.__lwbImageInfoScroll = 0;
            node.setDirtyCanvas?.(true, true);
        });
    },
});
