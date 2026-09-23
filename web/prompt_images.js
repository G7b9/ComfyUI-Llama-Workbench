import { app } from "../../scripts/app.js";

const PROMPT_NODE = "LlamaWorkbench_Prompt";
const PROMPT_ENHANCER_NODE = "LlamaWorkbench_PromptEnhancer";
const PE_CANVAS_NODE = "LlamaWorkbench_QwenImage21PECanvas";
const CHAT_NODE = "LlamaWorkbench_Chat";
const DYNAMIC_IMAGE_NODES = new Set([PROMPT_NODE, PROMPT_ENHANCER_NODE, PE_CANVAS_NODE, CHAT_NODE]);
const MAX_IMAGES = 10;
const IMAGE_NAME = /^image(10|[1-9])$/;

function isImageSlot(input) {
    return input?.name === "image" || IMAGE_NAME.test(input?.name || "");
}

function isConnected(input) {
    return input?.link != null || (Array.isArray(input?.links) && input.links.length > 0);
}

function imageIndex(input) {
    if (input?.name === "image") return 1;
    const match = IMAGE_NAME.exec(input?.name || "");
    return match ? Number(match[1]) : null;
}

function removeInput(node, input) {
    const index = node.inputs?.indexOf(input) ?? -1;
    if (index >= 0) node.removeInput(index);
}

function resizeForInputs(node) {
    if (typeof node.computeSize !== "function" || typeof node.setSize !== "function") return;
    const current = node.size || [0, 0];
    const minimum = node.computeSize();
    node.setSize([Math.max(current[0] || 0, minimum[0] || 0), Math.max(current[1] || 0, minimum[1] || 0)]);
}

function synchronizeImageSlots(node) {
    if (node.__lwbSynchronizingImages) return;
    node.__lwbSynchronizingImages = true;
    try {
        const inputs = (node.inputs || []).filter(isImageSlot);
        const connected = inputs.filter(isConnected);
        const assigned = new Map();

        for (const input of connected) {
            const preferred = imageIndex(input);
            if (preferred !== null && !assigned.has(preferred)) assigned.set(preferred, input);
        }

        // A legacy `image` connection becomes the first numbered slot. If a
        // loaded graph already has image1, preserve that connection and use the
        // next free number instead.
        const legacy = connected.find((input) => input.name === "image");
        if (legacy) {
            let target = 1;
            while (assigned.has(target) && assigned.get(target) !== legacy) target += 1;
            assigned.set(Math.min(target, MAX_IMAGES), legacy);
        }

        const usedInputs = new Set(assigned.values());
        for (const input of [...inputs].reverse()) {
            if (!usedInputs.has(input)) removeInput(node, input);
        }

        const ordered = [...assigned.entries()].sort(([left], [right]) => left - right);
        for (const [index, input] of ordered) input.name = `image${index}`;

        const largest = ordered.length ? ordered[ordered.length - 1][0] : 0;
        const wanted = largest ? Math.min(MAX_IMAGES, largest + 1) : 1;
        for (let index = 1; index <= wanted; index += 1) {
            const name = `image${index}`;
            if (!(node.inputs || []).some((input) => input.name === name)) node.addInput(name, "IMAGE");
        }

        if (largest === 0) {
            const first = (node.inputs || []).find((input) => input.name === "image1");
            if (first) first.name = "image";
        }
        resizeForInputs(node);
        node.setDirtyCanvas?.(true, true);
    } finally {
        node.__lwbSynchronizingImages = false;
    }
}

function scheduleSynchronization(node) {
    queueMicrotask(() => synchronizeImageSlots(node));
}

app.registerExtension({
    name: "LlamaWorkbench.DynamicImages",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!DYNAMIC_IMAGE_NODES.has(nodeData.name)) return;

        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            scheduleSynchronization(this);
            return result;
        };

        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = configured?.apply(this, arguments);
            scheduleSynchronization(this);
            return result;
        };

        const connectionsChanged = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = connectionsChanged?.apply(this, arguments);
            scheduleSynchronization(this);
            return result;
        };
    },
});
