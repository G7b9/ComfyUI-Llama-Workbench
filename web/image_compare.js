import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const COMPARE_NODE = "LlamaWorkbench_ImageCompare";
let selectionGroupId = 0;

export function pairImages(a = [], b = [], repeat = false) {
    return Array.from({ length: Math.max(a.length, b.length) }, (_, index) => {
        const ai = index < a.length ? index : repeat && a.length ? a.length - 1 : -1;
        const bi = index < b.length ? index : repeat && b.length ? b.length - 1 : -1;
        return { a: a[ai], b: b[bi], ai, bi, repeatA: ai >= 0 && ai !== index, repeatB: bi >= 0 && bi !== index };
    });
}

function element(tag, className, text) {
    const result = document.createElement(tag);
    result.className = className;
    if (text !== undefined) result.textContent = text;
    return result;
}

function imageURL(image) {
    return api.apiURL(`/view?${new URLSearchParams({
        filename: image.filename, subfolder: image.subfolder || "", type: image.type || "temp",
    })}`);
}

function imagePanel(image, label) {
    const panel = element("div", "lwb-compare-image");
    const status = element("span", "lwb-compare-placeholder", image ? "Loading…" : `${label}: no image`);
    panel.append(status);
    if (image) {
        const img = element("img", "");
        img.alt = label;
        img.loading = "lazy";
        img.draggable = false;
        img.onload = () => { status.hidden = true; };
        img.onerror = () => {
            img.hidden = true;
            status.hidden = false;
            status.textContent = "Preview unavailable — run this node again";
        };
        img.src = imageURL(image);
        panel.append(img);
    }
    return panel;
}

function sourceLabel(group, index, image, repeated) {
    if (!image) return `${group} — missing`;
    return `${group}${index + 1}${repeated ? " · repeat" : ""} · ${image.width}×${image.height}`;
}

function pairCard(pair, index, positions, selectionGroup, select) {
    const card = element("section", "lwb-compare-card");
    card.setAttribute("aria-label", `Pair ${index + 1}`);
    const heading = element("div", "lwb-compare-pair-title", `Pair ${index + 1}`);
    const labels = element("div", "lwb-compare-labels");
    labels.append(
        element("span", pair.repeatA ? "lwb-compare-repeated" : "", sourceLabel("A", pair.ai, pair.a, pair.repeatA)),
        element("span", pair.repeatB ? "lwb-compare-repeated" : "", sourceLabel("B", pair.bi, pair.b, pair.repeatB)),
    );
    const stage = element("div", "lwb-compare-stage");
    const reference = pair.a || pair.b;
    const aspect = reference ? reference.width / reference.height : 1;
    stage.style.setProperty("--compare-aspect", Math.max(0.65, Math.min(2, aspect)));
    stage.style.setProperty("--compare-split", `${positions[index] ?? 100}%`);
    stage.append(imagePanel(pair.a, `A${pair.ai + 1}`), imagePanel(pair.b, `B${pair.bi + 1}`));
    const divider = element("div", "lwb-compare-divider");
    divider.setAttribute("aria-hidden", "true");
    stage.append(divider);
    const slider = element("input", "lwb-compare-slider");
    slider.type = "range";
    slider.min = "0";
    slider.max = "100";
    slider.value = positions[index] ?? 100;
    slider.setAttribute("aria-label", `A/B split for pair ${index + 1}`);
    slider.disabled = !pair.a || !pair.b;
    if (slider.disabled) card.classList.add("lwb-compare-unpaired");
    function setSplit(value) {
        const split = Math.max(0, Math.min(100, Math.round(value)));
        positions[index] = split;
        slider.value = split;
        stage.style.setProperty("--compare-split", `${split}%`);
    }
    slider.oninput = () => setSplit(Number(slider.value));
    stage.addEventListener("pointermove", event => {
        if (slider.disabled || event.pointerType === "touch" ||
            stage.closest(".lwb-compare")?.dataset.mode !== "slide") return;
        const bounds = stage.getBoundingClientRect();
        if (bounds.width > 0) setSplit((event.clientX - bounds.left) / bounds.width * 100);
    });
    const selection = element("label", "lwb-compare-selection");
    const radio = element("input", "");
    radio.type = "radio";
    radio.name = selectionGroup;
    radio.value = index;
    radio.disabled = !pair.a || !pair.b;
    radio.onchange = () => { if (radio.checked) select(index); };
    selection.append(radio, element("span", "", `Select pair ${index + 1} · index ${index}`));
    card.append(heading, labels, stage, slider, selection);
    return card;
}

export function mountCompare(node) {
    if (node.__lwbCompare) return node.__lwbCompare;
    if (!document.getElementById("lwb-image-compare-style")) {
        const style = document.createElement("link");
        style.id = "lwb-image-compare-style";
        style.rel = "stylesheet";
        style.href = new URL("./image_compare.css", import.meta.url).href;
        document.head.append(style);
    }
    node.properties ||= {};
    const root = element("div", "lwb-compare");
    const toolbar = element("div", "lwb-compare-toolbar");
    const summary = element("div", "lwb-compare-summary");
    summary.setAttribute("aria-live", "polite");
    const modes = element("div", "lwb-compare-modes");
    modes.setAttribute("aria-label", "Comparison view");
    const buttons = [];
    for (const [value, label] of [["side", "Side by side"], ["slide", "Slider"]]) {
        const button = element("button", "", label);
        button.type = "button";
        button.onclick = () => {
            node.properties.lwbCompareMode = value;
            updateMode();
        };
        buttons.push([value, button]);
        modes.append(button);
    }
    const hint = element("div", "lwb-compare-hint");
    const selectionHint = element("div", "lwb-compare-hint");
    selectionHint.setAttribute("aria-live", "polite");
    const grid = element("div", "lwb-compare-grid");
    toolbar.append(summary, modes);
    root.append(toolbar, hint, selectionHint, grid);
    // Native controls own pointer/keyboard input; scrolling the grid must not zoom the graph.
    for (const type of ["pointerdown", "mousedown", "dblclick", "keydown", "wheel"]) {
        root.addEventListener(type, event => event.stopPropagation());
    }
    const positions = {};
    const selectionGroup = `lwb-compare-selection-${++selectionGroupId}`;
    let data = null;
    const inputValue = (name, fallback) => node.widgets?.find(widget => widget.name === name)?.value ?? data?.[name] ?? fallback;
    function updateSelection() {
        const enabled = Boolean(inputValue("selection_enabled", false));
        const selected = Number(inputValue("selected_index", -1));
        root.dataset.selection = String(enabled);
        selectionHint.hidden = !enabled;
        let valid = false;
        for (const radio of grid.querySelectorAll('input[type="radio"]')) {
            radio.checked = enabled && !radio.disabled && Number(radio.value) === selected;
            radio.closest(".lwb-compare-card").classList.toggle("lwb-compare-selected", radio.checked);
            valid ||= radio.checked;
        }
        selectionHint.textContent = valid
            ? `Selected pair ${selected + 1} · index ${selected}. Run the workflow to output this pair.`
            : selected >= 0 ? `Index ${selected} is unavailable. Choose a complete pair or enable repeat.`
                : "Select one pair below, then run the workflow to output its index and images.";
    }
    function select(index) {
        const widget = node.widgets?.find(item => item.name === "selected_index");
        if (!widget) return;
        widget.value = index;
        widget.callback?.(index);
        updateSelection();
        node.setDirtyCanvas?.(true, true);
        node.graph?.setDirtyCanvas?.(true, true);
    }
    function updateMode() {
        root.dataset.mode = node.properties.lwbCompareMode === "slide" ? "slide" : "side";
        for (const [value, button] of buttons) button.setAttribute("aria-pressed", String(value === root.dataset.mode));
    }
    function render() {
        const scroll = grid.scrollTop;
        const repeatWidget = node.widgets?.find(widget => widget.name === "repeat");
        const repeat = Boolean(repeatWidget?.value ?? data?.repeat);
        const a = data?.a || [];
        const b = data?.b || [];
        const pairs = pairImages(a, b, repeat);
        summary.textContent = `A ${a.length}  /  B ${b.length} · ${pairs.length} pairs`;
        hint.textContent = !data ? "Connect image_a and image_b, then run to preview all pairs."
            : !a.length || !b.length ? "One group is empty; there is no last image to repeat."
            : a.length !== b.length ? repeat ? "Repeat on · the shorter group uses its last image."
                : "Counts differ · turn on repeat to fill missing images."
            : "Paired in order · images fit without stretching.";
        grid.replaceChildren(...pairs.map((pair, index) => pairCard(pair, index, positions, selectionGroup, select)));
        grid.scrollTop = scroll;
        updateMode();
        updateSelection();
    }
    const widget = node.addDOMWidget("lwb_comparison", "image_compare", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 360,
        getHeight: () => Math.max(360, (node.size?.[1] || 640) - 166),
    });
    widget.serialize = false;
    const repeatWidget = node.widgets?.find(item => item.name === "repeat");
    if (repeatWidget) {
        const callback = repeatWidget.callback;
        repeatWidget.callback = function () {
            const result = callback?.apply(this, arguments);
            render();
            return result;
        };
    }
    for (const name of ["selection_enabled", "selected_index"]) {
        const input = node.widgets?.find(item => item.name === name);
        if (!input) continue;
        const callback = input.callback;
        input.callback = function () {
            const result = callback?.apply(this, arguments);
            updateSelection();
            return result;
        };
    }
    const state = {
        root,
        render,
        setData(value) {
            if (!value || !Array.isArray(value.a) || !Array.isArray(value.b)) return;
            data = value;
            node.properties.lwbComparePreview = value;
            render();
        },
    };
    node.__lwbCompare = state;
    state.setData(node.properties.lwbComparePreview);
    render();
    return state;
}

app.registerExtension({
    name: "LlamaWorkbench.BatchImageCompare",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== COMPARE_NODE) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            mountCompare(this);
            this.setSize?.([Math.max(640, this.size?.[0] || 0), Math.max(640, this.size?.[1] || 0)]);
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const result = executed?.apply(this, arguments);
            mountCompare(this).setData(output?.comparison?.[0]);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = configured?.apply(this, arguments);
            // Older saved workflows had no output sockets on this preview node.
            for (const [index, [name, type]] of [["index", "INT"], ["selected_a", "IMAGE"], ["selected_b", "IMAGE"]].entries()) {
                if (!this.outputs?.[index]) this.addOutput?.(name, type);
            }
            const view = mountCompare(this);
            view.setData(this.properties?.lwbComparePreview);
            view.render();
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.__lwbCompare?.root.remove();
            delete this.__lwbCompare;
            return removed?.apply(this, arguments);
        };
    },
});
