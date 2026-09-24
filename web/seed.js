import { app } from "../../scripts/app.js";

const SEED_NODE = "LlamaWorkbench_Seed";
const SEED_MAX = 0x7fffffff;
const SEED_MODES = new Set(["fixed", "increment", "decrement", "randomize"]);

function findSeedWidget(node) {
  return node.widgets?.find((widget) => widget.name === "seed");
}

function findControlWidget(node, seedWidget) {
  const linkedControl = seedWidget?.linkedWidgets?.find((widget) =>
    SEED_MODES.has(widget.value),
  );
  if (linkedControl) return linkedControl;

  return node.widgets?.find((widget) => {
    const name = String(widget.name || "");
    return (
      name === "control_after_generate" ||
      name.endsWith("control_after_generate")
    );
  });
}

function markChanged(node) {
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
}

function setSeed(node, value) {
  const widget = findSeedWidget(node);
  if (!widget) return;

  widget.value = Math.max(0, Math.min(SEED_MAX, Math.trunc(value)));
  widget.callback?.(widget.value);
  markChanged(node);
}

function setMode(node, mode) {
  const seedWidget = findSeedWidget(node);
  const controlWidget = findControlWidget(node, seedWidget);
  if (!controlWidget || !SEED_MODES.has(mode)) return;

  controlWidget.value = mode;
  controlWidget.callback?.(mode);
  markChanged(node);
}

function currentSeed(node) {
  const value = Number(findSeedWidget(node)?.value);
  return Number.isFinite(value) ? Math.trunc(value) : 0;
}

function randomSeed() {
  if (globalThis.crypto?.getRandomValues) {
    const value = new Uint32Array(1);
    globalThis.crypto.getRandomValues(value);
    return value[0] & SEED_MAX;
  }
  return Math.floor(Math.random() * (SEED_MAX + 1));
}

function addButton(node, label, tooltip, callback) {
  return node.addWidget("button", label, null, callback, {
    serialize: false,
    tooltip,
  });
}

function addSeedControls(node) {
  if (node.__lwbSeedControlsAdded) return;
  node.__lwbSeedControlsAdded = true;

  addButton(
    node,
    "🎲 New Fixed Random",
    "Generate a new random seed immediately and keep it fixed.",
    () => {
      setSeed(node, randomSeed());
      setMode(node, "fixed");
    },
  );
  addButton(
    node,
    "🎲 Randomize Each Run",
    "Generate a random seed now and randomize it after every queued run.",
    () => {
      setSeed(node, randomSeed());
      setMode(node, "randomize");
    },
  );
  addButton(
    node,
    "+ Increment Each Run",
    "Increment the displayed seed now and after every queued run.",
    () => {
      setSeed(node, (currentSeed(node) + 1) % (SEED_MAX + 1));
      setMode(node, "increment");
    },
  );
  addButton(
    node,
    "- Decrement Each Run",
    "Decrement the displayed seed now and after every queued run.",
    () => {
      setSeed(node, (currentSeed(node) - 1 + SEED_MAX + 1) % (SEED_MAX + 1));
      setMode(node, "decrement");
    },
  );
  addButton(
    node,
    "🔒 Keep Fixed",
    "Keep the currently displayed seed fixed.",
    () => setMode(node, "fixed"),
  );

  const computedSize = node.computeSize?.();
  if (computedSize && node.setSize) {
    node.setSize([
      Math.max(node.size?.[0] || 0, computedSize[0]),
      Math.max(node.size?.[1] || 0, computedSize[1]),
    ]);
  }
}

app.registerExtension({
  name: "LlamaWorkbench.SeedControls",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== SEED_NODE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      addSeedControls(this);
      return result;
    };
  },
});
