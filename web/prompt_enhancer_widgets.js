// Older PE exports predate thinking and task-specific prompt widgets.
// Restore by saved names before LiteGraph applies positional widgets_values.
export function migratePromptEnhancerWidgets(info, widgets) {
    if (!Array.isArray(info?.widgets_values) || !Array.isArray(info?.inputs)) return info;
    const savedNames = [];
    for (const input of info.inputs) {
        if (!input.widget?.name) continue;
        savedNames.push(input.widget.name);
        if (input.widget.name === "seed") savedNames.push("control_after_generate");
    }
    // Unknown serialization layouts retain LiteGraph's normal handling.
    if (!savedNames.length || savedNames.length !== info.widgets_values.length) return info;
    const saved = new Map(savedNames.map((name, index) => [name, info.widgets_values[index]]));
    return {
        ...info,
        widgets_values: widgets.filter((widget) => widget.serialize !== false).map((widget) =>
            saved.has(widget.name) ? saved.get(widget.name) : widget.value),
    };
}
