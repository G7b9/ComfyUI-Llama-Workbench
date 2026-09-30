"""Exercise the actual browser execution callback with a minimal Node.js host."""

import json
from pathlib import Path
import subprocess

import pytest

from lwb.nodes import LlamaWorkbenchChat


CHAT_JS = Path(__file__).resolve().parents[1] / "web" / "chat.js"


def run_frontend(script, data):
    source = "\n".join(
        line for line in CHAT_JS.read_text().splitlines() if not line.startswith("import ")
    )
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            "const app = {registerExtension() {}}; const api = {};\n" + source + "\n" + script,
            json.dumps(data),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_workflow_thinking_change_resubmits_the_original_message():
    class Backend:
        label = "fake"

        def __init__(self):
            self.calls = []

        def chat(self, messages, **settings):
            self.calls.append((messages, settings))
            return "answer"

    backend = Backend()
    chat = LlamaWorkbenchChat()
    first_inputs = dict(
        lwb_user_message="Explain this result",
        lwb_history_json="[]",
        lwb_flow_state_json="{}",
        lwb_request_id="same-request",
        use_cache=True,
        thinking="off",
        clear_context_before_run=True,
    )
    first = chat.chat(backend, **first_inputs)
    next_inputs = run_frontend(
        """
const {inputs, output} = JSON.parse(process.argv[1]);
const node = {widgets: Object.entries(inputs).map(([name, value]) => ({name, value}))};
onChatExecuted(node, output);
findWidget(node, "thinking").value = "on";
console.log(JSON.stringify(Object.fromEntries(node.widgets.map(w => [w.name, w.value]))));
""",
        {"inputs": first_inputs, "output": first["ui"]},
    )
    assert next_inputs["lwb_user_message"] == first_inputs["lwb_user_message"]
    assert next_inputs["lwb_request_id"] == first_inputs["lwb_request_id"]
    chat.chat(backend, **next_inputs)
    assert len(backend.calls) == 2
    assert [settings["enable_thinking"] for _, settings in backend.calls] == [False, True]
    assert backend.calls[0][0] == backend.calls[1][0]


@pytest.mark.parametrize("busy", [False, True])
@pytest.mark.parametrize("clear_before", [False, True])
def test_executed_and_cached_events_preserve_input_until_explicit_clear(busy, clear_before):
    values = run_frontend(
        """
const {busy, clearBefore} = JSON.parse(process.argv[1]);
const node = {__lwbChatBusy: busy, widgets: [
    {name: "lwb_user_message", value: "original prompt"},
    {name: "clear_context_before_run", value: clearBefore},
    {name: "lwb_history_json", value: "[]"},
    {name: "lwb_flow_state_json", value: "{}"},
]};
const history = JSON.stringify([{role: "user", content: "original prompt"}, {role: "assistant", content: "answer"}]);
const output = {sent: [true], history_json: [history], flow_state_json: ['{"stage":"complete"}']};
onChatExecuted(node, output);
const original = findWidget(node, "lwb_user_message").value;
// A cached result arriving after an edit must not erase the next message.
findWidget(node, "lwb_user_message").value = "edited prompt";
onChatExecuted(node, output);
const edited = findWidget(node, "lwb_user_message").value;
clearInput(node);
console.log(JSON.stringify({
    messages: [original, edited, findWidget(node, "lwb_user_message").value],
    history: JSON.parse(findWidget(node, "lwb_history_json").value),
    flow: JSON.parse(findWidget(node, "lwb_flow_state_json").value),
}));
""",
        {"busy": busy, "clearBefore": clear_before},
    )
    assert values["messages"] == ["original prompt", "edited prompt", ""]
    if clear_before:
        assert values["history"] == []
        assert values["flow"] == {}
    else:
        assert values["history"] == [
            {"role": "user", "content": "original prompt"},
            {"role": "assistant", "content": "answer"},
        ]
        assert values["flow"] == {"stage": "complete"}
