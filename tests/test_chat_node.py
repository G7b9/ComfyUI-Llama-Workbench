from __future__ import annotations

import json
import inspect
import math
import sys
import types
from pathlib import Path

import pytest

import lwb.nodes as nodes_module
from lwb.media import image_tensor_to_data_urls
from lwb.backend import BackendError, ChatResponse, EmbeddedBackend, ServerBackend
from lwb.nodes import (
    LlamaWorkbenchChat,
    LlamaWorkbenchChatDisplay,
    LlamaWorkbenchChatSettings,
    LlamaWorkbenchPrompt,
    LlamaWorkbenchSkillLoader,
    LlamaWorkbenchStartServer,
)


class FakeBackend:
    label = "fake"

    def __init__(self):
        self.calls = []

    def chat(self, messages, **settings):
        self.calls.append((messages, settings))
        return "hello from local model"


def test_chat_persists_history_without_comfyui_runtime():
    node = LlamaWorkbenchChat()
    backend = FakeBackend()
    result = node.chat(
        backend,
        "hello",
        "[]",
        "{}",
        "request-1",
        max_tokens=32,
        seed=1234,
        settings={"max_history_messages": 3, "max_tokens": 32},
    )
    reply, history_raw, state_raw, thinking = result["result"]
    assert reply == "hello from local model"
    assert thinking == ""
    assert [item["role"] for item in json.loads(history_raw)] == ["user", "assistant"]
    assert json.loads(state_raw)["stage"] == "not started"
    assert backend.calls[0][1]["max_tokens"] == 32
    assert backend.calls[0][1]["seed"] == 1234
    assert backend.calls[0][1]["enable_thinking"] is False
    assert json.loads(result["ui"]["context_state_json"][0])["estimated_tokens"] > 0


def test_chat_context_meter_uses_owned_server_context_size():
    backend = FakeBackend()
    backend.context_size = 1024
    result = LlamaWorkbenchChat().chat(backend, "你好", "[]", "{}", "request-1")
    state = json.loads(result["ui"]["context_state_json"][0])
    assert state["context_size"] == 1024
    assert state["estimated_tokens"] > 0
    assert state["usage_percent"] is not None


def test_chat_settings_own_generation_values_when_connected():
    backend = FakeBackend()
    LlamaWorkbenchChat().chat(
        backend,
        "hello",
        "[]",
        "{}",
        "request-1",
        max_tokens=32,
        seed=11,
        settings={"max_tokens": 96, "seed": 22},
    )
    assert backend.calls[0][1]["max_tokens"] == 96
    assert backend.calls[0][1]["seed"] == 22


def test_chat_clears_persisted_context_by_default_but_can_continue_when_disabled():
    node = LlamaWorkbenchChat()
    backend = FakeBackend()
    existing_history = json.dumps([
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
    ])

    fresh = node.chat(backend, "new question", existing_history, '{"stage":"complete"}', "request-1")
    fresh_history = json.loads(fresh["result"][1])
    assert [item["content"] for item in fresh_history] == ["new question", "hello from local model"]

    continued = node.chat(
        backend,
        "another question",
        existing_history,
        '{"stage":"complete"}',
        "request-2",
        clear_context_before_run=False,
    )
    continued_history = json.loads(continued["result"][1])
    assert [item["content"] for item in continued_history[:2]] == ["old question", "old answer"]


def test_chat_uses_namespaced_hidden_state_widgets_and_safe_frontend_hiding():
    required = LlamaWorkbenchChat.INPUT_TYPES()["required"]
    assert {"lwb_user_message", "lwb_history_json", "lwb_flow_state_json", "lwb_request_id"} <= set(required)
    assert required["max_tokens"][1]["default"] == -1
    assert required["seed"][1]["default"] == -1
    assert required["use_cache"][1]["default"] is True
    assert required["thinking"][1]["default"] == "off"
    assert required["clear_context_before_run"][1]["default"] is True
    assert required["release_comfy_cache_after_run"][1]["default"] is True
    assert required["release_owned_server_after_run"][1]["default"] is True
    assert required["lwb_history_json"][1]["hidden"] is True
    assert required["lwb_flow_state_json"][1]["hidden"] is True
    assert required["lwb_request_id"][1]["hidden"] is True
    assert "multiline" not in required["lwb_user_message"][1]
    assert "multiline" not in required["lwb_history_json"][1]
    assert "multiline" not in required["lwb_flow_state_json"][1]
    source = (Path(__file__).resolve().parent.parent / "web" / "chat.js").read_text(encoding="utf-8")
    assert "converted-widget:lwb" not in source
    assert "Object.defineProperty" not in source
    assert "addDOMWidget" not in source
    assert "widget.hidden = true" in source
    assert "widget.options.hidden = true" in source
    assert '"max_tokens",' in source
    assert '"seed",' in source
    assert '"use_cache",' in source
    assert "addDOMWidget" not in source
    assert "api.queuePrompt" in source
    assert "app.graphToPrompt" in source
    assert "collectPromptLinks" in source
    assert "const CHAT_LAYOUT_VERSION = 7" in source
    assert "function initializeChatSize" in source
    assert "function nativeWidgetBottom" in source
    assert "function setScrollFromTrack" in source
    assert "function handlePanelDrag" in source
    assert "function contextSizeFromLinkedBackend" in source
    assert "function contextLimit" in source
    assert "const bubbleTextWidth = Math.max(24, bubbleWidth - 18);" in source
    assert "const MESSAGE_RIGHT_GUTTER = 18;" in source
    assert "node.widgets_start_y =" not in source
    output_source = (Path(__file__).resolve().parent.parent / "web" / "chat_output.js").read_text(encoding="utf-8")
    assert "function setOutputScrollFromTrack" in output_source
    assert "function startOutputDrag" in output_source
    assert "function dragOutput" in output_source
    assert "app.canvas?.graph_mouse" in output_source


def test_chat_is_a_terminal_node_and_large_model_startup_default_is_longer():
    assert LlamaWorkbenchChat.OUTPUT_NODE is True
    optional = LlamaWorkbenchStartServer.INPUT_TYPES()["optional"]
    assert optional["wait_seconds"][1]["default"] == 600.0
    assert optional["timeout_seconds"][1]["default"] == 120.0
    assert optional["release_comfy_models"][1]["default"] is True
    assert optional["cleanup_previous_server"][1]["default"] is True
    connection = nodes_module.LlamaWorkbenchConnection.INPUT_TYPES()["required"]
    assert connection["use_environment_proxy"][1]["default"] is False


def test_start_server_can_release_comfy_managed_models_before_loading(monkeypatch):
    calls = []
    comfy_module = types.ModuleType("comfy")
    management_module = types.ModuleType("comfy.model_management")
    management_module.unload_all_models = lambda: calls.append("unload")
    management_module.soft_empty_cache = lambda: calls.append("empty")
    comfy_module.model_management = management_module
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management_module)

    assert nodes_module._release_comfy_model_cache() is True
    assert calls == ["unload", "empty"]


def test_chat_schedules_full_comfy_cache_release_after_the_current_workflow(monkeypatch):
    calls = []
    server_module = types.ModuleType("server")

    class Queue:
        @staticmethod
        def set_flag(name, value):
            calls.append((name, value))

    class PromptServer:
        instance = types.SimpleNamespace(prompt_queue=Queue())

    server_module.PromptServer = PromptServer
    monkeypatch.setitem(sys.modules, "server", server_module)

    assert nodes_module._schedule_comfy_cache_release_after_run(True) is True
    assert calls == [("unload_models", True), ("free_memory", True)]
    assert nodes_module._schedule_comfy_cache_release_after_run(False) is False


def test_prompt_defaults_to_unlimited_tokens_with_thinking_disabled():
    required = LlamaWorkbenchPrompt.INPUT_TYPES()["required"]
    assert required["max_tokens"][1]["default"] == -1
    assert required["max_tokens"][1]["min"] == -1
    assert required["thinking"][1]["default"] == "off"
    assert required["seed"][1]["default"] == -1
    assert required["max_images"][1]["default"] == 8
    assert required["max_image_edge"][1]["default"] == 0
    assert required["auto_unload"][1]["default"] is False
    assert inspect.signature(image_tensor_to_data_urls).parameters["max_edge"].default == 0
    optional_images = LlamaWorkbenchPrompt.INPUT_TYPES()["optional"]
    assert {"image", *(f"image{index}" for index in range(1, 9))} <= set(optional_images)
    settings = LlamaWorkbenchChatSettings.INPUT_TYPES()["required"]
    assert settings["max_tokens"][1]["default"] == -1
    assert settings["seed"][1]["default"] == -1
    assert settings["max_image_edge"][1]["default"] == 0
    assert LlamaWorkbenchChatSettings().build(seed=-1)[0]["seed"] == -1


def test_chat_exposes_dynamic_image_inputs_and_direct_edge_limit():
    required = LlamaWorkbenchChat.INPUT_TYPES()["required"]
    optional = LlamaWorkbenchChat.INPUT_TYPES()["optional"]
    assert required["max_images"][1]["default"] == 8
    assert required["max_image_edge"][1]["default"] == 0
    assert required["auto_unload"][1]["default"] is False
    assert {"image", *(f"image{index}" for index in range(1, 9))} <= set(optional)
    assert LlamaWorkbenchChatSettings.IS_CHANGED(system_prompt="") is False
    assert LlamaWorkbenchChat.IS_CHANGED() is False
    assert math.isnan(LlamaWorkbenchChat.IS_CHANGED(use_cache=False))
    assert math.isnan(LlamaWorkbenchChat.IS_CHANGED(settings={"use_cache": False}))


def test_chat_separates_structured_reasoning_from_the_answer_and_history():
    class ReasoningBackend:
        label = "fake"

        def chat_response(self, messages, **settings):
            return ChatResponse(content="formal answer", reasoning="first consider the request")

    result = LlamaWorkbenchChat().chat(ReasoningBackend(), "hello", "[]", "{}", "request-1", thinking="on")
    answer, history_raw, _, thinking = result["result"]
    assert answer == "formal answer"
    assert thinking == "first consider the request"
    assert json.loads(history_raw)[-1] == {"role": "assistant", "content": "formal answer"}


def test_chat_recovers_a_final_answer_after_a_thinking_only_response():
    class InlineBackend:
        label = "fake"

        def __init__(self):
            self.settings = []

        def chat(self, messages, **settings):
            self.settings.append(settings)
            if settings["enable_thinking"]:
                return "Here's a thinking process:\nwork through the details"
            return "formal answer after thinking"

    backend = InlineBackend()
    result = LlamaWorkbenchChat().chat(backend, "hello", "[]", "{}", "request-1", thinking="on")
    answer, _, _, thinking = result["result"]
    assert answer == "formal answer after thinking"
    assert thinking == "work through the details"
    assert [settings["enable_thinking"] for settings in backend.settings] == [True, False]


def test_chat_display_node_accepts_and_returns_separate_reasoning_and_answer():
    required = LlamaWorkbenchChatDisplay.INPUT_TYPES()["required"]
    assert required["thinking"][1]["forceInput"] is True
    assert required["assistant_message"][1]["forceInput"] is True
    result = LlamaWorkbenchChatDisplay().display("reasoning", "answer")
    assert result["result"] == ("reasoning", "answer")
    assert result["ui"]["thinking"] == ["reasoning"]


def test_prompt_passes_seed_to_the_backend():
    class PromptBackend:
        label = "fake"

        def __init__(self):
            self.settings = None

        def chat(self, messages, **settings):
            self.settings = settings
            return "prompt result"

    backend = PromptBackend()
    response, _ = LlamaWorkbenchPrompt().generate(
        backend,
        "describe",
        "",
        -1,
        0.2,
        0.9,
        40,
        thinking="off",
        seed=9876,
        max_images=3,
    )
    assert response == "prompt result"
    assert backend.settings["max_tokens"] == -1
    assert backend.settings["seed"] == 9876
    assert backend.settings["enable_thinking"] is False


def test_auto_unload_runs_after_prompt_and_chat_generation(monkeypatch):
    calls = []
    monkeypatch.setattr(nodes_module, "_auto_unload_backend", lambda backend, enabled: calls.append((backend, enabled)) or True)

    class LocalBackend:
        label = "fake"

        def chat(self, messages, **settings):
            return "result"

    backend = LocalBackend()
    LlamaWorkbenchPrompt().generate(backend, "describe", "", -1, 0.2, 0.9, 40, auto_unload=True)
    # Chat's GPU-handoff guard is on by default even if the legacy
    # auto_unload widget in an existing workflow is false.
    LlamaWorkbenchChat().chat(backend, "hello", "[]", "{}", "request-1", auto_unload=False)
    assert calls == [(backend, True), (backend, True)]


def test_chat_releases_owned_resources_when_generation_fails(monkeypatch):
    releases = []
    cache_releases = []
    monkeypatch.setattr(nodes_module, "_auto_unload_backend", lambda backend, enabled: releases.append((backend, enabled)) or True)
    monkeypatch.setattr(
        nodes_module,
        "_schedule_comfy_cache_release_after_run",
        lambda enabled: cache_releases.append(enabled) or True,
    )

    class FailingBackend:
        label = "fake"

        def chat(self, messages, **settings):
            raise BackendError("HTTP 502")

    backend = FailingBackend()
    with pytest.raises(BackendError, match="HTTP 502"):
        LlamaWorkbenchChat().chat(backend, "hello", "[]", "{}", "request-1")

    assert releases == [(backend, True)]
    assert cache_releases == [False]


def test_chat_cache_toggle_controls_comfy_cache_release(monkeypatch):
    cache_releases = []
    monkeypatch.setattr(
        nodes_module,
        "_schedule_comfy_cache_release_after_run",
        lambda enabled: cache_releases.append(enabled) or True,
    )
    backend = FakeBackend()

    LlamaWorkbenchChat().chat(
        backend,
        "cached request",
        "[]",
        "{}",
        "request-1",
        settings={"use_cache": True},
    )
    LlamaWorkbenchChat().chat(
        backend,
        "fresh request",
        "[]",
        "{}",
        "request-2",
        settings={"use_cache": False},
    )

    assert cache_releases == [False, True]


def test_auto_unload_only_releases_workbench_owned_backends(monkeypatch):
    stops = []
    embedded_releases = []
    monkeypatch.setattr(nodes_module.OWNED_SERVER, "stop", lambda: stops.append(True) or True)
    monkeypatch.setattr(nodes_module.EMBEDDED_MODELS, "release", lambda backend: embedded_releases.append(backend) or True)

    attached = ServerBackend("http://127.0.0.1:8080")
    owned = ServerBackend("http://127.0.0.1:8080", owned_by_workbench=True)
    embedded = EmbeddedBackend(llm=object())
    assert nodes_module._auto_unload_backend(attached, True) is False
    assert nodes_module._auto_unload_backend(owned, True) is True
    assert nodes_module._auto_unload_backend(embedded, True) is True
    assert stops == [True]
    assert embedded_releases == [embedded]
    assert LlamaWorkbenchStartServer.IS_CHANGED() is False


def test_cached_start_server_backend_restarts_only_when_a_new_request_needs_it(monkeypatch):
    backend = ServerBackend("http://127.0.0.1:8080", owned_by_workbench=True)
    starts = []
    releases = []

    def start_server(config, wait_seconds, cleanup_previous_server, timeout_seconds=120.0):
        starts.append((config, wait_seconds, cleanup_previous_server, timeout_seconds))
        backend.timeout_seconds = timeout_seconds
        return backend

    monkeypatch.setattr(nodes_module.OWNED_SERVER, "start", start_server)
    monkeypatch.setattr(nodes_module.OWNED_SERVER, "is_running_for", lambda config: False)
    monkeypatch.setattr(nodes_module, "_release_comfy_model_cache", lambda: releases.append(True) or True)

    LlamaWorkbenchStartServer().start(
        model_path="model.gguf",
        binary_path="llama-server",
        host="127.0.0.1",
        port=8080,
        context_size=8192,
        gpu_layers=-1,
        extra_args="",
        timeout_seconds=600.0,
        release_comfy_models=True,
    )
    backend.ensure_available()

    assert len(starts) == 2
    assert [call[3] for call in starts] == [600.0, 600.0]
    assert backend.timeout_seconds == 600.0
    assert releases == [True, True]


def test_auto_unload_uses_ownership_marker_after_a_custom_node_reload(monkeypatch):
    stops = []
    monkeypatch.setattr(nodes_module.OWNED_SERVER, "stop", lambda: stops.append(True) or True)

    class ReloadedOwnedBackend:
        # Simulates an object created by a previous import of backend.py, for
        # which isinstance(current ServerBackend) is intentionally false.
        owned_by_workbench = True

    assert nodes_module._auto_unload_backend(ReloadedOwnedBackend(), True) is True
    assert stops == [True]


def test_prompt_sends_an_image_batch_up_to_the_selected_limit(monkeypatch):
    class PromptBackend:
        label = "fake"

        def __init__(self):
            self.messages = None

        def chat(self, messages, **settings):
            self.messages = messages
            return "prompt result"

    calls = []

    def encode_images(image, max_images, max_edge):
        calls.append((max_images, max_edge))
        return [f"data:image/jpeg;base64,image-{index}" for index in range(max_images)]

    monkeypatch.setattr(nodes_module, "image_tensor_to_data_urls", encode_images)
    backend = PromptBackend()
    LlamaWorkbenchPrompt().generate(backend, "describe", "", -1, 0.2, 0.9, 40, max_images=3, image=object())

    assert calls == [(3, 0)]
    assert len(backend.messages[-1]["content"]) == 4


def test_prompt_collects_multiple_dynamic_image_inputs_in_order(monkeypatch):
    class PromptBackend:
        label = "fake"

        def __init__(self):
            self.messages = None

        def chat(self, messages, **settings):
            self.messages = messages
            return "prompt result"

    first, second = object(), object()
    calls = []

    def encode_images(image, max_images, max_edge):
        calls.append((image, max_images, max_edge))
        return [f"data:image/jpeg;base64,{len(calls)}"]

    monkeypatch.setattr(nodes_module, "image_tensor_to_data_urls", encode_images)
    backend = PromptBackend()
    LlamaWorkbenchPrompt().generate(
        backend, "describe", "", -1, 0.2, 0.9, 40, max_images=3, image1=first, image2=second
    )

    assert calls == [(first, 3, 0), (second, 2, 0)]
    assert len(backend.messages[-1]["content"]) == 3


def test_chat_collects_multiple_dynamic_image_inputs_with_the_direct_edge_limit(monkeypatch):
    class ChatBackend:
        label = "fake"

        def __init__(self):
            self.messages = None

        def chat(self, messages, **settings):
            self.messages = messages
            return "chat result"

    first, second = object(), object()
    calls = []

    def encode_images(image, max_images, max_edge):
        calls.append((image, max_images, max_edge))
        return [f"data:image/jpeg;base64,{len(calls)}"]

    monkeypatch.setattr(nodes_module, "image_tensor_to_data_urls", encode_images)
    backend = ChatBackend()
    result = LlamaWorkbenchChat().chat(
        backend,
        "describe these",
        "[]",
        "{}",
        "request-1",
        max_images=2,
        max_image_edge=768,
        settings={"max_image_edge": 1280},
        image1=first,
        image2=second,
    )

    assert result["result"][0] == "chat result"
    assert calls == [(first, 2, 768), (second, 1, 768)]
    assert len(backend.messages[-1]["content"]) == 3


def test_prompt_image_frontend_uses_native_dynamic_slots_without_dom_widgets():
    source = (Path(__file__).resolve().parent.parent / "web" / "prompt_images.js").read_text(encoding="utf-8")
    assert "onConnectionsChange" in source
    assert 'node.addInput(name, "IMAGE")' in source
    assert "`image${index}`" in source
    assert 'const CHAT_NODE = "LlamaWorkbench_Chat";' in source
    assert "DYNAMIC_IMAGE_NODES" in source
    assert "addDOMWidget" not in source


def test_chat_frontend_preserves_the_current_node_size_and_output_display_is_canvas_only():
    chat_source = (Path(__file__).resolve().parent.parent / "web" / "chat.js").read_text(encoding="utf-8")
    display_source = (Path(__file__).resolve().parent.parent / "web" / "chat_output.js").read_text(encoding="utf-8")
    assert "drawConversation" in chat_source
    assert "handlePanelScroll" in chat_source
    assert "clearContext" in chat_source
    assert "clearInput" in chat_source
    assert "contextSummary" in chat_source
    assert "context_state_json" in chat_source
    assert "onDrawForeground" in display_source
    assert "onMouseWheel" in display_source
    assert "__lwbThinkingScroll" in display_source
    assert "ctx.clip" in display_source
    assert "addDOMWidget" not in display_source
