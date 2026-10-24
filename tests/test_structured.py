from copy import deepcopy
import itertools
from types import SimpleNamespace

from jsonschema import Draft202012Validator
import pytest

from lwb.structured import (
    StructuredOutputError,
    check_local_schema,
    finite_keys,
    prepare,
    validate_response,
)


def payload(schema, **kwargs):
    return dict(
        messages=[{"role": "user", "content": "synthetic"}],
        max_tokens=20,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "test", "strict": False, "schema": schema},
        },
        **kwargs,
    )


def keyed(**kwargs):
    return dict(
        type="object",
        propertyNames={"enum": ["first", "last"]},
        additionalProperties={"type": ["integer", "null"]},
        **kwargs,
    )


def schema_of(wire):
    return wire["response_format"]["json_schema"]["schema"]


@pytest.mark.parametrize("additional", [True, False, {"type": ["integer", "null"]}])
@pytest.mark.parametrize("required", [[], ["first"]])
def test_equivalence_and_optional_missing_null(additional, required):
    schema = keyed(
        required=required,
        properties={"first": {"type": "integer"}},
        minProperties=0,
        maxProperties=2,
    )
    schema["additionalProperties"] = additional
    before = deepcopy(schema)
    converted, notes = finite_keys(schema)
    assert schema == before and notes
    assert converted["required"] == required
    assert converted["minProperties"] == 0 and converted["maxProperties"] == 2
    assert converted["additionalProperties"] is False
    for present in itertools.product([False, True], repeat=3):
        for values in itertools.product([None, 1, "bad"], repeat=3):
            instance = {
                k: v for k, v, yes in zip(["first", "last", "other"], values, present) if yes
            }
            assert Draft202012Validator(schema).is_valid(instance) == Draft202012Validator(
                converted
            ).is_valid(instance), instance


def test_nested_definitions_arrays_and_data_are_not_keywords():
    data = {"propertyNames": {"enum": ["ordinary", "data"]}}
    schema = {
        "type": "array",
        "items": {"$ref": "#/$defs/Frame"},
        "$defs": {
            "Frame": {
                "type": "object",
                "properties": {
                    "propertyNames": {"enum": [data], "default": data, "examples": [data]},
                    "keyframes": keyed(),
                },
                "additionalProperties": False,
            }
        },
    }
    original = payload(schema)
    before = deepcopy(original)
    wire, info = prepare(original)
    assert original == before and wire is not original
    assert info["strategy"] == "equivalent_transform"
    assert (
        schema_of(wire)["$defs"]["Frame"]["properties"]["propertyNames"]
        == schema["$defs"]["Frame"]["properties"]["propertyNames"]
    )
    assert schema_of(wire)["$defs"]["Frame"]["properties"]["keyframes"]["properties"]["last"] == {
        "type": ["integer", "null"]
    }
    assert wire["response_format"]["json_schema"]["strict"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"required": ["other"]},
        {"required": ["last"], "additionalProperties": False},
        {"minProperties": 3},
        {"maxProperties": 0, "required": ["first"]},
        {"propertyNames": {"pattern": "^x"}},
        {"allOf": [{}]},
        {"patternProperties": {".*": {}}},
        {"unevaluatedProperties": False},
        {"propertyNames": {"enum": ["first"], "minLength": 2}},
    ],
)
def test_conflicts_and_complex_rules_fail_explicitly(change):
    schema = keyed()
    schema.update(change)
    with pytest.raises(StructuredOutputError) as caught:
        prepare(payload(schema), backend="ninfer")
    assert caught.value.detail["schema_path"] == "/propertyNames"
    assert caught.value.detail["backend"] == "ninfer"
    assert caught.value.detail["strategy"] == "unsupported"


def test_additional_false_keeps_only_existing_allowed_names():
    schema = keyed(properties={"first": {"type": "integer"}, "other": {}}, required=[])
    schema["additionalProperties"] = False
    wire, _ = prepare(payload(schema))
    assert schema_of(wire)["properties"] == {"first": {"type": "integer"}}


def test_post_validate_is_explicit_and_preserves_original():
    schema = keyed(propertyNames_unused="annotation")
    # Unknown generation constraints are refused, never recursively removed.
    with pytest.raises(StructuredOutputError):
        prepare(payload(schema))
    original = payload(schema, structured_output_mode="post_validate")
    before = deepcopy(original)
    wire, info = prepare(original)
    assert original == before and schema_of(wire) == {}
    assert "structured_output_mode" not in wire
    assert info["strategy"] == "post_validate"
    assert info["original_schema_sha256"] != info["generation_schema_sha256"]


def test_capability_policy_and_native_property_names():
    request = payload(keyed(minProperties=1))
    with pytest.raises(StructuredOutputError) as caught:
        prepare(request)
    assert caught.value.detail["keyword"] == "minProperties"
    wire, info = prepare(
        request, keywords=["propertyNames", "minProperties"], binary_sha256="a" * 64
    )
    assert schema_of(wire) == schema_of(request)
    assert info["strategy"] == "backend_direct"
    from lwb.scheduler import ManagedServerProfile

    with pytest.raises(ValueError, match="pinned"):
        ManagedServerProfile("binary", "model", structured_output_keywords=["propertyNames"])
    profile = ManagedServerProfile(
        "binary", "model", binary_sha256="a" * 64, structured_output_keywords=["propertyNames"]
    )
    assert profile.structured_output_keywords == ("propertyNames",)


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "https://example.com/schema"},
        {"$id": "https://example.com"},
        {"$ref": "#/$defs/missing"},
        {"$ref": "#/properties/first"},
        {"$schema": "http://json-schema.org/draft-07/schema#"},
    ],
)
def test_no_remote_resolution_or_relocated_references(schema):
    with pytest.raises(StructuredOutputError):
        check_local_schema(schema)


@pytest.mark.parametrize(
    "content,reason",
    [
        ('{"other": 1}', "stop"),
        ('{"first": "bad"}', "stop"),
        ('{"first": NaN}', "stop"),
        ('{"first":1,"first":2}', "stop"),
        ("```json\n{}\n```", "stop"),
        ("{}", "length"),
        ("null", "stop"),
    ],
)
def test_output_validated_against_original(content, reason):
    with pytest.raises(StructuredOutputError) as caught:
        validate_response(
            payload(keyed()),
            {"choices": [{"message": {"content": content}, "finish_reason": reason}]},
        )
    assert caught.value.detail["stage"] == "output_validation"
    assert "bad" not in str(caught.value)


@pytest.mark.parametrize("content", ["{}", '{"first":1}', '{"last":null}', '{"first":1,"last":2}'])
def test_valid_results(content):
    validate_response(
        payload(keyed()), {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    )


def test_preflight_count_and_inference_use_one_wire_schema():
    from lwb.scheduler import Scheduler

    calls = []

    class Session:
        def get(self, url, **kwargs):
            value = (
                {"backend": "ninfer", "default_generation_settings": {"n_ctx": 8192}}
                if url.endswith("/props")
                else {"data": [{"id": "test"}]}
            )
            return SimpleNamespace(ok=True, json=lambda: value)

        def post(self, url, json, **kwargs):
            calls.append((url, deepcopy(json)))
            if url.endswith("/apply-template"):
                value = {"prompt": str(json["response_format"])}
            elif url.endswith("/tokenize"):
                assert json["content"] == str(calls[0][1]["response_format"])
                value = {"tokens": [1, 2, 3]}
            else:
                value = {"choices": []}
            return SimpleNamespace(ok=True, json=lambda: value)

    scheduler = object.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(config=None)
    backend = SimpleNamespace(
        _session=Session(), server_url="http://synthetic", context_size=8192, timeout_seconds=1
    )
    original = payload(keyed())
    before = deepcopy(original)
    _, info = scheduler.infer(backend, original)
    assert original == before
    assert calls[0][1]["response_format"] == calls[2][1]["response_format"]
    assert "propertyNames" not in schema_of(calls[2][1])
    assert info["input_tokens"] == 3
    assert info["structured_output"]["strategy"] == "equivalent_transform"
    assert len(calls) == 3


def test_min_max_property_constraints_not_dropped():
    original = payload(keyed(minProperties=1, maxProperties=2))
    wire, _ = prepare(original, keywords=["minProperties", "maxProperties"])
    assert schema_of(wire)["minProperties"] == 1
    assert schema_of(wire)["maxProperties"] == 2


def test_original_ref_validation():
    schema = {"$ref": "#/$defs/Keys", "$defs": {"Keys": keyed()}}
    check_local_schema(schema)
    prepare(payload(schema))
    validate_response(
        payload(schema), {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
    )


def test_template_error_preserves_backend_body_and_stage():
    from lwb.scheduler import Scheduler, CapabilityError

    scheduler = object.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(config=None)
    calls = []
    original_error = {
        "error": {
            "code": "invalid_response_format",
            "message": "unsupported JSON schema keyword: minItems",
        }
    }

    class Session:
        def get(self, url, **kwargs):
            value = (
                {"backend": "ninfer", "default_generation_settings": {"n_ctx": 8192}}
                if url.endswith("/props")
                else {"data": [{"id": "test"}]}
            )
            return SimpleNamespace(ok=True, json=lambda: value)

        def post(self, url, **kwargs):
            calls.append(url)
            return SimpleNamespace(ok=False, status_code=400, json=lambda: original_error)

    backend = SimpleNamespace(_session=Session(), server_url="http://synthetic", context_size=8192)
    with pytest.raises(CapabilityError) as caught:
        scheduler.infer(backend, payload({"type": "array", "items": {}, "minItems": 1}))
    detail = caught.value.detail
    assert detail["body"] == original_error
    assert detail["backend"] == "ninfer" and detail["stage"] == "template_rendering"
    assert detail["schema_path"] == "/minItems"
    assert calls == ["http://synthetic/apply-template"]


@pytest.mark.parametrize(
    "schema",
    [
        {"const": 1, "minimum": 2},
        {"enum": [1, "x"], "type": "integer"},
        {"anyOf": [{}], "properties": {"x": {}}},
        {"type": "object", "properties": {"x": False}},
    ],
)
def test_unsafe_generation_intersections_are_not_silently_relaxed(schema):
    with pytest.raises(StructuredOutputError):
        prepare(payload(schema))


def test_object_literal_not_given_sibling_assertions():
    schema = {"type": "object", "const": {"x": 1}}
    wire, info = prepare(payload(schema))
    assert schema_of(wire) == schema
    assert info["strategy"] == "backend_direct"


@pytest.mark.parametrize("mode", [None, [], {}, 1, "silent_fallback"])
def test_invalid_modes(mode):
    from lwb.scheduler import validate_payload

    with pytest.raises(ValueError):
        validate_payload(payload({}, structured_output_mode=mode))
    with pytest.raises(StructuredOutputError):
        prepare(payload({}, structured_output_mode=mode))


def test_invalid_schema_diagnostics():
    from lwb.scheduler import validate_payload

    with pytest.raises(StructuredOutputError) as caught:
        validate_payload(payload({"properties": {"a/b": {"type": "wrong"}}}))
    assert caught.value.detail["stage"] == "schema_validation"
    assert caught.value.detail["schema_path"] == "/properties/a~1b/type"


@pytest.mark.parametrize("schema", [{"items": {}}, {"type": "string", "required": ["x"]}])
def test_implicit_or_conflicting_types_not_silently_ignored(schema):
    with pytest.raises(StructuredOutputError):
        prepare(payload(schema))


def test_business_property_named_additional_properties_does_not_bypass_boolean_check():
    with pytest.raises(StructuredOutputError):
        prepare(payload({"type": "object", "properties": {"additionalProperties": False}}))
