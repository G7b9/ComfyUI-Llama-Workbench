"""Bounded JSON Schema compatibility for managed execution (no network or inference).

Transforms visit schema positions only. The caller's schema remains the authority
for success; backend acceptance is never a substitute for instance validation.
"""

from copy import deepcopy
import hashlib
import json
import math

from jsonschema import Draft202012Validator, SchemaError


MAPS = {"properties", "$defs", "patternProperties", "dependentSchemas"}
SINGLES = {
    "additionalProperties",
    "items",
    "contains",
    "propertyNames",
    "not",
    "if",
    "then",
    "else",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contentSchema",
}
ARRAYS = {"anyOf", "oneOf", "allOf", "prefixItems"}
ANNOTATIONS = {
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$comment",
}
# Deliberately small portable contract. Build-specific additions require an
# administrator's pinned executable; they are not inferred from an engine name.
PORTABLE = ANNOTATIONS | {
    "$schema",
    "$defs",
    "$ref",
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "anyOf",
    "minItems",
    "maxItems",
}
DIALECT = "https://json-schema.org/draft/2020-12/schema"


def pointer(parts):
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)


class StructuredOutputError(ValueError):
    def __init__(
        self,
        message,
        *,
        stage="schema_compatibility",
        path=(),
        keyword=None,
        backend="unknown",
        **extra,
    ):
        super().__init__(message)
        self.detail = dict(
            kind="structured_output",
            stage=stage,
            backend=backend,
            schema_path=pointer(path),
            keyword=keyword,
            message=message,
            **extra,
        )


def walk(schema, path=()):
    yield path, schema
    if not isinstance(schema, dict):
        return
    for key, value in schema.items():
        if key in MAPS and isinstance(value, dict):
            for name, child in value.items():
                yield from walk(child, (*path, key, name))
        elif key in SINGLES and isinstance(value, (dict, bool)):
            yield from walk(value, (*path, key))
        elif key in ARRAYS and isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, (*path, key, index))


def check_local_schema(schema):
    """Only local, stable definition references; never fetch caller URLs."""
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise StructuredOutputError(
            "Invalid Draft 2020-12 schema",
            stage="schema_validation",
            path=exc.absolute_path,
            keyword=exc.validator,
        ) from exc
    for path, node in walk(schema):
        if not isinstance(node, dict):
            continue
        if "$schema" in node and node["$schema"].rstrip("#") != DIALECT:
            raise StructuredOutputError(
                "Only Draft 2020-12 is supported", path=(*path, "$schema"), keyword="$schema"
            )
        for key in ("$id", "$anchor", "$dynamicAnchor", "$dynamicRef", "$vocabulary"):
            if key in node:
                raise StructuredOutputError(
                    "Schema resource rebasing/dynamic vocabularies unsupported",
                    path=(*path, key),
                    keyword=key,
                )
        if "$ref" in node:
            ref = node["$ref"]
            parts = ref.split("/")
            if len(parts) != 3 or parts[:2] != ["#", "$defs"] or "%" in ref:
                raise StructuredOutputError(
                    "Only root #/$defs/name references are supported",
                    path=(*path, "$ref"),
                    keyword="$ref",
                )
            name = parts[2].replace("~1", "/").replace("~0", "~")
            if name not in schema.get("$defs", {}):
                raise StructuredOutputError(
                    "Unresolved local definition", path=(*path, "$ref"), keyword="$ref"
                )


def finite_keys(schema):
    result = deepcopy(schema)
    notes = []
    # Collect first: newly expanded properties are already transformed children
    # when parents are visited in reverse order.
    for path, node in reversed(list(walk(result))):
        if not isinstance(node, dict) or "propertyNames" not in node:
            continue
        rule = node["propertyNames"]
        at = (*path, "propertyNames")
        if (
            not isinstance(rule, dict)
            or set(rule) - {"enum", "type", *ANNOTATIONS}
            or not isinstance(rule.get("enum"), list)
            or not rule["enum"]
            or any(not isinstance(k, str) for k in rule["enum"])
            or rule.get("type", "string") != "string"
        ):
            raise StructuredOutputError(
                "propertyNames requires a finite string enum for conversion",
                path=at,
                keyword="propertyNames",
                conversion_available=False,
            )
        conflicts = {
            "patternProperties",
            "unevaluatedProperties",
            "allOf",
            "anyOf",
            "oneOf",
            "not",
            "if",
            "then",
            "else",
            "dependentSchemas",
            "$ref",
        } & node.keys()
        if conflicts or node.get("type") != "object":
            raise StructuredOutputError(
                "Finite key conversion requires a plain object without composition",
                path=at,
                keyword="propertyNames",
                conversion_available=False,
            )
        names = rule["enum"]
        existing = node.get("properties", {})
        extra = node.get("additionalProperties", True)
        props = {
            k: deepcopy(existing[k] if k in existing else ({} if extra is True else extra))
            for k in names
            if k in existing or extra is not False
        }
        if (
            set(node.get("required", [])) - props.keys()
            or node.get("minProperties", 0) > len(props)
            or node.get("maxProperties", len(props)) < len(node.get("required", []))
        ):
            raise StructuredOutputError(
                "Finite key constraints are unsatisfiable",
                path=at,
                keyword="propertyNames",
                conversion_available=False,
            )
        node["properties"] = props
        node["additionalProperties"] = False
        del node["propertyNames"]
        notes.append({"schema_path": pointer(at), "conversion": "finite_keys_to_properties"})
    return result, list(reversed(notes))


def digest(schema):
    return hashlib.sha256(
        json.dumps(
            schema, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def prepare(payload, *, backend="unknown", keywords=(), binary_sha256=None):
    """Return one independent wire payload and non-content diagnostic metadata."""
    wire = deepcopy(payload)
    mode = wire.pop("structured_output_mode", "enforce")
    if not isinstance(mode, str) or mode not in {"enforce", "post_validate"}:
        raise StructuredOutputError("Unknown structured_output_mode", backend=backend)
    fmt = wire.get("response_format") or {}
    if fmt.get("type") != "json_schema":
        return wire, None
    original = fmt["json_schema"]["schema"]
    try:
        check_local_schema(original)
        if mode == "post_validate":
            # Explicitly syntax-only generation; no implicit text fallback. Keep
            # name/description/strict unchanged, and validate the original below.
            generated, notes = {}, []
            strategy = "post_validate"
        else:
            if "propertyNames" in keywords:
                generated, notes = deepcopy(original), []
            else:
                generated, notes = finite_keys(original)
            nodes = list(walk(generated))
            schema_paths = {path for path, _ in nodes}
            for path, node in nodes:
                if node is False and (
                    not path or path[-1] != "additionalProperties" or path[:-1] not in schema_paths
                ):
                    raise StructuredOutputError(
                        "False subschemas are not portable", path=path, keyword="false"
                    )
                if not isinstance(node, dict):
                    continue
                for key in node:
                    if key not in PORTABLE | set(keywords):
                        raise StructuredOutputError(
                            "Generation constraint is not in the configured capability policy",
                            path=(*path, key),
                            keyword=key,
                            conversion_available=False,
                        )
                for expected, constraints in (
                    ("object", {"properties", "required", "additionalProperties"}),
                    ("array", {"items", "minItems", "maxItems"}),
                ):
                    if constraints & node.keys() and node.get("type") != expected:
                        key = sorted(constraints & node.keys())[0]
                        raise StructuredOutputError(
                            "Portable constraints require an explicit single type",
                            path=(*path, key),
                            keyword=key,
                        )
                # Common grammar compilers choose one branch and ignore siblings.
                for branch in ("enum", "const"):
                    if branch in node:
                        if set(node) - ANNOTATIONS - {branch, "type", "$defs", "$schema"}:
                            raise StructuredOutputError(
                                "Literal with sibling assertions is not portable",
                                path=(*path, branch),
                                keyword=branch,
                            )
                        values = node["enum"] if branch == "enum" else [node["const"]]
                        if "type" in node and any(
                            not Draft202012Validator({"type": node["type"]}).is_valid(v)
                            for v in values
                        ):
                            raise StructuredOutputError(
                                "Literal conflicts with its declared type",
                                path=(*path, branch),
                                keyword=branch,
                            )

                if "anyOf" in node and set(node) - ANNOTATIONS - {"anyOf", "$defs", "$schema"}:
                    raise StructuredOutputError(
                        "anyOf with sibling assertions is not portable",
                        path=(*path, "anyOf"),
                        keyword="anyOf",
                    )
                if "$ref" in node and set(node) - ANNOTATIONS - {"$ref", "$defs", "$schema"}:
                    raise StructuredOutputError(
                        "$ref with sibling assertions is not portable",
                        path=(*path, "$ref"),
                        keyword="$ref",
                    )
                # JSON Schema's default is true; some llama builds default false.
                if (
                    node.get("type") == "object"
                    and "additionalProperties" not in node
                    and not {"enum", "const"} & node.keys()
                ):
                    node["additionalProperties"] = True
                    notes.append(
                        {
                            "schema_path": pointer((*path, "additionalProperties")),
                            "conversion": "explicit_default_true",
                        }
                    )
            strategy = "equivalent_transform" if notes else "backend_direct"
        wire["response_format"]["json_schema"]["schema"] = generated
        return wire, {
            "strategy": strategy,
            "backend": backend,
            "capability_source": "portable_policy+configured_keywords",
            "binary_sha256": binary_sha256,
            "configured_keywords": list(keywords),
            "original_schema_sha256": digest(original),
            "generation_schema_sha256": digest(generated),
            "conversions": notes,
            "output_validation": "original_draft2020_12",
            "generation_acceptance": "backend_checked_on_request",
        }
    except StructuredOutputError as exc:
        exc.detail["backend"] = backend
        exc.detail["strategy"] = "unsupported"
        raise


def validate_response(payload, raw, backend="unknown"):
    fmt = payload.get("response_format") or {}
    if fmt.get("type") not in {"json_schema", "json_object"}:
        return
    schema = fmt["json_schema"]["schema"] if fmt["type"] == "json_schema" else {"type": "object"}
    validator = Draft202012Validator(schema)

    def bad_json(value):
        raise ValueError("Non-finite JSON number")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite JSON number")
        return number

    choices = raw.get("choices") if isinstance(raw, dict) else None
    if not isinstance(choices, list) or not choices:
        raise StructuredOutputError(
            "Missing completion choices", stage="output_validation", backend=backend
        )
    for index, choice in enumerate(choices):
        try:
            if choice.get("finish_reason") != "stop":
                raise ValueError("Structured completion did not finish normally")
            message = choice["message"]
            if message.get("refusal") or message.get("tool_calls"):
                raise ValueError("Refusal/tool response is not structured content")
            value = json.loads(
                message["content"],
                parse_constant=bad_json,
                object_pairs_hook=unique_object,
                parse_float=finite_float,
            )
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise StructuredOutputError(
                "Invalid or incomplete JSON content",
                stage="output_validation",
                backend=backend,
                choice=index,
            ) from exc
        try:
            error = next(validator.iter_errors(value), None)
        except (RecursionError, ValueError) as exc:
            raise StructuredOutputError(
                "Original schema validation could not complete",
                stage="output_validation",
                backend=backend,
                choice=index,
            ) from exc
        if error is not None:
            # Do not log instance values or schema literals; raw is stored only
            # in the existing authenticated journal response.
            raise StructuredOutputError(
                "Output does not satisfy original schema",
                stage="output_validation",
                path=error.absolute_schema_path,
                keyword=error.validator,
                backend=backend,
                instance_path=pointer(error.absolute_path),
                choice=index,
            )


def check_capability_config(config):
    keywords = config.structured_output_keywords
    if not isinstance(keywords, (tuple, list)) or any(
        not isinstance(k, str) or not k for k in keywords
    ):
        raise ValueError("structured_output_keywords must be a list of keyword names")
    if keywords and (
        len(config.binary_sha256) != 64
        or any(c not in "0123456789abcdef" for c in config.binary_sha256)
    ):
        raise ValueError("structured_output_keywords requires a pinned binary_sha256")
    object.__setattr__(config, "structured_output_keywords", tuple(keywords))
