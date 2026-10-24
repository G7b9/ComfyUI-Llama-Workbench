# Managed structured output compatibility

This contract applies to `/lwb/v1/requests` and Managed Profile workflow calls.
It does not change external Connection or embedded backend compatibility retries.
There is no application-specific schema, field name, model launch, or inference retry.

## Three separate checks

1. Submission checks the JSON Schema's syntax (Draft 2020-12), envelope, and local
   reference boundary. `check_schema` is not an instance validator.
2. After the existing model/context probes, Workbench constructs one independent
   generation payload. The actual backend receives it during template rendering,
   token counting and generation. Backend rejection remains a backend error.
3. After a successful HTTP response, every structured choice must finish with
   `stop`, contain one JSON value, and validate against the **original** schema.
   JSON mode must contain an object. Refusals, tool responses, fences, duplicate
   keys, nonfinite numbers, truncation and invalid content fail. No repair/retry.

The submitted payload, idempotency digest and in-memory original schema are not
mutated. As before, prompts/payloads are not durably stored or replayed after a
restart. The private journal retains the raw response, including failed output;
ComfyUI history/logs do not receive it. Schema literals may themselves be private:
metadata stores schema hashes and conversion pointers, not whole schema copies.

## Finite keys

For an explicitly typed object, a `propertyNames` schema containing a nonempty
string `enum` (optionally `type: string` and annotations) becomes explicit
`properties` and `additionalProperties: false`. Each existing allowed property's
schema is retained. An undeclared allowed key uses the original
`additionalProperties` value schema, or `{}` when additional values were allowed.
If additional properties were forbidden, undeclared keys remain forbidden.
Properties outside the enum are removed because the original already forbade them.

`required`, `minProperties`, `maxProperties` and value schemas are preserved.
Neither key becomes required merely by appearing in the enum; missing is never
converted into null. Obvious required/allowed-key and property-count conflicts
fail before generation. Example:

```json
{"type":"object","propertyNames":{"enum":["first","last"]},
 "additionalProperties":{"type":"integer"}}
```

becomes:

```json
{"type":"object","properties":{"first":{"type":"integer"},
 "last":{"type":"integer"}},"additionalProperties":false}
```

Both accept `{}`, either key, or both keys; neither accepts another key or null.
Nested objects, arrays and root `$defs` with `#/$defs/name` references are supported.
Only schema-bearing positions are visited. A property named `propertyNames`, or
objects inside `default`, `examples`, `enum` and `const`, remain ordinary data.

The conversion refuses pattern-based names, extra assertions on the name schema,
non-object types, and same-object composition, conditional, pattern-property or
unevaluated-property rules. It does not solve arbitrary schema satisfiability.
References to movable schema locations, remote resources, `$id`, anchors, dynamic
references and other dialects are rejected even in post-validation mode. This
keeps validation local and prevents a rewrite from changing reference meaning.

## Generation policies and build identity

`payload.structured_output_mode` is optional and only valid with `json_schema`:

| Mode | Generation | Success condition |
| --- | --- | --- |
| `enforce` (default) | Conservative generation policy plus equivalent conversions | Original schema validates |
| `post_validate` (explicit) | Same JSON-schema envelope with schema `{}`; JSON syntax only | Original schema validates |

Post-validation mode is **not** constrained generation against the original
schema. Describe the desired result in the messages; the original schema is not
automatically added to the prompt. Even this mode can be refused by the backend.
Neither mode falls back to text. `json_schema.strict` is forwarded unchanged;
it is not reinterpreted as Workbench's compatibility switch.

`model.structured_output.strategy` distinguishes `backend_direct`,
`equivalent_transform`, and `post_validate`. Compatibility failures carry
`error.strategy: unsupported`. Direct means the original assertions are sent for
backend execution, **not** that Workbench has empirically certified the compiler.
`generation_acceptance: backend_checked_on_request` makes this boundary explicit:
`/apply-template` acceptance alone does not prove decoder enforcement. Final
validation is always required, including on backends that silently ignore features.

The default policy is deliberately bounded: types, properties/required/additional
properties, items and item counts, enum/const, local definitions/references,
standalone anyOf and annotations. Object/array assertions require an explicit
single matching type; an omitted or conflicting type is not silently reinterpreted.
Literal, reference and union branches with
unsafe sibling assertions are refused. False subschemas outside
`additionalProperties` are refused. Object `additionalProperties` defaults are
made explicitly `true`, preserving JSON Schema's default on compilers whose
implicit default is false. This is also recorded as an equivalent conversion.

Other constraints (including property counts, numeric and string bounds, regex,
formats, unique items, conditionals and other composition) are not silently
removed. A finite-key conversion preserving `minProperties` does **not** imply
that the selected decoder can enforce that keyword. Default enforcement refuses
such a request unless the administrator has verified the needed capability, or
the caller explicitly chooses post-validation. Draft 2020-12 `format` is an
annotation in final validation, not a configured format-assertion vocabulary.

An administrator can add `structured_output_keywords` to either engine profile,
with a nonempty pinned `binary_sha256`, for example:

```json
{"structured_output_keywords":["minLength","maxLength"],
 "binary_sha256":"<actual verified executable SHA256>"}
```

This is a **trusted capability declaration**, not automatic feature detection.
Verify the precise keyword usages/combinations on that binary and configuration
first. The existing launcher verifies the hash before use; a changed binary
cannot inherit the declaration. `propertyNames` in this list requests native
execution, without the finite-key conversion. Structural/reference safeguards
still apply. Actual backend errors remain authoritative; no model is started
just to probe a capability and no experimental inference is added to requests.
The record contains engine, policy, configured keywords, binary identity,
original/generation schema hashes and conversion pointers. An unchanged empty
capability configuration preserves legacy request digests; changing capabilities
changes profile identity and therefore conflicts with old request IDs.

## Errors and compatibility impact

Compatibility and output errors contain a stage, backend, JSON Pointer and
keyword when available. Template HTTP errors preserve the original body/status/
endpoint and include matching generation-schema keyword locations. Multiple
matches remain a list rather than a fabricated precise location. Other backend
HTTP failures retain their original bodies; inference errors identify the stage.

A valid HTTP response that violates the schema now produces `state: failed`,
`error.stage: output_validation`, and the original `response`, usage and finish
reasons. This intentionally changes the previous HTTP-success-only behavior.
Existing schema features outside the conservative policy can now fail early;
review the actual build before declaring extra capabilities. Do not enable
post-validation implicitly to preserve acceptance. The journal table layout,
queue ownership, cancellation/drain, resource leases, model switching and unknown
transport-result quarantine are unchanged. Managed workflow validation failures
raise a backend error after a known response and do not quarantine an idle engine.

## Evaluation evidence (2026-10-24)

The original failure is consistent with a **backend schema capability rejection**,
not Workbench's Draft202012Validator rejecting a standard keyword:

- Repository submission validates schema syntax, then the native queue worker
  renders `/apply-template`, counts `/tokenize`, submits one chat completion and
  journals the response. Previously the last step did not validate the content.
- The inspected private `director-text` profile selects NInfer and pins executable
  SHA256 `a72443e4647d81d2f69b22d4a2b07561d320af6a0231d88ca6ae83aee1011ee8`.
- Local NInfer source checkout `8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3`
  (`iamwavecut/ninfer-all`, under deployment checkout
  `b25e1146ce3d97b36d3a7a52c87aa0f70d633840`) contains
  `src/text/structured_output.cpp::schema_check`: its whitelist omits
  `propertyNames` and throws the reported `unsupported JSON schema keyword:`
  message. Vendored XGrammar code having related functionality does not bypass
  this front-end check. A source checkout alone does not establish the exact
  source-to-binary build provenance of the installed artifact.
- Local llama.cpp checkout `d14e368275864a8986bbdbabec67787570715515`
  has a different schema-to-grammar converter. The configured binary SHA256 is
  `9d7beca7ede5dfac7f40e8338e64a7aa53532ea83fd2df9bccb34d00a9c35853`.
  Both installed executables were hashed read-only and match their profiles. Do not assign a permanent support boolean to either engine name.
- [Current upstream NInfer documentation](https://github.com/Neroued/ninfer/blob/master/docs/serving.md)
  describes newer structured decoding errors and no schema prompt injection.
  That differs from the older deployment contract; budget using actual template
  rendering with the actual generation format, not assumptions from latest docs.
- [llama.cpp grammar documentation](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
  documents a subset, silently skipped features and compiler-specific defaults.
  Backend acceptance cannot replace original-schema validation.

The shared layer is appropriate because schema semantics and final validation are
independent of launch mechanisms. Build-specific capability declarations belong
to trusted profiles, while existing engine adapters keep lifecycle ownership.
Implementing a full decoder/compiler or changing backend native schema support
would exceed this bounded compatibility change.

Tests distinguish pure schema equivalence, mocked transport, isolated local HTTP
fixtures and actual GPU inference. Read-only service checks found the ComfyUI
queue empty and no service listening at the configured model endpoint; no real
model inference was performed, and no process was restarted or model switched.
See the implementation's tests for adversarial schemas and regression coverage.

Callers can remove their finite-enum key workaround after this code is deployed
in the actual worker and their original schemas pass an end-to-end acceptance
request on the selected profile. Check the conversion metadata and final success;
if other constraints remain unsupported, address those explicitly first. This
change does not authorize deleting arbitrary-keyword workarounds wholesale.


## Verification results

- `python -m pytest -q`: **421 passed** (2026-10-24), including pure schema
  equivalence, mock endpoint requests, isolated local HTTP/process fixtures, and
  queue/idempotency/cancellation/unknown-result regression tests. This is not GPU
  model acceptance. The initial sandbox run could not create test sockets; the
  complete run used permission for isolated test sockets and passed.
- `python -m ruff check` on changed Python modules/tests: passed.
- `git diff --check`: passed.
- Real backend inference: **not run**. Existing service/queue inspection and
  executable hash comparison are read-only deployment checks, not inference tests.
- No private profile/workflow/data changes, runtime restart, commit, push or deploy.
