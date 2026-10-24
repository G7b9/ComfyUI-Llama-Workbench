# Managed engines and local acceptance

Workbench keeps the native ComfyUI worker and `/lwb/v1/requests`. Administrator
profiles select `engine: "llama.cpp"` (the default for old profiles) or
`engine: "ninfer"`. NInfer weights must not be supplied to Start Server. The new
**Llama Workbench Managed Profile** node returns the existing backend socket to
Chat, Skills, Prompt and Prompt Enhancer; it loads lazily when generation begins.
Existing node IDs, sockets and return values remain unchanged. Connection remains
an external client and acquires no stop permission.

## Configuration

Use [the four-engine example](../examples/api/engine-profiles.json), replacing all
placeholder paths and SHA256 values with administrator-reviewed local artifacts.
Keep actual profiles, credentials, model weights, private Skills, exported
workflows and results outside version control (this workspace uses `.local/`).
Run `python scripts/check_profiles.py /private/profiles.json`; `--check-build`
checks llama build options or the NInfer executable hash without loading weights.
Launch hashes NInfer weights/runtime and any pinned llama artifacts. Context is
never increased from free VRAM or reduced to recover from an error.

Every profile in a shared GPU group must use the same absolute `switch_lock`,
pointing to the external deployment's existing `model-switch.lock`. Legacy llama
profiles without a lock still work; a mixed installation inherits the group lock
from the configured profiles. The lock is Linux `flock`, not a separate
Workbench-only filename. Administrators must coordinate other GPU programs.

The NInfer adapter exposes fixed reviewed parameters, not shell commands or
arbitrary Docker arguments: immutable image ID, verified executable/weight,
read-only runtime/model/toolkit mounts, current user, loopback publish, single
concurrency, rk8v4, MTP3, resident vision and bounded host KV payload. It uses a
unique Workbench container and records its full ID **before** starting it. It
neither imports nor invokes `modelctl.py`, `start-*.sh` or `stop.sh`; the native
QueuedText job therefore does not fail the launcher's whole-queue idle check.
The command contract follows the administrator-reviewed local NInfer deployment;
upstream NInfer is Apache-2.0. No engine code, weights or upstream prompts are
bundled/copied here.

An optional `adopt_container_id` is an explicit administrator grant for a complete
64-character container ID. Only a live, idle instance with matching immutable
image, command, read-only mounts, ports, user and permitted host configuration
can be adopted. A known name, label, port or model path alone never grants stop
permission. Avoid leaving a one-time adoption grant in a general runtime profile;
use it to release the specifically authorized external instance, then create new
Workbench containers. Adopted containers are stopped but never removed.

## GPU execution and ownership

Lock order is the Workbench RLock, then the shared deployment flock, then the
engine operation. Flock acquisition is nonblocking: a launcher already switching
makes the native graph fail safely without killing its worker. The flock remains
held throughout a resident model lease and unknown results, across successive
text jobs. Native graph execution keeps it through diffusion/video sampling.
The old external launcher checks this same lock before its queue checks; there
is no reverse callback into Workbench while holding it.

Before another model loads, Workbench releases ComfyUI's model cache and checks
live compute PID allocations and total memory. Only its own bounded CUDA context
is permitted; unknown compute occupants or more than 2 GiB residual memory cause
refusal. After managed text completes, release checks engine idle state, exact
instance/generation, stop completion and GPU handoff before image allocation.
ComfyUI itself stays resident. Pure external-client graphs remain usable, but an
external Connection cannot authorize loading diffusion alongside its server.

This protects the native worker and cooperating launchers. External direct HTTP
clients, unrelated CUDA programs and launchers that ignore the flock are outside
that guarantee. They can introduce a request between an idle query and shutdown;
route shared workloads through Workbench and restrict direct endpoint access.
Docker access remains administrator access; no daemon socket is added to model
containers and no privileged/device/capability options are offered.

## Requests, budget and recovery

Both engines keep stable IDs, owner isolation, fair alternation with waiting
native image graphs, same-profile reuse, idle release, raw response/usage/finish
reasons and drain cancellation. Managed transport makes one generation submission
and never drops template kwargs/schema constraints or switches endpoints to retry.
The historical external Connection compatibility behavior is retained.
Managed Chat and Prompt Enhancer also suppress the historical format/thinking
fallback generation; they return the original result or validation failure.
Explicit bounded Skill reference follow-ups retain their existing semantics.

In a coordinated GPU group, llama.cpp must also report a nonempty `/slots` list
with every `is_processing` flag false before a new request or release. Missing,
busy or unreachable slot state prevents handoff. This applies to legacy llama
profiles inheriting a mixed group's lock, alongside the GPU preflight.

Text preflight reads `/v1/models` and `/props`, verifies the exact configured
context, renders `/apply-template` with **both** original template kwargs and
`response_format` after shared equivalent schema conversion, and counts `/tokenize`. NInfer injects structured schema text
into its real prompt; omitting the format would undercount. Inputs plus explicit
output reservation must fit; assertions must match. Requests remain text-only.

Managed NInfer visual Chat checks runtime vision capability, validates up to ten
local base64 PNG/JPEG/WebP images (32 MiB combined, 4MP and 4096 edge per image),
and calls the actual `/v1/responses/input_tokens` endpoint with the same turns,
template kwargs and structured format. This counts decoded media rather than
text placeholders. HTTP/file image URLs, tools, audio and video messages are not
accepted by this managed path. Managed llama visual budgeting is unavailable;
use the existing Connection path and distinguish its visual tests from the text
queue. Unlimited `max_tokens=-1` remains available on old nodes/backends; managed
profiles require a positive output reservation.

Snapshots include engine, profile, configuration fingerprint, logical context,
verified weight identity when configured, binary hash when configured and exact
instance identity. Deployment aliases are not weight identities. Runtime probes,
live GPU samples and historical performance must be reported separately; a 262K
NInfer logical window is not 262K physically resident KV. Current reviewed
NInfer configuration uses 114688 GPU KV tokens and a 12 GiB host payload cap;
current turbo entry uses 147456 context, not its historical 163840 experiment.

Cancellation during inference reports `drain_in_flight` until the real request
ends. No global interrupt, service shutdown or resend is used as targeted cancel.
Startup/transport uncertainty pins the lease and prevents the next model/graph.
Worker restart marks unfinished jobs unknown without replay. Docker recovery
queries the exact container through the daemon: a dead host Popen/PID is not a
container death criterion. A `created` container with a pending start is not
proof of a failed start. Daemon/network failure, unrecorded creation windows or
changed container generation remain quarantined. `/lwb/v1/reconcile` only clears
an identity proved absent/exited; it never stops anything or reconstructs an
unknown response. Preserve SQLite and inspect the original instance before
administrator recovery. Do not delete the lease database to bypass quarantine.

## Director compatibility and H3

Keep the director provider origin at its existing 50003 value while queue mode
is enabled: that value is the current client's compatibility gate. The selected
Workbench `director-text` profile can point to any supported engine/actual port,
including the managed deployment port. Thus existing clients need no business/UI
patch. Change only the administrator mapping; existing IDs include the config
fingerprint, so changing it makes conflicting resubmissions fail rather than
silently run a different model. The director uses a global queue profile today.

Workflow Chat is a separate chain: changing `director-text` does not rewrite its
Start Server node. To opt in, copy a successful exported API workflow privately,
replace its backend provider with Managed Profile, keep its existing Chat,
settings, Skill and ordered image wiring, select a profile explicitly, and keep
Chat's owned-release control enabled before downstream sampling. H3's existing
reply validator must approve the final enhancement before video submission.
Never guess custom-node mappings or overwrite a user's original export.

The inspected director private queue profile had 163840 plus F16 mmproj although
its README described text without a projector; its fallback runtime explicitly
strips mmproj, whereas its queue profile did not. Retain that original file as
historical/local configuration. Establish distinct reviewed text and visual
profiles instead of silently replacing it or treating a text memory measurement
as visual capacity. NInfer's reviewed deployment keeps embedded resident vision
for all three profiles; that resource cost belongs in its measurements.

## Executable acceptance and rollback

Run relevant tests first (`python -m pytest -q`). For real GPU evidence:

```sh
python scripts/verify_engine_profiles.py --model-url http://127.0.0.1:18200 \
  --profile gsq --profile ud-q4-k-xl --profile uncensored-q5-k-m \
  --profile llama-turbo --schema-dir /private/director-schemas \
  --output /private/engine-matrix.json
python scripts/verify_queue_api.py --profile gsq \
  --image-prompt /private/successful-image-api.json \
  --output /private/handoff.json
```

Use independent Workbench and model credentials in `LWB_API_TOKEN` and
`LWB_MODEL_API_TOKEN`; do not assume local service, LAN proxy and Workbench share
authentication. Matrix IDs are saved before submission. Unknown outcomes stop the
matrix. Reports contain private raw results; companion JSONL holds GPU/RSS/PSS
samples. The nonstream queue response cannot measure exact first-token latency;
that field is explicitly null, with backend prefill/decode timings recorded
separately. Do not relabel prefill or total HTTP latency as observed first token.
Use an explicitly coordinated streaming measurement for that metric.

`scripts/verify_streaming_latency.py` performs that separate measurement with
`--profiles-file`, repeated `--profile`, `--comfy-journal` and `--output`. It
verifies the native worker generation and empty queue/journal lease, holds the
same deployment flock, creates only its own instances and records first content
SSE arrival. A pending/unknown request stops the run without replay or release.
Keep exact director wire schemas and stronger full constraint schemas as separate
cases: a client that currently strips length constraints does not prove that the
engine supports those constraints.

Rollback while idle: query requests/queue, release the exact owned lease normally
(or prove it exited and reconcile if unknown), restore backed-up plugin runtime
files and original startup environment, then restart only the authorized ComfyUI
instance. Keep original profiles, containers, SQLite, outputs and historical
reports. After a running worker restart, expect quarantine if its old model is
still alive. Never use global kill/pkill, deployment-group stop, or hot reload.


Managed structured outputs now use original-schema result validation and a bounded
shared compatibility policy. See [structured output](structured-output.md) for
finite-key conversion, explicit post-validation mode, pinned build capability
declarations, diagnostic metadata and migration impact.
