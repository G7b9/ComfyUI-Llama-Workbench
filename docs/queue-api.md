# 文本与图片统一调度 API（v1）

## 接入方式与安全边界

桌面应用、批处理脚本、Web 服务及其他工作流工具可通过 `POST /lwb/v1/requests`
提交文本任务，随后按 `request_id` 查询状态和结果。**不要在受管进程上继续直连推理**：
绕过队列的其他进程/外部服务不受此协议调度。图片仍提交原生 ComfyUI `/prompt`，
包括 UI 手工提交的图片。无需重启 ComfyUI 来切换模型。

文本请求作为一个独立 `LlamaWorkbench_QueuedText` 输出节点进入**同一个原生队列**，
完整启动、预检、HTTP 推理和结果持久化均占用原生 worker。执行器入口还有 GPU 互斥；
任何普通图片图执行前必须先停止本调度器的空闲 llama 子进程。禁止把该内部节点混入
图片图；该节点只引用请求 ID，不接收提示词或生命周期参数。

进程所有权由当前 `OwnedServer` 的确切 `Popen` 对象证明；调用方租约由认证身份标记。
只有匹配租约且没有执行中的请求才允许停止或切换。不会扫描、接管或按名称/端口终止
外部服务。遇到外部服务占用端口直接报错。旧 `cleanup_previous_server` 输入保留但不再
执行任何扫描清理。Start/Stop/Status 的 `IS_CHANGED` 均返回 NaN，兼容输入签名缓存。

旧 Connection、Chat、Prompt、Embedded 节点和工作流格式保留。旧式直连外部服务器、
不经原生 worker 的自定义插件、其他 OS 进程不在统一调度的互斥保证范围内。
旧 Chat 的自动释放默认值不变；新 API 连续文本默认保留模型，空闲 30 秒后释放。

## 配置与启动

安装更新后的 `requirements.txt`。复制 `examples/api/profiles.json` 到管理员配置目录，
填入实际二进制和 GGUF 路径，设置：

```sh
export LWB_PROFILES_FILE=/absolute/path/profiles.json
export LWB_STATE_DIR=/absolute/path/persistent-lwb-state
export LWB_API_TOKEN='replace-with-a-long-random-secret'
# 用原来方式启动 ComfyUI
```

没有 `LWB_PROFILES_FILE` 时不允许提交文本请求。客户端只能选择 profile，不能指定任意
二进制、路径、端口或启动参数。配置文件是管理员信任边界，不能放入不可信用户可写目录。
自定义构建通过 profile 显式配置；当前允许的额外选项均为一参数选项：
`--flash-attn`、`--cache-type-k`、`--cache-type-v`、`--batch-size`、`--ubatch-size`、
`--threads`、`--threads-batch`、`--chat-template`、`--reasoning-format`。
此外支持下列定制构建选项；每个选项必须有且仅有一个值，重复选项拒绝：

| 选项 | 管理员 profile 允许值 |
| --- | --- |
| `--spec-type` | `draft-mtp` |
| `--spec-draft-n-max` | 整数 1–64 |
| `--spec-draft-type-k` / `--spec-draft-type-v` | `turbo3` 或 `turbo4` |
| `--fit` | `on` 或 `off` |

其他选项在加载 profile 时拒绝。带上述定制选项时，启动前运行所选二进制的 `--help`，
检查精确选项名；未声明支持或探测失败时明确报错，不尝试删参重启。实际参数取值仍由二进制
启动时验证，保留原始失败诊断。帮助文本声明只是必要条件，
具体参数及取值需通过所选构建与模型的真实运行验证。
管理器强制附加 `--no-context-shift --parallel 1`；不支持时启动失败，不回退上下文或参数。
文本 profile 默认 `mmproj_path: ""`，不加载视觉投影器。

`startup_timeout_seconds` 是管理员 profile 顶层字段，默认 600 秒，允许有限数值 1–1800 秒，
拒绝布尔值、字符串、NaN/Infinity 和越界值。HTTP 客户端不能覆盖它。
基础配置见 `examples/api/profiles.json`。需要 MTP 的兼容构建可参考可选的
`examples/api/profiles-mtp.json`；其中较大的上下文、GPU 层数及 900 秒启动超时用于演示
显式配置，不是通用推荐值。请按模型、硬件和实际冷加载耗时设置 profile，并替换占位路径。
服务不会在加载失败时自动删除参数或缩小上下文。

状态目录默认为 ComfyUI 用户目录下的 `lwb`，权限 0700，SQLite 使用 WAL 和 FULL 同步。
它保存请求摘要、状态、原始响应/错误，不保存提交提示词；响应可能含私有内容，需按私有
应用数据保护与备份。不要删除状态数据库后重启来绕过未知租约，不要在多个 worker 间共享
数据库。只支持一个 ComfyUI 进程/原生 worker；需要多个实例时须使用独立 GPU/端口/状态目录。

对 `/lwb/v1/*` 管理 API，未配置 token 时只接受 TCP loopback 调用；配置 token 后包括本机在内都需要 Bearer。
局域网调用必须认证，忽略 `X-Forwarded-For`，拒绝浏览器 Origin 的控制 API 请求。
反向代理必须配置 token，不能依赖代理到后端的 loopback 地址作为用户身份。
多个调用方使用如下环境变量代替单 token（它优先于 `LWB_API_TOKEN`）：

```sh
export LWB_API_TOKENS_JSON='{"client-a":"secret-a","client-b":"secret-b"}'
```

认证身份只能查询/取消自己的请求。各身份都可以选择已安装的 profiles。
原生 `/prompt`、`/api/prompt` 默认使用 ComfyUI 自身的访问控制：所有能提交 workflow 的
调用方都能使用 Start Server／Stop Server，包括内网网页用户，无需额外 Workbench token。
原有 workflow 中的程序路径、模型路径、端口及启动参数可继续使用，无需改为 profile。
此默认行为不把 `/lwb/v1/*` 管理 API 的权限授予这些调用方。

需要恢复旧行为时，在 **ComfyUI 服务器启动环境**设置 `LWB_RESTRICT_LIFECYCLE=true`。
未设置默认 `false`；支持 `true/false`、`1/0`、`yes/no`、`on/off`，忽略大小写和两端空白，
空字符串及其他值会报配置错误。初始化时固定策略，HTTP 或 workflow 不能覆盖；配置更改需
在计划中的服务重启时加载。严格模式保留原有本机 Origin／loopback 规则与 token 要求，
非 loopback 的生命周期提交即使携带有效 token 仍拒绝。
两种模式均禁止将内部 `QueuedText` 节点直接提交到原生队列入口；进程所有权、复用、
并发保护及 GPU 调度不变。纯状态节点不会公开命令行凭据。
启动日志只保留启动阶段的诊断，抑制含 prompt/authorization/API key/Bearer 标记的行；
就绪后继续排空日志管道但不记录推理日志。API 不向 ComfyUI 历史或 websocket 写入提示词/
模型响应。llama 本地传输与启动健康检查均禁用环境代理，因此继承 SOCKS/HTTP 代理不会改道。

## 请求与结果

可直接导入 `examples/api/queue-api.postman_collection.json`；JSON schema 完整示例在
`examples/api/text-request.json`。

```sh
curl --noproxy '*' http://127.0.0.1:8188/lwb/v1/requests \
  -H "Authorization: Bearer $LWB_API_TOKEN" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/api/text-request.json
```

生产调用方必须**先持久化自己生成的 request_id 再提交**。省略时服务端生成 UUID，但若提交
响应丢失，调用方也可能丢失 ID。相同身份、ID、profile 配置和 payload 重提返回原状态，
不会再次排队或推理；相同 ID 不同内容返回 400。更换 profile 配置后相同 ID 的重提也冲突。
未知 ID 查询返回 404，表示此数据库没有持久化接收记录。不要以新的 ID 重试未知结果。

| 接口 | 含义 |
| --- | --- |
| `POST /lwb/v1/requests` | `{"request_id":"…","profile":"local-text","payload":{…}}`；202 返回当前状态，并不表示推理成功 |
| `GET /lwb/v1/requests/{id}` | 状态和结果在同一个资源；重启后仍可查询 |
| `POST /lwb/v1/requests/{id}/cancel` | 只针对自己的此 ID；终态不变 |
| `GET /lwb/v1/status` | profiles、最近一次成功预检的模型信息、GPU 隔离状态、调度策略；模型信息是最近一次快照，不表示仍驻留 |
| `POST /lwb/v1/reconcile` | 只在能证明原进程已退出/被新 PID 世代替换后解除 GPU 隔离；绝不杀进程或重放推理 |

`payload` 接受：

- `messages`：非空数组；角色 `system/user/assistant`；字符串 content 或仅含 text 的内容片段；可选字符串 `name`。当前通用端点是纯文本，不支持图片/tool/音频消息。
- `temperature`：有限数值 0–2。
- `max_tokens`：必填正整数；不接受无限输出或隐式小上下文。
- `chat_template_kwargs`：JSON 对象，完整透传，无“兼容重试”删参。
- `response_format`：`text`、`json_object` 或标准 `json_schema` 包装；先验证 JSON Schema，再由实际 llama 构建验证支持的约束。服务端拒绝的格式原样报错，不降为文本。
- `input_tokens`：可选非负整数**断言**，须等于本次模板渲染后的服务端计数；不作为未经验证的预算依据，也不传作未知 llama 参数。

启动后通过 `/v1/models` 和 `/props` 获取真实 model id 与每 slot 的实际 `n_ctx`。
`/apply-template`（带原始模板 kwargs）和 `/tokenize` 完成输入计数。能力不可用时失败，
不猜测；`input_tokens + max_tokens > n_ctx` 拒绝提交推理。实际上下文小于请求配置也拒绝。
就绪后预检结果立即写入请求的 `model`，不用等生成结束。预检前会记录当时可用/总显存字节；
该值已包含常驻 ComfyUI CUDA 基础占用，但不是对模型 KV cache 能放下的承诺。测量不可用
明确返回 null。不会根据显存自动扩大上下文。端点依据
[llama.cpp server 接口](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)，
定制构建缺失所需端点时必须修正构建配置。

成功资源包含：

```json
{
  "request_id": "request-001-v1",
  "state": "completed",
  "pid": 12345,
  "cancel_requested": false,
  "model": {
    "model_id": "actual-model-id",
    "n_ctx": 8192,
    "input_tokens": 42,
    "token_counting": {"available": true, "method": "apply-template + tokenize"}
  },
  "response": {
    "choices": [{"message": {"role": "assistant", "content": "{\"description\":\"…\"}"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60}
  },
  "finish_reason": ["stop"],
  "usage": {"prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60}
}
```

`response` 保存完整解析后的原始 JSON，包括服务端附加字段；结构化 content 仍保留原有字符串，
不变成仅供显示的文字。HTTP 错误保存 `error.status` 和原始 `error.body`，不进行第二次生成请求。
`finish_reason: length` 不伪装成完整回答；调用方可据此拒绝被截断的输出。

## 状态、取消与恢复

正常状态：`accepted → queued → starting → running → completed/failed`。
排队取消为 `cancelled`；网络超时、断连或重启后的非终态为 `result_unknown`。

排队取消删除精确队列项，已取出但尚未启动的节点也会检查持久状态，不会漏掉取消。
启动期间取消会等待已启动的加载流程结束，若尚未发送推理则记为 `cancelled`，不再生成。
推理期间取消设置 `cancel_requested=true, cancellation=drain_in_flight`，**不声称已经中断**：
继续等待该次 HTTP 结束，保留真实完成结果，再允许切换。此版本没有可靠的 llama 单请求
强制中断协议，因此不调用 ComfyUI 全局 interrupt，也不通过停止共享进程冒充目标取消。
HTTP 客户端断开不会取消已排队任务。

每个文本请求是一个有限的原生队列项。已有图片等待时，新文本追加到队尾；有文本等待时，
后续图片也追加到队尾，限制反复 front/priority 插队造成的饥饿。取队列时两类都有等待
则交替执行，最大连续批次为 1，即使另一类到来前已有大量积压也一样；各类内部保留队列顺序。
此公平窗口中原生“插队”优先级让位于类间公平调度。连续同 profile/同身份文本复用 PID；不同调用方先释放旧租约。
没有竞争者时不逐条卸载；完成后空闲 30 秒释放。一个请求仍可执行较长时间；调度不抢占
正在生成的请求（管理员启动超时默认 600 秒、最高 1800 秒；HTTP 推理超时 3600 秒）。
新队列 API 启动就绪超时与 HTTP 推理超时均进入 `result_unknown`：保留子进程和租约，
不自动杀进程、重新启动、缩小上下文或删除参数。旧 StartServer 节点保留原有超时清理行为。

SQLite 在启动子进程**之前**记录租约；取得 PID 后记录 Linux boot id 与进程 start ticks。
worker 重启时不会自动重新提交任何请求。所有非终态转为 `result_unknown`，原响应已落盘的
completed 保持不变。驻留的孤儿 llama 也会让新 worker 隔离 GPU，即使上一请求已完成。

`result_unknown` **不代表空闲**：不自动停止模型、不启动下一模型、不重放、不因取消解除
租约；图片会得到失败状态，worker 本身继续存活。管理员确认原进程已退出后调用 reconcile，
它通过 boot id/PID 世代证明死亡才解除隔离；进程仍活着时保持隔离。结果仍为 unknown，
不会伪造 completed。若崩溃发生在 spawn 与 PID 记录间的极短窗口，仅凭端口探测不能证明
进程死亡，必须等主机 boot id 改变。非 Linux 平台缺少可靠身份验证时同样保守隔离，不能
自动恢复。此限制有意优先保证不重复推理、不错误释放仍可能占 GPU 的执行。

## 验收与测试

`python -m pytest -q` 覆盖真实子进程启动诊断、代理隔离、PID 复用、图执行互斥、目标取消
竞态、幂等/冲突、结构化透传、上下文拒绝、认证、worker 重启与进程身份核对。
模拟服务测试不等于真实 CUDA 模型验收；部署后还需在同一个 ComfyUI PID 下完成：

1. 用两个不同 ID 提交同一 profile，查询两次完成资源，检查 llama PID 一致。
2. 在第二次文本尚运行时从其他客户端提交真实图片图，确认图片只能在文本结束并释放后加载。
3. 再提交第三次文本，确认成功且 llama PID 已更换，ComfyUI PID 不变。
4. 在启用缓存的图执行 start → stop → start，检查实时状态；使用合适大小的 context 和空 mmproj。
5. 注入启动失败/超时、取消、客户端断开、worker 重启；确认原 ID 未再次生成，外部手动服务器未被停止。

不要对生产进程做无关重启或终止来执行故障测试。建议在隔离测试端口和测试模型上验证。

可在完成 profile 配置后运行真实模型验收脚本（需要导出的扩散模型 API 图）：

```sh
python scripts/verify_queue_api.py --profile local-text \
  --image-prompt /absolute/path/real-diffusion-api.json \
  --output /private/path/acceptance-results.json
```

脚本在第二次文本执行期间提交原生图片图，检查两次文本 PID 复用、图片完成后文本 PID 更换，
以及 ComfyUI worker PID 不变。请求 ID 在发送前写入结果文件；网络异常停止验收，不重放。
该脚本检查排队结果和生命周期；GPU 分配不重叠由执行入口互斥及并发测试验证，实机亦可配合
GPU 监控观察。不要把此脚本的模拟/单元测试通过当作实机 CUDA 已验收。

## 同步后仍返回 404 的启用检查

仅同步插件磁盘代码不会让正在运行的 ComfyUI 注册新路由；本插件不支持进程内热重载。
先检查当前进程的启动日志、实际插件加载路径和启动时间，确认是否加载了更新后的代码。
配置完成后，在空闲窗口重启 ComfyUI，使新路由和环境变量生效。

1. 在 **ComfyUI 使用的 Python 环境**安装本仓库 requirements；确认同步的是实际加载的插件目录。
2. 根据 `examples/api/profiles.json` 配置模型和启动参数，设置 `LWB_PROFILES_FILE`、持久
   `LWB_STATE_DIR` 和 `LWB_API_TOKEN`（或多身份配置）。环境变量必须进入 ComfyUI 启动进程，
   仅在另一个终端 export 对当前服务无效。
3. 在重启前离线检查 profile；可显式检查构建，不加载模型、不注册路由：

   ```sh
   python scripts/check_profiles.py /absolute/path/profiles.json
   python scripts/check_profiles.py /absolute/path/profiles.json --check-build
   ```

4. 先确认原生队列无运行/等待任务，且其他客户端没有仍在直连推理；仅队列为空不证明直连请求已结束。
   在确认空闲后用现有启动方式重启 ComfyUI，不用本插件按端口或进程名清理服务。
5. 新进程日志应有 `[Llama Workbench] /lwb/v1 routes registered`，附 profile 数量和隔离状态，
   不输出凭据。带认证查询 `GET /lwb/v1/status`，检查 profile 名、worker PID 与 `gpu.blocked`。

状态判断：**404** 表示此服务未注册路由（旧进程、错误端口/代理路径或插件导入失败）；
**403** 表示路由存在但未通过认证；**200 且 profiles 为空**表示路由已启用但没有配置 profile，
不能把它与 404 混为一谈。无效 profile 会使初始化失败并阻止未协调的 GPU 图执行，应先修正
启动日志中的配置错误，不能靠跳过验证或删除持久状态库恢复。

确认路由及 profile 可用后，将客户端文本推理迁移到 requests API，并执行真实
文本→文本→图片→文本验收；图片继续使用原生 `/prompt`。
