# 导演工作台统一调度 API（v1）

## 接入方式与安全边界

把导演工作台的「StartServer → 直连 llama.cpp → StopServer」替换为一次
`POST /lwb/v1/requests`，随后按 `request_id` 轮询。**不要在受管进程上继续直连推理**：
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
`integrations/comfyui_director_workbench` 不在本仓库；迁移到此 API 后可由集成方移除临时适配层。

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
其他选项在加载 profile 时拒绝，受支持选项的实际取值再由所选构建校验并返回启动诊断。
管理器强制附加 `--no-context-shift --parallel 1`；不支持时启动失败，不回退上下文或参数。
文本 profile 默认 `mmproj_path: ""`，不加载视觉投影器。

状态目录默认为 ComfyUI 用户目录下的 `lwb`，权限 0700，SQLite 使用 WAL 和 FULL 同步。
它保存请求摘要、状态、原始响应/错误，不保存提交提示词；响应可能含私有内容，需按私有
应用数据保护与备份。不要删除状态数据库后重启来绕过未知租约，不要在多个 worker 间共享
数据库。只支持一个 ComfyUI 进程/原生 worker；需要多个实例时须使用独立 GPU/端口/状态目录。

未配置 token 时只接受 TCP loopback 调用；配置 token 后包括本机在内都需要 Bearer。
局域网调用必须认证，忽略 `X-Forwarded-For`，拒绝浏览器 Origin 的控制 API 请求。
反向代理必须配置 token，不能依赖代理到后端的 loopback 地址作为用户身份。
多个调用方使用如下环境变量代替单 token（它优先于 `LWB_API_TOKEN`）：

```sh
export LWB_API_TOKENS_JSON='{"director-a":"secret-a","director-b":"secret-b"}'
```

认证身份只能查询/取消自己的请求。各身份都可以选择已安装的 profiles。
远端原生 `/prompt` 中的生命周期节点禁止执行，即使有 token；请使用受限 profile API。
本机原生工作流仍可使用原有二进制 widget。配置 token 时，本机生命周期 HTTP 调用也需带
认证；普通图片 `/prompt` 不受此中间件影响。纯状态节点不会公开命令行凭据。
启动日志只保留启动阶段的诊断，抑制含 prompt/authorization/API key/Bearer 标记的行；
就绪后继续排空日志管道但不记录推理日志。API 不向 ComfyUI 历史或 websocket 写入提示词/
模型响应。llama 本地传输与启动健康检查均禁用环境代理，因此继承 SOCKS/HTTP 代理不会改道。

## 请求与结果

可直接导入 `examples/api/director.postman_collection.json`；JSON schema 完整示例在
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
| `POST /lwb/v1/requests` | `{"request_id":"…","profile":"director-text","payload":{…}}`；202 返回当前状态，并不表示推理成功 |
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
  "request_id": "director-scene-001-v1",
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
正在生成的请求（启动上限 600 秒、HTTP 推理超时 3600 秒，超时进入未知隔离）。

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
python scripts/verify_director_api.py --profile director-text \
  --image-prompt /absolute/path/real-diffusion-api.json \
  --output /private/path/acceptance-results.json
```

脚本在第二次文本执行期间提交原生图片图，检查两次文本 PID 复用、图片完成后文本 PID 更换，
以及 ComfyUI worker PID 不变。请求 ID 在发送前写入结果文件；网络异常停止验收，不重放。
该脚本检查排队结果和生命周期；GPU 分配不重叠由执行入口互斥及并发测试验证，实机亦可配合
GPU 监控观察。不要把此脚本的模拟/单元测试通过当作实机 CUDA 已验收。
