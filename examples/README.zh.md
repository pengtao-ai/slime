# PyroMind SDK

适用于 [PyroMind AI](https://pyromind.ai/) 平台 API 的轻量级 Python SDK — 管理训练工作流、Jupyter 实例、推理任务、EchoMind 等。

## 安装

```bash
pip install pyromind-sdk
```

需要 Python >= 3.8。

## 快速开始

```python
from pyromind_sdk import PyroMindAPIClient
from pyromind_sdk.client.models import TrainingTaskCreateRequest

client = PyroMindAPIClient(api_key="your-api-key")

# 创建并运行一个 Studio 任务
task = client.studio.create(
    TrainingTaskCreateRequest(
        name="my-workflow",
        workflow={"nodes": [...]}
    )
)
print(f"Created task: {task.task_id}")
```

## 通过 Docker CLI 管理 Kubernetes Sandbox（docker-rt）

`pyromind_sdk.docker_rt` 内置了 Docker Engine API 门面：它监听 Unix Socket（或 TCP），
把 `docker` 命令翻译成 Kubernetes Pod 操作，并把 `KubeEnvironment` 作为 SDK 侧的适配层。

```bash
pip install -e .

# 启动 daemon（可用 Docker Desktop 或任意可达集群）
docker-rt
# 也可用统一 SDK CLI 启动
# pyromind docker-rt

# SDK 默认值：kube-context=docker-desktop、namespace=default、node-selector=关闭
# 需要连其他集群时用 DOCKER_RT_* 环境变量覆盖。

# 后台启动
pyromind docker-rt --daemon
# pyromind docker-rt --daemon --log-file /tmp/docker-rt.log --pid-file /tmp/docker-rt.pid

# 带参数后台启动
export PYROMIND_API_KEY=XXXXXXXXX
export PYROMIND_BASE_URL=https://pre-api.pyromind.ai/api/v1
export PYROMIND_CLUSTER='us-west-1#pre'
export DOCKER_RT_BACKEND=k8s-middleware
pyromind docker-rt --daemon

# 把 Docker CLI 指向 docker-rt
docker-rt-context

docker version
docker run -d --name demo busybox:1.36 sleep 300
docker ps
docker exec demo echo hello
```

### `pyromind docker-rt` 参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--sock SOCK` | 暴露给 Docker CLI 的 Unix socket 路径 | `$DOCKER_RT_SOCK` 或 `/tmp/docker-rt.sock` |
| `--daemon` | 后台启动 docker-rt，命令立即返回 | 关闭 |
| `--log-file FILE` | `--daemon` 模式使用的日志文件 | `$DOCKER_RT_LOG_FILE` 或 `/tmp/docker-rt.log` |
| `--pid-file FILE` | 把后台进程 PID 写入 `FILE` | 不写 |
| `-h`, `--help` | 显示帮助并退出 | - |

```bash
pyromind docker-rt \
  --daemon \
  --sock /tmp/docker-rt.sock \
  --log-file /tmp/docker-rt.log \
  --pid-file /tmp/docker-rt.pid
```

#### docker-rt 环境变量

| 变量 | 默认值 | 含义 |
|------|--------|------|
| `DOCKER_RT_SOCK` | `/tmp/docker-rt.sock` | Unix socket 路径 |
| `DOCKER_RT_HOST` / `DOCKER_RT_PORT` | 空 / `2375` | 改用 TCP 监听 |
| `DOCKER_RT_LOG_FILE` | `/tmp/docker-rt.log` | 后台日志文件 |
| `DOCKER_RT_KUBECONFIG` / `KUBECONFIG` | `~/.kube/config` 或包内 `.kube.yaml` | kubeconfig 路径 |
| `DOCKER_RT_KUBE_CONTEXT` | `docker-desktop` | Kubernetes context 名 |
| `DOCKER_RT_NAMESPACE` | `default` | 目标 Kubernetes namespace |
| `DOCKER_RT_NODE_SELECTOR` | `none` | Pod `nodeSelector`（`key=val,...`；`none` 关闭） |
| `DOCKER_RT_BACKEND` | `kube` | 后端：`kube` 或 `k8s-middleware` |
| `DOCKER_RT_GPU_CARD` | 空 | k8s-middleware 后端配合 `docker run --gpus` 时指定 GPU 卡型号 |
| `DOCKER_RT_INSPECT_MODE` | `sandbox` | `docker inspect` 返回结构：`sandbox` 或 `standard` |
| `DOCKER_RT_DEFAULT_IMAGE` | SWE-bench 默认镜像 | `docker images` 默认条目 |
| `DOCKER_RT_PORT_FORWARD_MODE` | `auto` | `-p` 后端：`auto` / `direct` / `api` |
| `DOCKER_RT_BUILDKIT_ADDR` | 空 | `buildctl` 地址，如 `unix:///run/buildkit/buildkitd.sock` |
| `DOCKER_RT_BUILD_REGISTRY` | 空 | 短镜像 tag 的推送前缀 |
| `DOCKER_RT_BUILD_PUSH` | `true` | build 后是否 push |
| `DOCKER_RT_BUILD_TIMEOUT` | `3600` | `buildctl` 超时秒数 |
| `DOCKER_RT_SERVICE_DNS` | `true` | 创建 ClusterIP Service 支持 Compose 服务名 DNS |
| `DOCKER_RT_ORPHAN_POLICY` | `adopt` | `adopt` 恢复受管 Pod；`reap` 启动时删除 |
| `DOCKER_RT_CLEANUP_ON_EXIT` | `false` | `true` 时退出删除受管 Pod |
| `DOCKER_RT_JUICEFS_UID` | 从 namespace 推导 | JuiceFS subPath 用户 ID |
| `DOCKER_RT_JUICEFS_PVC` | 自动发现 | JuiceFS PVC 名 |
| `DOCKER_RT_JUICEFS_HOST_PREFIXES` | 空 | 宿主机路径到 JuiceFS subPath 的额外映射 |
| `DOCKER_RT_CONTEXT` | `docker-rt` | `docker-rt-context` 使用的 Docker context 名 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

#### `docker inspect` 返回结构

默认 `DOCKER_RT_INSPECT_MODE=sandbox`，`docker inspect` 只返回：

```json
{
  "id": "sb-94d290262ee8",
  "name": "test-for-doc",
  "type": "custom",
  "status": "Stopped",
  "configuration": {},
  "resources": {},
  "created_at": "",
  "updated_at": "",
  "image": "",
  "volume_mounts": [],
  "port_mappings": []
}
```

设置 `DOCKER_RT_INSPECT_MODE=standard` 可以保留标准 Docker inspect 字段。

#### 通过 Docker 参数指定 GPU 卡型号

`docker run --gpus` 只传 GPU 数量；不想设置 `DOCKER_RT_GPU_CARD` 时，可以用
`docker-rt.gpu-card` label 指定卡型号：

```bash
docker create \
  --name gpu-demo \
  --cpus 4 \
  --memory 8g \
  --gpus 1 \
  --label docker-rt.gpu-card=L40S \
  busybox:1.36 sleep 300
```

如果希望直接写 `--gpu-card L40S`，每次运行 `pyromind docker-rt` 都会询问是否
安装本地 docker wrapper。确认后安装 `~/.pyromind/bin/docker` 并写入 shell PATH；
拒绝仍会启动 docker-rt，只是不能使用 `--gpu-card` 简写，可以用
`--label docker-rt.gpu-card=L40S` 或 `DOCKER_RT_GPU_CARD`。

且wrapper没有安装的话，`docker ps` 和 `docker inspect` 返回的结构没有针对pyromind做优化

也可以手动安装：

```bash
pyromind docker-install
```

卸载 SDK 前先清理 wrapper：

```bash
pyromind docker-uninstall
# 或
pyromind-docker-uninstall
```

`pip uninstall` 没有卸载钩子，所以需要显式执行该命令删除
`~/.pyromind/bin/docker` 并清理 shell PATH 配置。

安装后重新打开终端即可使用：

```bash
docker create \
  --name gpu-demo \
  --gpus 1 \
  --gpu-card L40S \
  busybox:1.36 sleep 300
```

默认 `docker ps` 只显示 Running 的 sandbox；Stopped 的 sandbox 用
`docker ps -a` 查看。

docker wrapper 生效后，`docker ps` 表头会变成：
`ID / NAME / STATUS / PORTS / IMAGE`。

### Docker 命令参考

#### `docker run` / `docker create`

`docker run` = 创建并启动；
`docker create` = 只创建本地记录；
`docker start` = 真正调用 SDK 创建/启动 sandbox（Pending -> Running）。

| 参数 | 说明 | 示例 |
|------|------|------|
| `--name` | sandbox 名称 | `--name gpu-demo` |
| 镜像 | 容器镜像 | `busybox:1.36` |
| `--cpus` | CPU 数量 | `--cpus 4` |
| `--memory` | 内存大小 | `--memory 8g` |
| `--gpus` | GPU 数量 | `--gpus 1` |
| `--gpu-card` / `--gpu_card` | GPU 卡型号，需要 wrapper | `--gpu-card L40S` |
| `--label docker-rt.gpu-card=L40S` | GPU 卡型号，不需要 wrapper | `--label docker-rt.gpu-card=L40S` |
| `-p` / `--publish` | 端口映射 | `-p 8080:80` |
| `-v` / `--volume` | 目录挂载 | `-v /workspace:/data` |
| `-v ...:ro` | 只读挂载 | `-v /workspace:/data:ro` |
| `-e` / `--env` | 环境变量（k8s-middleware 暂不支持） | `-e FOO=bar` |
| `-w` / `--workdir` | 工作目录（k8s-middleware 暂不支持） | `-w /workspace` |
| `--tmpfs` | 临时内存盘（k8s-middleware 暂不支持） | `--tmpfs /tmp:rw` |

示例：

```bash
docker create \
  --name gpu-demo \
  --cpus 4 \
  --memory 8g \
  --gpus 1 \
  --label docker-rt.gpu-card=L40S \
  -p 8080:80 \
  -v /workspace:/data:ro \
  busybox:1.36 sleep 300

docker start gpu-demo
```

#### `docker ps` / `docker ps -a`

```bash
docker ps      # 只显示 Running
docker ps -a   # 显示 Running + Stopped
```

wrapper 生效时，表头为：

```text
ID  NAME  STATUS  RESOURCES  PORTS  VOLUMES  IMAGE
```

长字段自动截断显示 `...`，完整内容用 `docker inspect` 查看。

#### `docker inspect`

```bash
docker inspect gpu-demo
docker inspect gpu-demo --format '{{json .resources}}'
```

默认只返回 sandbox 字段；设置 `DOCKER_RT_INSPECT_MODE=standard` 可返回标准
Docker inspect 字段。

#### `docker exec`

```bash
docker exec gpu-demo echo hello
docker exec -w /workspace gpu-demo ls -la
```

非交互式 exec 已支持；`docker exec -it <name>` 复用
`/sandboxes/{id}/terminal`，进入 k8s_middleware 交互 shell。
原有的 `pyromind terminal <sandbox-id>` 子命令保持原参数和逻辑不变。

#### `docker logs`

```bash
docker logs gpu-demo
docker logs -f gpu-demo
```

`k8s_middleware` 后端暂未提供 `/logs` 接口，该功能当前依赖后端能力补齐。

#### `docker cp`

```bash
docker cp gpu-demo:/etc/os-release /tmp/os-release
docker cp /tmp/file.txt gpu-demo:/workspace/file.txt
```

#### `docker stop` / `docker start`

```bash
docker stop gpu-demo
docker start gpu-demo
```

`k8s_middleware` 后端下，stop 对应 pause，start 对应 resume。

#### `docker restart`

```bash
docker restart gpu-demo
```

映射为 pause 后 resume。

#### `docker rename`

```bash
docker rename gpu-demo gpu-demo-2
```

`k8s_middleware` 后端只改 name 时不会触发 Pod 滚动更新。

#### `docker rm`

```bash
docker rm -f gpu-demo
```

`k8s_middleware` 后端会先 pause 再 delete。

#### `docker port`

```bash
docker port gpu-demo
```

端口来自 k8s_middleware 的 `port_mappings`。

#### `docker events`

```bash
docker events --since 0s
```

当前为进程内事件流，重启 daemon 后历史事件不保留。

#### 不支持的 Docker 命令

启动 docker-rt 后，以下命令当前不支持：

```text
docker build
docker buildx build
docker compose build
docker compose up --build
```

这些命令依赖真实 Docker daemon / BuildKit 容器生命周期，docker-rt 不提供假实现。
建议先用正常 Docker/BuildKit 构建镜像并推送到 registry，再通过
`docker run` 使用该镜像。

链路：`Docker CLI -> docker-rt daemon -> KubeEnvironment -> Kubernetes API`。
当前实现由 `KubeEnvironment` 直接通过官方 Kubernetes Python SDK 调用集群；
如果希望 `k8s_middleware` 成为唯一后端，下一阶段需要把这一跳替换成
`k8s_middleware` HTTP API 适配器。

通过 `k8s_middleware` OpenAPI 运行：

```bash
DOCKER_RT_BACKEND=k8s-middleware \
PYROMIND_API_KEY=your-key \
PYROMIND_BASE_URL=https://api.pyromind.ai/api/v1 \
PYROMIND_CLUSTER=us-west-2 \
pyromind docker-rt
```

该模式下 docker-rt 使用 `PyromindSDK` 适配器：先读取当前 sandbox，合并修改字段，
再提交完整 sandbox 更新；本地端口访问继续保留 `PortForwarder`；
`k8s_middleware` 只改 `name` 时会跳过 StatefulSet 滚动更新。

## 配置

### 客户端参数

| 参数 | 必填 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `api_key` | 是* | `str` | `PYROMIND_API_KEY` 环境变量 | API 认证 Bearer Token |
| `cluster` | 否 | `str` | `PYROMIND_CLUSTER` 环境变量或 `"us-west-2"` | 目标集群（`X-Cluster` 请求头） |
| `timeout` | 否 | `int` | `30` | 请求超时时间（秒） |
| `max_retries` | 否 | `int` | `3` | 失败请求最大重试次数 |

\* `api_key` 可通过参数传入或设置 `PYROMIND_API_KEY` 环境变量。

### 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `PYROMIND_API_KEY` | 是 | — | API Bearer Token |
| `PYROMIND_CLUSTER` | 否 | `us-west-2` | 目标集群标识 |
| `PYROMIND_STORAGE_ENDPOINT` | 否 | `https://storage.pyromind.ai` | 存储端点 URL |
| `PYROMIND_STORAGE_SECRET_KEY` | 否 | — | 存储密钥 |
| `PYROMIND_STORAGE_BUCKET` | 否 | — | 默认存储桶名 |

## 项目结构

```
pyromind_sdk/
├── client/                          # API 客户端
│   ├── base.py                      # 基础 HTTP 客户端
│   ├── client.py                    # PyroMindAPIClient（统一入口）
│   ├── async_client.py              # PyroMindAsyncAPIClient（异步入口）
│   ├── studio.py / async_studio.py  # Studio / 训练任务
│   ├── jupyterLab.py / async_jupyterlab.py  # Jupyter 实例
│   ├── inference.py / async_inference.py    # 推理任务
│   ├── echomind.py / async_echomind.py      # EchoMind 实例
│   ├── storage.py                   # 文件存储
│   ├── profile.py                   # 用户信息与 SSH 密钥
│   ├── models.py                    # Pydantic 数据模型
│   └── workflow/                    # 工作流验证与转换
├── nodes/                           # 自定义节点 SDK
│   ├── function_call_wrapper.py     # Python 函数 → 节点
│   ├── python_function_executor.py  # Python 节点执行器
│   ├── python_to_yaml.py            # Python 转 YAML
│   └── yaml_loader.py               # YAML 节点加载器
├── common/                          # 公共工具
│   ├── constants.py
│   └── node_sdk.py
├── cli.py                           # CLI 入口
├── python_function_to_yaml_cli.py   # Python → YAML CLI 工具
├── examples/                        # 使用示例
│   └── openapi/                     # API 使用示例
└── tests/                           # 测试
```

## 服务

### Studio（`client.studio`）

训练工作流管理 — 创建、监控和管理工作流任务。

| 方法 | 输入 | 输出 | 描述 |
|--------|------|------|------|
| `list()` | — | `List[TrainingTaskResponse]` | 列出所有 Studio 任务 |
| `create(request)` | `TrainingTaskCreateRequest` | `TrainingTaskCreateResponse` | 创建训练任务 |
| `get_job(task_id)` / `get_task(task_id)` | `str` | `TrainingTaskResponse` | 获取任务详情 |
| `delete(task_id, force=False)` | `str`, `bool` | `None` | 删除任务 |
| `stop(task_id)` | `str` | `TrainingTaskResponse` | 停止运行中的任务 |
| `get_node_output(task_id, node_id)` | `str`, `str` | `Optional[Dict]` | 获取节点级输出 |
| `get_node_info(names=None)` | `Optional[str]` | `Dict[str, Any]` | 获取节点定义信息 |
| `reload_nodes(node_name=None)` | `Optional[str]` | `Dict[str, Any]` | 重新加载节点 YAML 定义 |
| `create_node(...)` | `yaml_path/yaml_content` + 选项 | `Dict[str, Any]` | 注册自定义节点 |
| `delete_node_by_name(node_name)` | `str` | `Dict[str, Any]` | 删除自定义节点 |
| `move_node(node_name, source_file_path)` | `str`, `str` | `Dict[str, Any]` | 移动节点源码路径 |
| `run_with_params(request)` | `WorkflowRunRequest` | `TrainingTaskCreateResponse` | 使用参数运行已存储的工作流 |
| `export_node_outputs(task_id, nodes_info, ...)` | `str`, `List`, `Optional[List]` | `List[Dict]` | 导出所有节点输出 |
| `wait_for_task_completion(task_id, ...)` | `str` + 选项 | `str` (状态) | 轮询直到任务结束 |
| `create_and_wait(request, ...)` | `TrainingTaskCreateRequest` + 选项 | `Dict[str, Any]` | 创建 + 轮询 + 可选导出输出 |

**`TrainingTaskCreateRequest` 参数说明：**

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `name` | 是 | `str` | 任务名称 |
| `workflow` | 是 | `Dict[str, Any]` | 工作流 JSON 结构，包含节点定义 |

**`WorkflowRunRequest` 参数说明：**

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `workflow_name` | 是 | `str` | 已存储工作流的名称 |
| `primitive_node_map` | 否 | `Dict[str, Any]` | 注入的原始节点值（默认 `{}`） |

**示例：**

```python
from pyromind_sdk.client.models import TrainingTaskCreateRequest, WorkflowRunRequest

# 创建训练任务
task = client.studio.create(
    TrainingTaskCreateRequest(
        name="my-workflow",
        workflow={"nodes": [...]}
    )
)
print(f"Task ID: {task.task_id}")

# 列出任务
tasks = client.studio.list()

# 使用参数运行工作流
result = client.studio.run_with_params(
    WorkflowRunRequest(workflow_name="my-workflow", primitive_node_map={"key": "value"})
)

# 等待完成
status = client.studio.wait_for_task_completion(task.task_id, timeout=600)
print(f"Final status: {status}")
```



### Jupyter（`client.jupyter`）

Jupyter 实例管理。

| 方法 | 输入 | 输出 | 描述 |
|--------|------|------|------|
| `list()` | — | `List[JupyterResponse]` | 列出所有 Jupyter 实例 |
| `create(request)` | `JupyterRequest` | `JupyterResponse` | 创建实例 |
| `get_instance(jupyter_id)` | `str` | `JupyterResponse` | 获取实例详情 |
| `update(jupyter_id, request)` | `str`, `JupyterRequest` | `JupyterResponse` | 更新实例配置 |
| `delete(jupyter_id)` | `str` | `None` | 删除实例 |
| `pause(jupyter_id)` / `resume(jupyter_id)` | `str` | `JupyterResponse` | 暂停/恢复 |

**`JupyterRequest` 参数说明：**

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `name` | 否 | `str` | 实例显示名称 |
| `resources` | 否 | `ResourceConfig` | CPU/内存/GPU 配置 |

**示例：**

```python
from pyromind_sdk.client.models import JupyterRequest, ResourceConfig

# 创建 Jupyter 实例
jupyter = client.jupyter.create(
    JupyterRequest(
        name="my-notebook",
        resources=ResourceConfig(cpu="4", memory="16Gi", gpu="1")
    )
)
print(f"Jupyter ID: {jupyter.id}, URL: {jupyter.url}")
```

### 推理（`client.inference`）

推理任务管理。

| 方法 | 输入 | 输出 | 描述 |
|--------|------|------|------|
| `list()` | — | `List[InferenceJobResponse]` | 列出所有推理任务 |
| `create(request)` | `InferenceJobRequest` | `str` (job_id) | 创建推理任务 |
| `get_job(job_id)` | `str` | `InferenceJobResponse` | 获取任务详情 |
| `update(job_id, request)` | `str`, `InferenceJobRequest` | `InferenceJobResponse` | 更新任务配置 |
| `delete(job_id)` | `str` | `None` | 删除任务 |
| `pause(job_id)` / `resume(job_id)` | `str` | `InferenceJobResponse` | 暂停/恢复 |
| `get_framework()` | — | `List[str]` | 列出可用框架 |
| `get_inf_image(framework)` | `str` | `List[str]` | 列出推理镜像 |

**`InferenceJobRequest` 参数说明：**

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `model_path` | 是 | `str` | 模型路径 |
| `inference_framework` | 否 | `str` | 推理框架（通过 `get_framework()` 获取） |
| `resources` | 否 | `ResourceConfig` | CPU/内存/GPU 配置 |
| `name` | 否 | `str` | 任务显示名称 |
| `inf_image` | 否 | `str` | 推理镜像（通过 `get_inf_image()` 获取） |
| `model_name` | 否 | `str` | 模型名称覆盖 |
| `model_length` | 否 | `int` | 模型上下文长度 |
| `startup_args` | 否 | `List[dict]` 或 `List[str]` | 自定义推理服务启动参数。推荐 `[{"--arg": value}]`；key 需要自己带 `-` 或 `--` 前缀；与系统默认参数重复时以用户参数为准 |

**示例：**

```python
from pyromind_sdk.client.models import InferenceJobRequest, ResourceConfig

# 列出可用框架和镜像
frameworks = client.inference.get_framework()
images = client.inference.get_inf_image(frameworks[0])

# 创建推理任务
job_id = client.inference.create(
    InferenceJobRequest(
        model_path="/path/to/model",
        inference_framework=frameworks[0],
        resources=ResourceConfig(cpu="8", memory="32Gi", gpu="1", gpu_card="H100"),
        startup_args=[{"--trust-remote-code": None}],
        name="my-inference"
    )
)
print(f"Job ID: {job_id}")

# 获取任务详情
job = client.inference.get_job(job_id)
print(f"Status: {job.status}")
```

### EchoMind（`client.echomind`）

EchoMind 实例生命周期管理。

| 方法 | 输入 | 输出 | 描述 |
|--------|------|------|------|
| `list()` | — | `List[EchoMindJobResponse]` | 列出所有 EchoMind 实例 |
| `create(request)` | `EchoMindJobRequest` | `str` (job_id) | 创建实例 |
| `get_job(job_id)` | `str` | `EchoMindJobResponse` | 获取实例详情 |
| `update(job_id, request)` | `str`, `EchoMindJobRequest` | `EchoMindJobResponse` | 更新实例配置 |
| `delete(job_id)` | `str` | `None` | 删除实例 |
| `pause(job_id)` / `resume(job_id)` | `str` | `EchoMindJobResponse` | 暂停/恢复 |

**`EchoMindJobRequest` 参数说明：**

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `name` | 否 | `str` | 实例显示名称 |
| `resources` | 否 | `ResourceConfig` | CPU/内存/GPU 配置 |

**示例：**

```python
from pyromind_sdk.client.models import EchoMindJobRequest, ResourceConfig

# 创建 EchoMind 实例
job_id = client.echomind.create(
    EchoMindJobRequest(
        name="my-echomind",
        resources=ResourceConfig(cpu="4", memory="16Gi")
    )
)
print(f"EchoMind ID: {job_id}")

# 列出实例
instances = client.echomind.list()

# 清理
client.echomind.delete(job_id)
```

### 存储（`client.storage`）

MinIO/S3 兼容文件存储。需要安装 `minio` 包（`pip install minio`）。

| 方法 | 输入 | 输出 | 描述 |
|--------|------|------|------|
| `list_files(folder_path, ...)` | `str` + 选项 | `List[Dict]` | 列出目录中的文件 |
| `file_exists(file_path)` | `str` | `bool` | 检查文件是否存在 |
| `upload_file(file_path, object_name, ...)` | `str/Path/BinaryIO` + 选项 | `Dict[str, Any]` | 上传文件（支持分片） |
| `upload_folder(folder_path, ...)` | `str/Path` + 选项 | `List[Dict]` | 上传整个文件夹 |
| `download_file(object_name, ...)` | `str` + 选项 | `Union[bytes, Path]` | 下载文件 |
| `download_folder(folder_path, local_path)` | `str`, `str/Path` + 选项 | `List[Dict]` | 下载文件夹 |
| `delete_file(object_name)` | `str` | `None` | 删除文件 |
| `delete_folder(folder_path)` | `str` + 选项 | `Dict` | 删除文件夹 |

**Storage 初始化参数说明：**

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `endpoint` | 否 | `str` | 存储端点（环境变量：`PYROMIND_STORAGE_ENDPOINT`，默认：`https://storage.pyromind.ai`） |
| `access_key` | 否 | `str` | 访问密钥（环境变量：`PYROMIND_API_KEY`） |
| `secret_key` | 否 | `str` | 密钥（环境变量：`PYROMIND_STORAGE_SECRET_KEY`） |
| `bucket_name` | 否 | `str` | 默认桶名（环境变量：`PYROMIND_STORAGE_BUCKET`） |
| `secure` | 否 | `bool` | 是否使用 HTTPS（自动从端点 URL 检测） |
| `region` | 否 | `str` | 存储区域（默认：`us-east-1`） |

**示例：**

```python
from pyromind_sdk.client.storage import StorageClient

storage = StorageClient()

# 列出文件
files = storage.list_files(folder_path="documents/")
for f in files:
    print(f"{f['object_name']} ({f['size']} bytes)")

# 上传文件
storage.upload_file("local/file.txt", "remote/file.txt")

# 下载文件
storage.download_file("remote/file.txt", "downloaded/file.txt")

# 检查文件是否存在
if storage.file_exists("remote/file.txt"):
    print("File exists")
```

### 用户信息（`client.profile`）

用户信息与 SSH 密钥管理。

| 方法 | 输入 | 输出 | 描述 |
|--------|------|------|------|
| `get_user_info(credit_info=False)` | `bool` | `ProfileUserInfoResponse` | 获取用户信息 |
| `get_access_key()` | — | `str` | 获取访问密钥 |
| `get_storage_info()` | — | `ProfileStorageInfoResponse` | 获取存储凭证 |
| `add_key(request)` | `UserPubKeyRequest` | `bool` | 添加 SSH 公钥 |
| `list_keys()` | — | `List[UserPubKey]` | 列出 SSH 公钥 |

**示例：**

```python
# 获取用户信息
user = client.profile.get_user_info()
print(f"User: {user.username}")

# 获取存储信息
storage_info = client.profile.get_storage_info()
print(f"已用: {storage_info.human_used_size} / 总量: {storage_info.human_total_size}")

# SSH 密钥管理
from pyromind_sdk.client.models import UserPubKeyRequest

client.profile.add_key(UserPubKeyRequest(key="ssh-ed25519 AAAA..."))
keys = client.profile.list_keys()
```

## 异步支持

所有服务均有对应的异步客户端 `PyroMindAsyncAPIClient`：

```python
from pyromind_sdk import PyroMindAsyncAPIClient

async with PyroMindAsyncAPIClient(api_key="your-api-key") as client:
    tasks = await client.studio.list()
    task = await client.studio.create(request)
```

异步客户端（方法集与同步版一致）：
- `client.studio` → `AsyncStudioClient`
- `client.instances` → `AsyncJupyterLabClient`
- `client.inference` → `AsyncInferenceClient`
- `client.echomind` → `AsyncEchoMindClient`

## 异常处理

所有 API 调用失败时抛出 `PyroMindAPIError`（同步）或 `PyroMindAsyncAPIError`（异步）：

```python
from pyromind_sdk.client.base import PyroMindAPIError

try:
    task = client.studio.get_task("invalid-id")
except PyroMindAPIError as e:
    print(f"Error {e.status_code}: {e.message}")
    if e.response:
        print(f"Response: {e.response}")
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `message` | `str` | 错误描述 |
| `status_code` | `Optional[int]` | HTTP 状态码 |
| `response` | `Optional[Dict]` | API 错误响应体 |

## 关键响应模型

每个服务返回结构化的 Pydantic 模型对象。主要字段如下：

### `TrainingTaskResponse`（Studio）

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | `str` | 任务唯一 ID |
| `name` | `str` | 任务名称 |
| `status` | `str` | 当前状态（`running`、`completed`、`failed` 等） |
| `workflow` | `Dict` | 工作流配置 |
| `nodes` | `List[TrainingTaskNodeInfo]` | 节点执行详情 |
| `error_message` | `Optional[str]` | 失败时的错误信息 |
| `created_at` | `datetime` | 创建时间戳 |

### `JupyterResponse`（Jupyter）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | 实例 ID |
| `name` | `str` | 实例名称 |
| `status` | `str` | 当前状态 |
| `url` | `Optional[str]` | Jupyter URL |
| `password` | `Optional[str]` | 访问密码 |

### `InferenceJobResponse`（推理）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | 任务 ID |
| `name` | `str` | 任务名称 |
| `model_path` | `str` | 模型路径 |
| `status` | `str` | 当前状态 |
| `endpoint_url` | `Optional[str]` | 推理端点 |
| `resources` | `Optional[ResourceConfig]` | 分配的资源 |

### `EchoMindJobResponse`（EchoMind）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | 实例 ID |
| `name` | `str` | 实例名称 |
| `status` | `str` | 当前状态 |

## 工作流验证与转换

`client/workflow/` 模块提供工作流验证和格式转换功能：

```python
from pyromind_sdk.client import validate_workflow, ValidationError

# 验证工作流结构
try:
    validate_workflow(workflow_dict)
    print("Workflow is valid")
except ValidationError as e:
    print(f"Invalid workflow: {e}")
```

| 工具 | 描述 |
|------|------|
| `validate_workflow(workflow)` | 验证工作流 JSON 结构 |
| `ValidationError` | 工作流无效时抛出的异常 |
| `converter.py` | 在工作流格式之间转换 |

## CLI 工具

| 命令 | 描述 |
|---------|-------------|
| `python -m pyromind_sdk.cli` | SDK CLI（多种工具） |
| `python -m pyromind_sdk.python_function_to_yaml_cli` | 将 Python 函数转换为 YAML 节点定义 |

## 自定义节点 SDK

除了 YAML 定义，SDK 还提供程序化节点创建工具：

**将 Python 函数包装为自定义节点：**

```python
from pyromind_sdk.nodes.function_call_wrapper import create_node_from_function

# 将任何函数装饰为节点定义
@create_node_from_function(
    name="my_custom_node",
    description="处理输入数据",
    category="data-processing"
)
def process_data(input_text: str, threshold: float = 0.5) -> dict:
    # 你的逻辑
    return {"result": "processed", "value": len(input_text)}
```

**运行时执行 Python 函数节点：**

```python
from pyromind_sdk.nodes.python_function_executor import execute_python_node

result = execute_python_node(
    source_code="print('hello')",
    node_type="python"
)
```

**将 Python 函数转换为 YAML 配置：**

```python
from pyromind_sdk.nodes.python_to_yaml import python_function_to_yaml_config

def my_func(input: str) -> str:
    return input.upper()

yaml_config = python_function_to_yaml_config(my_func)
# yaml_config 可以保存为 .yaml 文件并通过 studio.create_node() 注册
```

**验证和加载 YAML 节点定义：**

```python
from pyromind_sdk.nodes.yaml_loader import load_yaml_node
from pyromind_sdk.nodes.node_validator import validate_node_config

node_config = load_yaml_node("path/to/node.yaml")
validate_node_config(node_config)
```

## 测试

```bash
pytest
```

## 示例

| 示例 | 描述 |
|---------|-------------|
| `api_client_basic.py` | 基础客户端设置 |
| `studio_example.py` | Studio 任务 CRUD + 节点输出 |
| `studio_monitor.py` | 循环监控任务状态 |
| `workflow_cli.py` | 工作流管理 CLI 工具 |
| `complete_workflow_example.py` | 端到端工作流演示 |
| `jupyter_instance_example.py` | Jupyter 实例 CRUD |
| `inference_example.py` | 推理任务管理 |
| `echomind_example.py` | EchoMind 生命周期 |
| `storage_example.py` | 文件上传/下载 |
| `release_all_instance.py` | 批量释放资源 |
| `async_training_example.py` | 异步 Studio 训练 |
| `async_inference_example.py` | 异步推理 |
| `async_echomind_example.py` | 异步 EchoMind |
| `async_jupyter_instance_example.py` | 异步 Jupyter |

## 开发

### 从源码安装

```bash
git clone https://github.com/pyromind/pyromind-sdk.git
cd pyromind-sdk
pip install -e .
```
