# Setup —— 环境与日常命令

> 一句话：`make setup` 一键就绪，`uv` 管依赖，`make` 管任务。

---

## 1. 一键安装

```bash
cd inferlite
make setup
```

这一条命令会自动完成：

1. 检测 `uv` 是否存在 → 没有就优先 `brew install uv`，否则走 astral 官方脚本
2. `uv sync` 创建 `.venv`、安装 Python 3.11、装齐所有依赖（首次约 30s）
3. 健康检查：打印 Python / torch / transformers 版本 + MPS/CUDA 可用性

完成后无需 `source .venv/bin/activate`，所有命令用 `uv run ...` 即可（自动注入 venv）。

---

## 2. 工具角色分工

| 工具 | 角色 | 何时直接用 |
| --- | --- | --- |
| **uv** | 包管理器（取代 pyenv + venv + pip + pip-tools） | 加依赖、锁定版本 |
| **make** | 任务运行器（取代一堆 `./scripts/*.sh`） | 日常 test / lint / fmt / clean |
| **ruff** | 静态检查 + 格式化（取代 flake8 + black + isort） | `make lint` / `make fmt` |
| **pytest** | 测试 | `make test` |
| **mypy** | 类型检查 | `make typecheck` |

### 2.1 uv 是什么

[Astral](https://astral.sh) 出品、Rust 写的 Python 包管理器，2024 年起成为主流。

- **快**：装 PyTorch + transformers 这种重依赖，pip 60s，uv 5s
- **一站式**：管 Python 版本、虚拟环境、依赖锁定，一个工具搞定四件套
- **lockfile 默认开**：`uv.lock` 跨机器复现，commit 进 git
- **PyTorch 友好**：原生支持索引源切换（cpu / cu121 / cu124）

### 2.2 make 是什么

Unix 自带的任务运行器（1976 年就有，Mac/Linux 默认装好），用 `Makefile` 定义"任务名 → shell 命令"。

```makefile
setup:                       # 任务名 (target)
	bash scripts/setup.sh    # 执行命令 (注意 Tab 缩进，不能用空格)

test:
	uv run pytest
```

- `make setup` 等价于 `bash scripts/setup.sh`
- `make test` 等价于 `uv run pytest`

价值：① 一句话取代一长串命令；② 任务名跨项目统一约定（`test/lint/clean` 全行业通用）；③ 不用再写一堆 shell 脚本

---

## 3. 常用命令速查

```bash
# 一次性
make setup                            # 一键装 uv + 同步依赖 + 健康检查

# 日常
make sync                             # 重新同步依赖（拉一遍 uv.lock）
make test                             # 跑全部测试
make test-fast                        # 跳过 @pytest.mark.slow
make lint                             # ruff 检查
make fmt                              # ruff 格式化
make typecheck                        # mypy
make clean                            # 清 .venv 和缓存
make help                             # 列出所有目标

# 加依赖
uv add "torch>=2.5"                   # 主依赖
uv add --dev pytest-xdist             # dev 依赖
uv lock --upgrade transformers        # 升级单个包

# 跑 CLI / 单文件
uv run python -m inferlite.cli "你好"
uv run python tests/unit/test_rmsnorm.py
```

---

## 4. 为什么不用 conda / poetry / pip

| 工具 | 不用的原因 |
| --- | --- |
| conda / mamba | 慢、生态分裂；PyTorch 在 pip 渠道更新更快 |
| poetry | 比 uv 慢一个数量级，锁定算法偶发 hang |
| pdm | 用户基数小 |
| 裸 pip + venv | 无 lockfile，跨机器不可复现 |

---

## 5. 大坑预警

### 5.1 Mac MPS + PyTorch 版本

- M1–M5 阶段全程用 `torch.float32` + MPS，不要碰 fp16/bf16（MPS 后端某些算子有坑）
- transformers 必须 ≥ 4.51（更早版本没有 `Qwen3Config`）

### 5.2 zsh 把 `#` 当文件名

```bash
# ❌ 这样会报错
curl -LsSf https://astral.sh/uv/install.sh | sh   # 一次性安装

# ✅ 注释单独起一行
# 一次性安装
curl -LsSf https://astral.sh/uv/install.sh | sh
```

不过你不会遇到 —— `make setup` 已经替你处理好了。

### 5.3 Makefile 的 Tab

`Makefile` 里执行命令那一行**必须用 Tab**，不能用空格。否则报错：
```
Makefile:3: *** missing separator.  Stop.
```

VSCode 默认会按 `.editorconfig` / 文件类型识别成 Tab，问题不大。

---

## 6. 新人 onboarding 完整流程

```bash
git clone git@github.com:luhao2013/inferlite.git
cd inferlite
make setup        # 装 uv + sync 依赖 + 健康检查
make test         # 跑测试，应全绿
```

成功后即可开始 M1 的 `inferlite/model/layers.py`。
