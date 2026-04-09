# Quant Trading Project — Agent Team Refactoring Guidance

> 本文档定义了量化交易项目全盘重构的 Agent Team 架构、各成员职责、技能栈、工具链与协作规范。
> 目标：从策略逻辑、代码结构、训练/回测效率、数据管线、风控体系五个维度进行系统性重构。

---

## 1. 团队全景

```
                        ┌─────────────────┐
                        │   🏛️ Architect   │  ← 总架构师，握全局
                        │   (项目负责人)    │
                        └────────┬────────┘
                                 │
          ┌──────────────┬───────┴───────┬──────────────┐
          ▼              ▼               ▼              ▼
   ┌─────────────┐ ┌──────────┐ ┌─────────────┐ ┌───────────┐
   │ 📐 Strategist│ │ 🔧 Engineer│ │ ⚡ Optimizer │ │ 🛡️ RiskGrd │
   │  策略研究员   │ │  工程重构师 │ │  性能优化师  │ │  风控审计员 │
   └─────────────┘ └──────────┘ └─────────────┘ └───────────┘
          │              │               │              │
          ▼              ▼               ▼              ▼
   ┌─────────────┐ ┌──────────┐ ┌─────────────┐ ┌───────────┐
   │ 📊 DataEng  │ │ 🧪 QA     │ │ 📝 DocWriter│ │ 🔍 Reviewer│
   │  数据工程师   │ │  测试工程师 │ │  文档撰写员  │ │  Code审查员 │
   └─────────────┘ └──────────┘ └─────────────┘ └───────────┘
```

---

## 2. 角色定义

### 2.1 🏛️ Architect — 总架构师

**核心职责**
- 制定重构总体路线图（Roadmap），拆解为可并行的阶段性里程碑
- 审批所有架构级变更（模块拆分、依赖引入、接口协议）
- 在角色之间分配任务，协调优先级冲突
- 最终 PR merge 的 gatekeeper

**技能要求**
- 系统设计：微服务/单体权衡、依赖注入、事件驱动架构
- 量化领域知识：理解 alpha 因子、回测引擎、执行层、风控模块的耦合关系
- 项目管理：任务拆解、关键路径分析、风险识别

**工具与权限**
| 工具 | 用途 |
|------|------|
| GitHub Projects / Linear | 任务看板、里程碑追踪 |
| Mermaid / draw.io | 架构图绘制 |
| Claude Code (Opus) | 全局规划、设计文档生成 |
| 全部代码仓库 | 读写权限（唯一有 `main` 分支直推权的角色） |

**输出物**
- `ARCHITECTURE.md` — 系统架构总览
- `ROADMAP.md` — 分阶段重构计划
- `ADR/` 目录 — Architecture Decision Records

---

### 2.2 📐 Strategist — 策略研究员

**核心职责**
- 复盘现有策略逻辑，评估每条 alpha 因子的有效性与衰减周期
- 设计新的因子研究框架（因子挖掘 → IC 分析 → 组合优化）
- 定义策略配置规范：策略参数、信号生成、仓位管理的解耦接口
- 编写策略文档，确保非策略人员也能理解策略意图

**技能要求**
- 因子研究：横截面因子、时序因子、因子正交化、IC/IR 分析
- 组合优化：均值方差、风险平价、Black-Litterman
- 统计检验：多重检验校正（Bonferroni/FDR）、过拟合检测（CSCV）
- 市场微观结构：订单簿、滑点建模、冲击成本

**工具与权限**
| 工具 | 用途 |
|------|------|
| Jupyter / Marimo | 交互式因子研究 |
| Alphalens / QuantStats | 因子分析与绩效归因 |
| Claude Code (Sonnet) | 因子代码生成、统计检验脚本 |
| `strategies/` 目录 | 读写权限 |
| `data/` 目录 | 只读权限 |

**输出物**
- `strategies/README.md` — 策略目录与简介
- `strategies/{name}/spec.yaml` — 每条策略的配置规范
- `research/` — 因子研究 notebook 归档

---

### 2.3 🔧 Engineer — 工程重构师

**核心职责**
- 重构项目目录结构，建立清晰的模块边界
- 实现核心抽象层：`Strategy` / `Portfolio` / `Execution` / `DataFeed` 的基类与接口
- 统一配置管理（YAML/TOML + Pydantic validation）
- 引入依赖注入、消除全局状态和硬编码

**技能要求**
- Python 工程：`abc`、`Protocol`、`dataclass`/`pydantic`、`typing`
- 设计模式：Strategy Pattern、Observer、Pipeline、Registry
- 包管理：`pyproject.toml`、`uv`/`poetry`、monorepo 结构
- 异步编程：`asyncio`、事件循环、并发回测

**工具与权限**
| 工具 | 用途 |
|------|------|
| Claude Code (Sonnet/Opus) | 大规模代码重构、接口设计 |
| `ruff` / `mypy` | Linting + 类型检查 |
| `pytest` + `hypothesis` | 属性测试、边界测试 |
| 全部代码仓库 | 读写权限（feature branch） |

**输出物**
- 重构后的项目目录结构
- `core/` — 抽象基类与核心接口
- `pyproject.toml` — 统一依赖管理
- `CONTRIBUTING.md` — 代码规范与提交约定

**推荐目录结构**
```
quant_project/
├── pyproject.toml
├── src/
│   └── quant/
│       ├── core/              # 抽象层
│       │   ├── strategy.py    # Strategy Protocol
│       │   ├── portfolio.py   # Portfolio/Position 管理
│       │   ├── execution.py   # Order/Execution 接口
│       │   ├── data_feed.py   # DataFeed 抽象
│       │   └── risk.py        # Risk 检查接口
│       ├── strategies/        # 策略实现
│       ├── data/              # 数据获取与清洗
│       ├── backtest/          # 回测引擎
│       ├── optimize/          # 参数优化 / 训练
│       ├── live/              # 实盘执行层
│       ├── risk/              # 风控实现
│       ├── utils/             # 工具函数
│       └── config/            # 配置 schema
├── tests/
├── notebooks/
├── configs/                   # 运行时配置文件
├── data/                      # 本地数据缓存
└── docs/
```

---

### 2.4 ⚡ Optimizer — 性能优化师

**核心职责**
- Profiling 现有回测引擎，定位瓶颈（CPU-bound / IO-bound / 内存）
- 实施关键路径优化：向量化计算、内存映射、增量计算
- 设计并行回测方案（参数扫描、多标的、多时间段）
- 建立性能基准（benchmark suite），每次变更必须回归

**技能要求**
- 高性能 Python：`numpy` 向量化、`numba` JIT、`polars` / `pandas` 优化
- 并行/分布式：`multiprocessing`、`ray`、`dask`、`joblib`
- 内存优化：`mmap`、`numpy.memmap`、dtype 压缩、`__slots__`
- Profiling：`cProfile`、`line_profiler`、`py-spy`、`memray`

**工具与权限**
| 工具 | 用途 |
|------|------|
| `py-spy` / `memray` | CPU + 内存 profiling |
| `pytest-benchmark` | 回测速度回归测试 |
| `ray` / `dask` | 分布式参数扫描 |
| Claude Code (Sonnet) | 向量化重写、Numba 加速 |
| `backtest/` + `optimize/` | 读写权限 |

**输出物**
- `benchmarks/` — 性能基准测试套件
- `PERFORMANCE.md` — 优化记录与前后对比
- 优化后的回测引擎核心模块

**性能目标参考**
| 指标 | 重构前基线 | 目标 |
|------|-----------|------|
| 单标的 1 年日频回测 | 测量后填入 | < 1s |
| 10,000 组参数扫描 | 测量后填入 | < 10min (8 核) |
| 内存峰值（单次回测） | 测量后填入 | 减少 50%+ |

---

### 2.5 📊 DataEng — 数据工程师

**核心职责**
- 设计统一数据层：行情数据、因子数据、交易数据的存储与访问接口
- 实现数据管线：获取 → 清洗 → 对齐 → 缓存 → 提供
- 处理数据质量问题：缺失值、复权、除权、合约换月
- 建立数据版本管理，确保回测可复现

**技能要求**
- 时序数据库：`Arctic` / `DuckDB` / `QuestDB` / `Parquet + DeltaLake`
- API 集成：IBKR TWS/Gateway、Tushare、AKShare、Yahoo Finance
- 数据处理：`polars`/`pandas`、数据对齐、timezone 处理
- 缓存策略：本地 Parquet 缓存、增量更新、TTL 管理

**工具与权限**
| 工具 | 用途 |
|------|------|
| DuckDB / Parquet | 本地分析型存储 |
| `polars` | 高性能数据处理 |
| IBKR API / AKShare | 数据源接入 |
| Claude Code (Sonnet) | ETL 脚本生成 |
| `data/` 目录 | 读写权限 |

**输出物**
- `data/README.md` — 数据源目录与字段说明
- `DataFeed` 实现类（对接 `core/data_feed.py` 接口）
- 数据质量检查脚本

---

### 2.6 🛡️ RiskGuard — 风控审计员

**核心职责**
- 审计策略的风险暴露：最大回撤、VaR/CVaR、尾部风险
- 设计风控规则引擎：仓位限制、止损、波动率调仓、相关性约束
- 实现实盘安全机制：下单频率限制、异常检测、kill switch
- 审查所有涉及资金安全的代码变更

**技能要求**
- 风险度量：VaR、CVaR、Calmar Ratio、Sortino、最大回撤分析
- 风控规则：硬性约束（仓位上限）、软性约束（动态止损）
- 实盘安全：幂等性、断线重连、订单状态机
- 合规意识：期货保证金计算、涨跌停板处理

**工具与权限**
| 工具 | 用途 |
|------|------|
| `quantstats` / `empyrical` | 风险指标计算 |
| Claude Code (Opus) | 风控逻辑审查、边界条件分析 |
| `risk/` + `live/` 目录 | 读写权限 |
| 实盘配置 | **只读** — 不可修改实盘参数 |

**输出物**
- `risk/rules.yaml` — 风控规则定义
- `RISK_AUDIT.md` — 审计报告
- Kill switch 与熔断机制实现

---

### 2.7 🧪 QA — 测试工程师

**核心职责**
- 建立测试金字塔：单元测试 → 集成测试 → 回归测试 → 端到端回测对比
- 设计"已知答案测试"（Known Answer Tests）：用历史数据验证回测引擎正确性
- 实现 CI/CD pipeline：每次提交自动跑测试 + 性能基准
- 数值精度验证：浮点累积误差、资金计算精度

**技能要求**
- 测试框架：`pytest`、`hypothesis`（属性测试）、`factory_boy`（测试数据）
- CI/CD：GitHub Actions、`pre-commit`、`tox`
- 数值测试：`numpy.testing`、`math.isclose`、精度断言
- 覆盖率：`coverage.py`、分支覆盖、mutation testing

**工具与权限**
| 工具 | 用途 |
|------|------|
| `pytest` + `hypothesis` | 测试编写与运行 |
| GitHub Actions | CI/CD 流水线 |
| `coverage.py` | 覆盖率分析 |
| Claude Code (Sonnet) | 测试用例生成 |
| 全部代码仓库 | 只读（仅 `tests/` 目录写权限） |

**输出物**
- `tests/` — 完整测试套件
- `.github/workflows/ci.yml` — CI 配置
- 测试覆盖率报告

**覆盖率目标**
| 模块 | 最低覆盖率 |
|------|-----------|
| `core/` | 95% |
| `backtest/` | 90% |
| `risk/` | 95% |
| `strategies/` | 80% |
| `data/` | 75% |

---

### 2.8 📝 DocWriter — 文档撰写员

**核心职责**
- 撰写项目级文档：README、快速上手指南、部署手册
- 为每个模块编写 API 文档（docstring 规范 + 自动生成）
- 维护 CHANGELOG 与版本发布说明
- 绘制策略文档中的公式与流程图

**工具与权限**
| 工具 | 用途 |
|------|------|
| MkDocs / Sphinx | 文档站点生成 |
| Mermaid | 流程图 |
| Claude Code (Sonnet) | 文档内容生成 |
| 全部代码仓库 | 只读（仅 `docs/` 目录写权限） |

---

### 2.9 🔍 Reviewer — Code 审查员

**核心职责**
- 审查所有 PR，重点关注：接口一致性、边界条件、安全隐患
- 检查是否引入"未来信息泄漏"（look-ahead bias）
- 确保代码风格一致（ruff 配置 + 自定义规则）
- 标记技术债务，维护 `TECH_DEBT.md`

**工具与权限**
| 工具 | 用途 |
|------|------|
| GitHub PR Review | 代码审查 |
| `ruff` / `mypy` / `bandit` | 静态分析 |
| Claude Code (Opus) | 深度代码审查 |
| 全部代码仓库 | 只读 + PR 评论权限 |

**审查清单（Checklist）**
- [ ] 无 look-ahead bias（未来信息泄漏）
- [ ] 无硬编码的魔法数字
- [ ] 类型标注完整
- [ ] 异常处理合理（不吞异常）
- [ ] 配置与逻辑分离
- [ ] 有对应的测试用例
- [ ] 文档/docstring 已更新

---

## 3. 协作规范

### 3.1 分支策略

```
main ─────────────────────────────────────── (仅 Architect merge)
  └── dev ────────────────────────────────── (集成分支)
        ├── feat/refactor-core-interfaces
        ├── feat/vectorize-backtest-engine
        ├── feat/unified-data-layer
        ├── feat/risk-rule-engine
        └── fix/look-ahead-bias-in-ma
```

### 3.2 任务流转

```
Architect 创建任务
    → 指派给对应角色
    → 角色在 feature branch 开发
    → 提交 PR
    → Reviewer 审查 + 相关角色 approve
    → Architect 最终 merge
```

### 3.3 沟通协议

| 场景 | 渠道 | 格式 |
|------|------|------|
| 架构决策 | ADR 文档 | `ADR-{NNN}: {Title}` |
| 日常讨论 | PR 评论 / Issue | 自由格式 |
| 跨角色依赖 | Issue 标签 `blocked-by:` | 关联 Issue ID |
| 性能报告 | `benchmarks/` 目录 | 标准化 JSON + markdown |


---

## 4. 技术栈统一

### 4.1 核心依赖

| 领域 | 选型 | 备注 |
|------|------|------|
| 数据处理 | `polars` (主) / `pandas` (兼容) | 新代码优先 polars |
| 数值计算 | `numpy` + `numba` | 热路径用 numba 加速 |
| 配置管理 | `pydantic` + YAML | 强类型校验 |
| 回测引擎 | 自研（基于事件驱动/向量化混合） | 见 Phase 3 |
| 可视化 | `plotly` / `matplotlib` | 交互式优先 |
| 测试 | `pytest` + `hypothesis` | 属性测试覆盖核心逻辑 |
| Lint/Format | `ruff` (all-in-one) | 替代 black + isort + flake8 |
| 类型检查 | `mypy` (strict mode) | 渐进式引入 |
| 包管理 | `uv` | 快速依赖解析 |
| CI | GitHub Actions | PR 级自动化 |

### 4.2 Python 版本

- **最低要求**: Python 3.11+
- **推荐**: Python 3.12（性能改进 + 更好的错误提示）

### 4.3 Claude Code 模型分配

| 模型 | 分配给 | 用途 |
|------|--------|------|
| **Opus** | Architect, RiskGuard, Reviewer | 架构决策、安全审计、深度 Code Review |
| **Sonnet** | Engineer, Optimizer, Strategist, DataEng, QA, DocWriter | 日常编码、重构、测试生成 |

---

## 5. 质量门禁（Quality Gates）

每个阶段结束前，必须通过以下检查：

```yaml
quality_gates:
  code:
    - ruff check --fix .         # 零 warning
    - mypy src/ --strict         # 零 error（渐进式放宽）
    - pytest --cov=src/ -q       # 覆盖率达标
    - pytest benchmarks/ -q      # 性能无回退

  strategy:
    - no_look_ahead_bias: true   # 无未来信息泄漏
    - known_answer_test: pass    # 已知答案测试通过
    - backtest_consistency: |    # 新旧引擎结果一致
        max_pnl_diff < 0.01%

  risk:
    - kill_switch_tested: true   # Kill Switch 功能验证
    - max_position_enforced: true
    - order_rate_limit: true

  docs:
    - readme_updated: true
    - api_docs_generated: true
    - changelog_updated: true
```

---

## 6. 关键原则

1. **"No Look-Ahead" 是铁律** — 任何代码变更必须通过 look-ahead bias 检查，这是量化项目最致命的错误。

2. **可复现性优先** — 回测结果必须在相同配置 + 相同数据下 100% 可复现。随机种子固定，数据版本化。

3. **配置与逻辑分离** — 策略参数、运行环境、数据源全部外部化为 YAML，代码中零硬编码。

4. **渐进式重构** — 每个 Phase 结束后系统必须可运行。不允许"推倒重来"式重构。

5. **性能有基线** — 任何变更不能导致回测速度回退超过 5%，除非有明确的架构收益且记录在 ADR 中。

6. **风控是底线** — 所有涉及实盘资金的代码变更，必须经过 RiskGuard + Reviewer 双重审查。

7. **测试驱动** — 核心模块（core/backtest/risk）新增代码必须同时提交测试用例。

---

## 7. 启动清单

Agent Team 正式开工前的 Day 0 检查：

- [ ] 现有代码已 fork 到重构仓库
- [ ] 所有角色已 clone 并能本地运行现有项目
- [ ] 性能基线已测量并记录在 `benchmarks/baseline.json`
- [ ] 现有策略列表已梳理，每条标注 active/deprecated
- [ ] 数据源访问权限确认（IBKR API key / 其他数据源 token）
- [ ] CI pipeline 已搭建（至少能跑 lint + test）
- [ ] `requirement/ROADMAP.md` 已由 Architect 发布第一版
- [ ] 全员已阅读并确认本 Guidance 文档

---

*Last updated: 2026-03-30*
*Maintained by: Architect*