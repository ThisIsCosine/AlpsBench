## Task2: Memory Update / Conflict Resolution（Generator + Pipeline）

本目录实现 Task2 的 **probe/dataset 生成**、**模型评测** 与 **打分**。核心入口是 `run_task2_pipeline()`（见 `pipeline.py`），以及被它调用的 `Task2Generator.generate_probe()`（见 `generator.py`）。

---

## 你会得到什么输出（runs 目录）

每次运行会在 `runs/task2_batch/<timestamp>/`（或你指定的 `output_dir`）写出：

- **`dataset.jsonl`**：最终可用的 Task2 dataset（Task1-style entry）
  - `record.old_dialogue` / `record.new_dialogue`
  - `memory`：旧 memory 列表
  - `query`：Task2 system prompt（memory updater 规则）
  - `answer`：gold updated memory（= `expected_updated_memory_items`）
  - `metadata.controls`：本次生成的控制参数（fake/new/conflict 等）
- **`probes.jsonl`**：每个 seed 的 probe（更完整，包含 old/new dialogue、old memory、gold memory、metadata）
- **`events.jsonl`**：生成/评测过程日志（含 API 调用 start/done、错误等）
- **`errors.jsonl`**：异常记录（生成/评测阶段）
- **`reports.jsonl`**：评测模型的 raw 输出与解析后的 `memory_items`（如果你启用了评测）
- **`scores.jsonl`**：每个模型预测与 gold 的匹配分数（如果你启用了评测）

> 说明：gold（`answer`）**只包含 old memory + conflict replacement + new memory**；fake memory 仅作为对话噪声出现，不会进 gold。

---

## seed 输入格式（关键：如何让 conflict 生效）

`Task2Generator` 触发 conflict 段需要 seed 同时满足：

- `record["memory_items"]`（或兼容字段 `memories` / `past_memory`）存在且为 list
- `record["selected_memory_id"]` 存在
- `memory_items` 里能找到 `memory_id == selected_memory_id` 的那条

推荐直接用你已经准备好的目录：

- `data/wildchat/memories/selected/task4_final_with_selected_memory_id`

它是从 `task4_final` 生成的 “带 `selected_memory_id`” 版本（脚本见 `scripts/add_selected_memory_id_task4_final.py`）。

---

## 生成逻辑概览（3 段式 new_dialogue）

`Task2Generator.generate_probe()` 会生成 `new_dialogue`，按顺序拼成：

- **NoPersonalSegment（噪声段）**
  - 目标：生成不产生用户长期记忆的对话噪声
  - 可选：注入 fake memory（见 `task2_controls.fake_memory`）
  - 强制：该段 `memory_delta` 被置空（不会影响 gold）
- **UpdateSegment（新增段）**
  - 目标：引入 **恰好 M 条**新用户记忆（prefer indirect/implicit）
  - 产出：`m_new_1 ... m_new_M`（gold 会追加这些新 items）
- **ConflictSegment（冲突/更正段，可选）**
  - 目标：对 `selected_memory_id` 对应的旧记忆给出新证据并更正
  - 产出：`{selected_memory_id}_updated`（gold 中对该条做 1-for-1 replacement）

---

## 控制参数 `task2_controls`

通过 `api_config["task2_controls"]` 传入（pipeline 会转发给 generator）：

- **`no_personal_turns`**：NoPersonalSegment 生成的 turn 数（默认 6）
- **`fake_memory`**：NoPersonalSegment 中注入的噪声信息强度（默认 0）
  - 这是“非用户信息”的干扰（例如 unrelated trivia / hypothetical），**不进入 gold**
- **`new_memory`**：UpdateSegment 生成的新 memory 条数（默认 1）
- **`conflict_memory`**：是否生成冲突段（默认 0；>0 才启用）

> 注意：`fake_memory` 目前是 “prompt 软控制”，不保证严格等于 N 条；若你需要严格计数，建议扩展输出 schema 做显式校验。

---

## 最常用：只生成 10 条 dataset（不跑评测模型）

在仓库根目录（PowerShell）运行：

```powershell
python -c "import json; from src.agents.task_runner import run_task2_from_jsonl_dir; cfg=json.load(open('configs/api.json','r',encoding='utf-8')); cfg['task2_controls']={'no_personal_turns':8,'fake_memory':3,'new_memory':1,'conflict_memory':1}; run_task2_from_jsonl_dir(jsonl_dir=r'data\wildchat\memories\selected\task4_final_with_selected_memory_id', model_list=[], gen_call_model=None, eval_call_model=None, max_samples=10, max_attempts_per_seed=2, generator_model_id='gpt-5.2', api_config=cfg)"
```

说明：
- `model_list=[]` 表示 **不跑评测**（只生成 `dataset.jsonl` / `probes.jsonl` 等）
- 会在 `runs/task2_batch/<timestamp>/` 生成 run 目录

---

## 生成后怎么快速验证（conflict/new 是否生效）

（PowerShell，一行统计）

```powershell
python -c "import json,glob; ds=sorted(glob.glob(r'runs\\task2_batch\\*\\dataset.jsonl'))[-1]; rows=[json.loads(l) for l in open(ds,'r',encoding='utf-8') if l.strip()]; ok=sum(1 for e in rows if (e.get('metadata',{}).get('selected_memory_id') and any(m.get('memory_id')==f\"{e['metadata']['selected_memory_id']}_updated\" for m in (e.get('answer') or []) if isinstance(m,dict)))); ok_new=sum(1 for e in rows if any((m.get('memory_id','').startswith('m_new_') for m in (e.get('answer') or []) if isinstance(m,dict)))); print('dataset',ds,'n',len(rows),'conflict_ok',ok,'new_ok',ok_new)"
```

你也可以直接打开 `dataset.jsonl` 任意一条：
- `metadata.selected_memory_id` 应为非空
- `answer` 中应包含：
  - `"{selected_memory_id}_updated"`（conflict replacement）
  - `m_new_1`（以及更多 `m_new_*`，取决于 `new_memory`）

---

## 

- **目录里有 `_*.jsonl` 报告文件**：例如 `_selected_memory_id_report.jsonl` 这种不是 benchmark record。
  - 本仓库的 `list_jsonl_files()` 会默认跳过 `_*.jsonl`，避免把 report 当 seeds 读入。

