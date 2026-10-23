# BioTRACE

从生物医学论文表格中抽取整表级知识链。一条知识链包含实验条件、结果指标，以及基因、药物、细胞、疾病等实体。

方法、数据和实验结果写在论文里：[paper/BioTRACE.pdf](paper/BioTRACE.pdf)。

本仓库只保留可运行的流水线代码和这篇论文。原始表格、模型权重、训练日志和实验输出没有放进来。

## 流水线

代码在 `pipeline/`。各步按顺序衔接，后一步读取前一步的输出。

| 目录 | 作用 |
| --- | --- |
| `step1` | 从论文解析结果中抽出表格图片，并用 RapidTable 转成 Markdown |
| `step2` | 抽样表格，统计实体类型覆盖 |
| `step3` | 判断是不是数据表，以及是否包含目标实体 |
| `step4` | 把 Markdown 表转成统一 JSON |
| `step5` | 生成银标知识链，并划分训练与测试 |
| `step6` | SFT-1：让模型看表写出知识链 |
| `step7` | 用教师模型做多轮修正，记录轨迹 |
| `step8` | 多轮 SFT-2，以及 API 基线 |
| `step8_1` | 失败样本重跑 |
| `step9` | PPO 继续训练 |

## 环境

Python 3.10 及以上。各步依赖不完全相同，常用的有：

```text
requests
rapid_table
torch
transformers
peft
trl
```

复制环境变量文件并填入自己的接口地址和密钥：

```powershell
copy .env.example .env
```

| 变量 | 用途 |
| --- | --- |
| `LLM_API_URL` | OpenAI 兼容的对话接口，供抽样、教师修正和 API 基线使用 |
| `QWEN_API_URL` | 本地或远程 Qwen 接口，供表格筛选和 JSON 生成使用 |
| `GPT5_API_KEY` / `QWEN_API_KEY` / `DEEPSEEK_API_KEY` | 对应接口的密钥 |
| `MODEL_PATH` | 本地 SFT 或合并后的模型目录 |
| `TABLE_IMAGE_DIR` | 步骤 1 的表格图片目录 |

密钥只放在本机 `.env`。`.gitignore` 已经忽略它，也忽略了 `data/`、`checkpoints/`、`logs/` 和 `output/`。

## 准备数据

把论文解析后的表格放到 `data/step1/output/`：

```text
data/step1/output/table_images/
data/step1/output/table_captions/
```

在仓库根目录执行步骤 1B：

```bash
bash pipeline/step1/run_step1b.sh
```

后续步骤的默认输入也在 `data/` 下，例如 `data/step1/output/table_markdown_1/` 和 `data/step3/output`。训练脚本通过 `MODEL_PATH` 读取本地模型，默认是 `data/checkpoints/step8_merged`。
