# ProgressAwareBeliefUpdate

Official implementation of **PABU: Progress-Aware Belief Update for Efficient LLM Agents**.

---

## Overview

Large Language Model (LLM) agents are typically conditioned on full action–observation histories. While expressive, such histories often contain task-irrelevant information, leading to redundant actions and increased inference cost.

**PABU (Progress-Aware Belief Update)** introduces a compact belief-state framework that explicitly models *task progress* and selectively retains only informative past interactions. At each step, the agent predicts relative progress since the previous round and decides whether the new interaction should be stored. Future decisions are conditioned solely on this retained subset, rather than the full history.

Across eight environments in the **AgentGym** benchmark, using identical training trajectories, PABU achieves:

* **81.0% average task completion rate**, outperforming prior state-of-the-art full-history belief models by **23.9%**
* **26.9% fewer interaction steps on average** (9.5 vs. 13.0), demonstrating substantially improved efficiency

Ablation studies further show that *both explicit progress prediction and selective retention* are essential for robust belief learning and performance gains.

---

## Agent Environments

All environments are adopted from **AgentGym**:
[https://github.com/WooooDyy/AgentGym/blob/main/README.md](https://github.com/WooooDyy/AgentGym/blob/main/README.md)

We provide **stable parallel evaluation** for each environment. Stability is ensured through bug fixes to the original repositories and the use of deterministic random seeds. Users may directly rely on AgentGym train/test IDs for reproducible research.

> **Note**: We strongly recommend creating **one separate conda environment per agent environment**, as dependencies and launch procedures may differ.

### Environment Installation

```bash
conda create --name [envname] python=[python_version]  # see README.md in each env folder
conda activate [envname]
[env_function] --port [port_number] --instances [instance_number]
```

> ⚠️ Some environments require specifying a starting location or additional runtime arguments.

---

## PABU Training and Evaluation

### Code Environment Setup

```bash
conda create --name PABU python=3.11
conda activate PABU
pip install -r scripts/requirements.txt
accelerate config  # configure based on GPU capacity; see scripts/reference_config.yaml
cd scripts
```

---

## Training Data and Environment Scope

PABU training and evaluation are conducted on a curated subset of **AgentTraj-L**, augmented with:

* Explicit **progress supervision**
* **Step-level action augmentation**

We also remove trajectories that:

* Never yield a success signal
* Start in a terminal (reward = 1) state

The full environment scope is summarized below.

| Environment | Traj  | Steps  | Eval | Original Repo                                            | Env Server                                                                              |
| ----------- | ----- | ------ | ---- | -------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| MAZE        | 10    | 528    | 25   | [MAZE-Repo](https://github.com/abdulhaim/LMRL-Gym)       | [agentenv-lmrlgym](https://github.com/WooooDyy/AgentGym/tree/main/agentenv-lmrlgym)     |
| Wordle      | 955   | 7,134  | 25   | [Wordle-Repo](https://github.com/abdulhaim/LMRL-Gym)     | [agentenv-lmrlgym](https://github.com/WooooDyy/AgentGym/tree/main/agentenv-lmrlgym)     |
| ALFWorld    | 2,164 | 39,775 | 200  | [ALFWorld-Repo](https://github.com/alfworld/alfworld)    | [agentenv-alfworld](https://github.com/WooooDyy/AgentGym/tree/main/agentenv-alfworld)   |
| SciWorld    | 1,986 | 63,855 | 200  | [SciWorld-Repo](https://github.com/allenai/ScienceWorld) | [agentenv-sciworld](https://github.com/WooooDyy/AgentGym/tree/main/agentenv-sciworld)   |
| BabyAI      | 761   | 7,022  | 90   | [BabyAI-Repo](https://github.com/mila-iqia/babyai)       | [agentenv-babyai](https://github.com/WooooDyy/AgentGym/tree/main/agentenv-babyai)       |
| TextCraft   | 374   | 5,084  | 100  | [TextCraft-Repo](https://github.com/archiki/ADaPT)       | [agentenv-textcraft](https://github.com/WooooDyy/AgentGym/tree/main/agentenv-textcraft) |
| Weather     | 311   | 3,312  | 20   | [Weather-Repo](https://github.com/hkust-nlp/AgentBoard)  | [agentenv-tool](https://github.com/WooooDyy/AgentGym/tree/main/agentenv-tool)           |
| Movie       | 215   | 1,682  | 20   | [Movie-Repo](https://github.com/hkust-nlp/AgentBoard)    | [agentenv-tool](https://github.com/WooooDyy/AgentGym/tree/main/agentenv-tool)           |

---

## Evaluation

```bash
bash evaluation.sh
```

---

## Training

```bash
bash training.sh
```

---

## Pretrained Checkpoints (Fallback)

If you prefer to skip training and directly evaluate pretrained models:

```bash
hf auth
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download HunterJiang97/PABU-Agent-8B
hf download HunterJiang97/PABU-Data
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
Appearing Soon!
```