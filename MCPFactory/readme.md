### MCPFactory RL Training with VeRL
#### Overview
We perform RL training based on the [VeRL](https://github.com/verl-project/verl). We modify some VeRL code **inplace**. Here is our adaptation details:
- [x] Add `MCPClientManager` to manager for high concurrent MCP requests and isolation designs.
- [x] Customize the `ToolAgentLoop` to fix potential bugs and adapt to our own `MCPClientManager`.
- [x] Adapt the `RLHFDataset` to fix potential bugs and adapt to our dataset.
- [x] Add customized reward in `verl/verl/utils/reward_score/mcp_factory.py` to compute trace reward, state reward, and penalty.
- [x] Add `UserInteraction` for conversational tool-call scenarios.

#### Installation
```
git clone https://github.com/RolandXMR/verl
git switch release/v0.6.1 # use VeRL v0.6.1

conda create -n verl python=3.12
conda activate verl

cd verl

USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .

pip install jsonlines,fastmcp,numpy==2.0.0
```
* Please refer to the [VeRL Installation](https://verl.readthedocs.io/en/latest/start/install.html) for details

#### Data Preparation

#### Quick Start