## Modification
```
📁 EnvFactory/
├── 📁 configs/
│   ├── 📄 env_factory_config.yaml
│   ├── 📄 mcp_server.json
│   └── 📄 system_prompt.py
├── 📁 manager/
│   └── 📄 mcp_client_manager.py
├── 📁 reward/
│   └── 📄 tool_reward_rule.py
└── 📁 tools/

📁 verl/
├── 📁 experimental/
│   ├── 📁 agent_loop/
│   │   └── 📄 tool_agent_loop.py
│   └── 📁 reward_loop/
│       └── 📁 reward_manager/
│           └── 📄 naive.py
└── 📁 utils/
    └── 📁 dataset/
        └── 📄 rl_dataset.py
```

## Setup
Please configure the `.env` as follow:
```
MCP_CONFIG_PATH=EnvFactory/configs/mcp_server.json
LOGGING_LEVEL=WARN
```

Please install the following packages based on the VeRL environments:
```
pip install jsonlines
pip install fastmcp==3.1.0
```