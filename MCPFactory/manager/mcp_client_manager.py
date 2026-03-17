import asyncio
import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.utilities.logging import get_logger, configure_logging

to_client_logger = get_logger(name="fastmcp.server.context.to_client")
configure_logging(
    level=os.getenv("VERL_LOGGING_LEVEL", "WARNING"),
    logger=to_client_logger
)

load_dotenv()

class MCPClientManager:
    _instance = None
    _cls_lock = threading.Lock()
    _excluded_tools = frozenset({"load_scenario", "save_scenario"})

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_bootstrapped", False):
            return

        self._bootstrapped = True
        self._initialized = False
        self._lock = threading.Lock()
        self._register_lock = asyncio.Lock()
        self._conn_lock = asyncio.Lock()

        self.clients = {}
        self.stateless_clients = {}
        self.server_to_path_mapping = {}
        self.tools = {}

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.loop_thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    @property
    def tool_schemas(self) -> list:
        with self._lock:
            return list(self.tools.values())

    def init_config(self, config_path, overwrite=False):
        if self._initialized and not overwrite:
            return

        config_path = Path(config_path).resolve()
        with open(config_path, "r", encoding="utf-8") as file_obj:
            config = json.load(file_obj)

        future = asyncio.run_coroutine_threadsafe(
            self._init_config_async(config=config, overwrite=overwrite),
            self.loop,
        )
        return future.result(timeout=120)

    async def _init_config_async(self, config, overwrite=False):
        if self._initialized and not overwrite:
            return

        if overwrite:
            await self._close_all_clients_async()
            with self._lock:
                self.server_to_path_mapping = {}
                self.tools = {}

        tasks = [
            self.register_mcp_server_async(
                server_name=server_name,
                tool_path=server_config["tool_path"],
                is_stateless=server_config.get("stateless", False),
            )
            for server_name, server_config in config.get("mcpServers", {}).items()
        ]
        await asyncio.gather(*tasks)

        self._initialized = True

    async def register_mcp_server_async(self, server_name, tool_path, is_stateless=False):
        async def get_tool_schemas(client):
            schemas = []
            tools = await client.list_tools()
            for tool in tools:
                schemas.append((
                    f"{server_name}-{tool.name}",
                    {
                        "type": "function",
                        "function": {
                            "name": f"{server_name}-{tool.name}",
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    },
                ))
            return schemas

        client = Client(tool_path)
        if is_stateless:
            await client._connect()
            tool_schemas = await get_tool_schemas(client)
        else:
            async with client:
                tool_schemas = await get_tool_schemas(client)
            await client.close()

        async with self._register_lock:
            with self._lock:
                self.server_to_path_mapping[server_name] = tool_path
                for tool_name, tool_schema in tool_schemas:
                    self.tools[tool_name] = tool_schema
                if is_stateless:
                    self.stateless_clients[server_name] = client

    def filter_tools(self, servers):
        with self._lock:
            if servers is None:
                return list(self.tools.values())

            allowed_servers = frozenset(servers)
            filtered_tools = []
            for tool_name, tool_schema in self.tools.items():
                server_name, _, short_tool_name = tool_name.partition("-")
                if server_name in allowed_servers and short_tool_name not in self._excluded_tools:
                    filtered_tools.append(tool_schema)
            return filtered_tools

    @staticmethod
    def is_valid_client_id(client_id):
        if not isinstance(client_id, str):
            return False
        hyphen_index = client_id.find("-")
        return 0 < hyphen_index < len(client_id) - 1

    def get_client(self, client_id):
        assert self.is_valid_client_id(client_id), "client_id must use '<server>-<request>' format."

        with self._lock:
            if client_id in self.clients:
                client_info = self.clients[client_id]
                return client_info["client"], client_info["status"]

            server_name = client_id.split("-", 1)[0]
            if server_name in self.stateless_clients:
                return self.stateless_clients[server_name], True

            tool_path = self.server_to_path_mapping[server_name]
            client = Client(tool_path)
            self.clients[client_id] = {
                "client": client,
                "status": False,
                "connected": False,
            }
            return client, False

    def set_status(self, client_id):
        with self._lock:
            if client_id in self.clients:
                self.clients[client_id]["status"] = True

    def load_scenario(self, client_id, scenario=None, check=False):
        client, status = self.get_client(client_id)
        if status or scenario is None:
            return "This client is already initialized. Skipping..."

        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async("load_scenario", {"scenario": scenario}, client, client_id),
            self.loop,
        )
        result = future.result(timeout=30)
        if check:
            saved_scenario = self.call_tool(client_id=client_id, tool_name="save_scenario", tool_args={})
            try:
                if json.loads(saved_scenario) == json.loads(json.dumps(scenario)):
                    self.set_status(client_id)
            except Exception:
                pass
        else:
            self.set_status(client_id)
        return result

    def call_tool(self, tool_name, tool_args, client_id):
        assert self.is_valid_client_id(client_id), "client_id must use '<server>-<request>' format."

        if "load_scenario" in tool_name:
            scenario = tool_args.get("scenario", tool_args) if isinstance(tool_args, dict) else json.loads(tool_args)
            return self.load_scenario(client_id=client_id, scenario=scenario)

        client, _ = self.get_client(client_id)
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(tool_name, tool_args, client, client_id),
            self.loop,
        )
        try:
            return future.result(timeout=30)
        except TimeoutError:
            return f"{tool_name} timed out after 30 seconds"
        except ToolError as exc:
            return f"{tool_name} fail before execution: {exc}"
        except Exception as exc:
            return f"{tool_name} raise an unexpected error: {exc}"

    async def _call_tool_async(self, tool_name, tool_args, client, client_id):
        short_tool_name = tool_name.split("-", 1)[-1]
        async with self._conn_lock:
            with self._lock:
                client_info = self.clients.get(client_id)
                need_connect = client_info is not None and not client_info["connected"]

            if need_connect:
                await client._connect()
                with self._lock:
                    if client_id in self.clients:
                        self.clients[client_id]["connected"] = True

        if isinstance(tool_args, str):
            tool_args = json.loads(tool_args)

        result = await client.call_tool(short_tool_name, tool_args)
        texts = [item.text for item in result.content if hasattr(item, "text")]
        return ",".join(texts) if texts else ""

    def save_all_scenario(self, client_id_list):
        future = asyncio.run_coroutine_threadsafe(
            self._save_all_scenario_async(client_id_list),
            self.loop,
        )
        return future.result(timeout=60)

    async def _save_all_scenario_async(self, client_id_list):
        async def save_one(client_id):
            server_name = client_id.split("-", 1)[0]
            try:
                client, _ = self.get_client(client_id)
                raw_result = await self._call_tool_async("save_scenario", {}, client, client_id)
                return server_name, json.loads(raw_result)
            except Exception:
                return server_name, None

        results = await asyncio.gather(*[save_one(client_id) for client_id in client_id_list])
        return dict(results)

    def close_client(self, client_id=None, server_name=None):
        future = asyncio.run_coroutine_threadsafe(
            self.close_client_async(client_id=client_id, server_name=server_name),
            self.loop,
        )
        return future.result(timeout=10)

    async def close_client_async(self, client_id=None, server_name=None):
        if client_id is not None:
            with self._lock:
                client_info = self.clients.pop(client_id, None)
            client = None if client_info is None else client_info["client"]
            if client is not None:
                await client.close()

        if server_name is not None:
            with self._lock:
                client = self.stateless_clients.pop(server_name, None)
            if client is not None:
                await client.close()

    async def _close_all_clients_async(self):
        with self._lock:
            client_ids = list(self.clients.keys())
            server_names = list(self.stateless_clients.keys())

        await asyncio.gather(
            *[self.close_client_async(client_id=client_id) for client_id in client_ids],
            *[self.close_client_async(server_name=server_name) for server_name in server_names],
            return_exceptions=True,
        )

    def close_all_clients(self):
        future = asyncio.run_coroutine_threadsafe(
            self._close_all_clients_async(),
            self.loop,
        )
        return future.result(timeout=30)

    def shutdown(self, timeout=5):
        if not self.loop.is_running():
            return
        self.close_all_clients()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=timeout)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


MCPManager = MCPClientManager()
mcp_config_path = os.environ.get("MCP_CONFIG_PATH")
if mcp_config_path:
    MCPManager.init_config(mcp_config_path)