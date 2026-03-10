import asyncio
import json
import logging
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.exceptions import ToolError
from tqdm import tqdm

import jsonlines

logger = logging.getLogger()
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

load_dotenv()

class MCPClientManager:
    _instance = None
    _cls_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = super(MCPClientManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self.clients = {}  # {client_id: {"client": client, "status": bool, "connected": bool}}
            self.stateless_clients = {}  # {server_name: client}
            self.server_to_path_mapping = {}  # {server_name: tool_path}
            self.log_info = {}
            self.tools = {}  # {tool_name: tool_schema}; tool_schemas derived as property
            self._lock = threading.Lock()

            self.loop = asyncio.new_event_loop()
            self._register_lock = asyncio.Lock()  # Create in main thread to avoid race before loop starts
            self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
            self.loop_thread.start()

            self._initialized = False

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    @property
    def tool_schemas(self) -> list:
        """List of tool schemas (derived from self.tools for compatibility)."""
        with self._lock:
            return list(self.tools.values())

    # TODO
    def is_valid_mcp_servers(self, config: dict) -> bool:
        return True

    @staticmethod
    def is_valid_client_id(client_id: str) -> bool:
        """A valid client_id must follow "{mcp_server}-{request_id}" where each part is non-empty.
        
        Supported patterns:
        - {mcp_server}-{request_id}  (e.g., "AdobePDFServices-abc123")
        - {mcp_server}-{request_id}_test_{test_case_id}  (e.g., "AdobePDFServices-abc123_test_001")
        - {mcp_server}-{request_id}_{scenario_id}  (e.g., "GoogleMaps-abc123_scenario_001")
        """
        if not isinstance(client_id, str):
            return False
        idx = client_id.find("-")
        # Must have a hyphen, server name before it, and non-empty request_id after it
        return 0 < idx < len(client_id) - 1

    async def init_config_async(self, config: dict, overwrite: bool = False):
        """
        Initialize the MCP client manager from JSON config file.

        Args:
            config_path: Path to the MCP server config file
            overwrite: Whether to overwrite existing configuration
        """
        if getattr(self, "_initialized", False) and not overwrite:
            return

        with self._lock:
            self.tools = {}  # {tool_name: tool_schema}

        mcp_servers = config["mcpServers"]
        with self._lock:
            for server_name, server in mcp_servers.items():
                self.server_to_path_mapping[server_name] = server["tool_path"]

        # Register servers with limited concurrency (avoids blocking event loop when many parallel)
        server_items = list(mcp_servers.items())
        if server_items:
            sem = asyncio.Semaphore(10)  # Limit to 10 concurrent registrations

            async def register_with_sem(server_name: str, server: dict):
                async with sem:
                    return await self.register_MCP_server_async(
                        server_name=server_name,
                        tool_path=server["tool_path"],
                        is_stateless=server.get("stateless", False),
                    )

            tasks = [register_with_sem(sn, s) for sn, s in server_items]
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Initializing MCP servers"):
                await f

        self._initialized = True

    def init_config(self, config_path: str, overwrite: bool = False):
        """Synchronous wrapper for init_config_async.

        Args:
            config_path: Path to the MCP server config file
            overwrite: Whether to overwrite existing configuration
        """
        if getattr(self, "_initialized", False) and not overwrite:
            return

        config_path_obj = Path(config_path).resolve()
        try:
            with open(config_path_obj, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    config = {"mcpServers": {}}
                else:
                    config = json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError):
            config = {"mcpServers": {}}
            config_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path_obj, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)

        if not self.is_valid_mcp_servers(config):
            raise ValueError("MCP servers config is not valid")

        # Resolve tool_path relative to config file's project root (parent of configs/)
        base_dir = config_path_obj.parent.parent
        for server_name, server in config["mcpServers"].items():
            raw_path = server["tool_path"]
            if not Path(raw_path).is_absolute():
                resolved = (base_dir / raw_path).resolve()
                server["tool_path"] = str(resolved)

        future = asyncio.run_coroutine_threadsafe(
            self.init_config_async(config, overwrite), self.loop
        )
        return future.result()

    async def register_MCP_server_async(self, server_name: str, tool_path: str, is_stateless: bool = False) -> None:
        """
        Register a new MCP server into the manager.

        Args:
            server_name: Name of the MCP server
            tool_path: Path to the tool file
            is_stateless: Whether the server is stateless
        """
        async def add_tool_schema(server_name: str, client: Client):
            tools = await client.list_tools()
            new_schemas = []
            for tool in tools:
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": f"{server_name}-{tool.name}",
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                tool_name = f"{server_name}-{tool.name}"
                new_schemas.append((tool_name, tool_schema))
            return new_schemas

        client = Client(tool_path)
        if is_stateless:
            await client._connect()
            new_schemas = await add_tool_schema(server_name, client)
        else:
            async with client:
                new_schemas = await add_tool_schema(server_name, client)
            await client.close()

        async with self._register_lock:
            with self._lock:
                self.server_to_path_mapping[server_name] = tool_path
                for tool_name, tool_schema in new_schemas:
                    if tool_name not in self.tools:
                        self.tools[tool_name] = tool_schema
                if is_stateless:
                    self.stateless_clients[server_name] = client

    def register_MCP_server(self, server_name: str, tool_path: str, is_stateless: bool = False) -> None:
        """Synchronous wrapper for register_MCP_server_async"""
        future = asyncio.run_coroutine_threadsafe(
            self.register_MCP_server_async(server_name, tool_path, is_stateless),
            self.loop,
        )
        try:
            future.result()
        except Exception as e:
            raise e

    async def reload_MCP_server_async(self, server_name: str, tool_path: str, is_stateless: bool = False) -> None:
        """
        Reload an existing MCP server from the manager.
        
        Args:
            server_name: Name of the MCP server
            tool_path: Path to the updated tool file
            is_stateless: Whether the server is stateless
        """
        # 1. Close all related stateful clients
        with self._lock:
            clients_to_close = [
                cid for cid in list(self.clients.keys())
                if cid.startswith(f"{server_name}-")
            ]
        for client_id in clients_to_close:
            await self.close_client_async(client_id)

        # 2. Close all related stateless clients
        if server_name in self.stateless_clients:
            try:
                await self.stateless_clients[server_name].close()
            except Exception as e:
                logger.warning(f"Failed to close stateless client: {e}")
            del self.stateless_clients[server_name]

        # 3. Remove original tool schemas
        prefix = f"{server_name}-"
        with self._lock:
            self.tools = {
                name: schema for name, schema in self.tools.items()
                if not name.startswith(prefix)
            }

        # 4. Re-register the tool server
        await self.register_MCP_server_async(server_name, tool_path, is_stateless)
        logger.info(f"Successfully reloaded tool server: {server_name}")

    def reload_MCP_server(self, server_name: str, tool_path: str, is_stateless: bool = False) -> None:
        """Synchronous wrapper for reload_MCP_server_async"""
        future = asyncio.run_coroutine_threadsafe(
            self.reload_MCP_server_async(server_name, tool_path, is_stateless),
            self.loop
        )
        try:
            future.result()
        except Exception as e:
            logger.error(f"Error reloading MCP server: {e}")
            raise e

    _EXCLUDED_TOOLS = frozenset({"load_scenario", "save_scenario"})

    def filter_tools(self, servers: list[str] | None) -> list[dict]:
        with self._lock:
            if servers is None:
                return list(self.tools.values())
            server_set = frozenset(servers)
            result = []
            for name, schema in self.tools.items():
                server, _, tool = name.partition("-")
                if server in server_set and tool not in self._EXCLUDED_TOOLS:
                    result.append(schema)
            return result

    def dump_log(self, client_id, log_dump_path="log/log.jsonl") -> None:
        os.makedirs(os.path.dirname(log_dump_path), exist_ok=True)
        with self._lock:
            log_entry = self.log_info.pop(client_id, None)
        if log_entry is not None:
            with jsonlines.open(log_dump_path, mode="a") as writer:
                writer.write({client_id: log_entry})

    def add_log(self, client_id, info: dict) -> None:
        if not info:
            return
        no_class_client_id = client_id.split("-", 1)[-1]
        with self._lock:
            self.log_info.setdefault(no_class_client_id, []).append(info)

    def set_status(self, client_id):
        with self._lock:
            client_info = self.clients.get(client_id)
            if client_info:
                client_info["status"] = True

    def get_client(self, client_id: str) -> tuple[Client, bool]:
        """
        Get a client from a given client id or create a new client if not exist.
        """
        assert self.is_valid_client_id(
            client_id
        ), "The client_id should follow pattern {mcp_server}-{request_id} or {mcp_server}-{request_id}_test_{test_case_id}!"

        with self._lock:
            if client_id in self.clients:
                info = self.clients[client_id]
                return info["client"], info["status"]
            tool_class = client_id.split("-", 1)[0]
            if tool_class in self.stateless_clients:
                return self.stateless_clients[tool_class], True
            tool_path = self.server_to_path_mapping.get(tool_class)
            if tool_path is None:
                raise KeyError(f"Unknown server: {tool_class}")
            new_client = Client(tool_path)
            self.clients[client_id] = {
                "client": new_client,
                "status": False,
                "connected": False,
            }
        return new_client, False

    async def close_client_async(self, client_id: str = None, server_name: str = None):
        """
        Close and remove a client from the manager.

        Args:
            client_id: Client ID of the stateful client to close
            server_name: Server name of the stateless client to close
        """
        if client_id:
            with self._lock:
                client_info = self.clients.pop(client_id, {})
            client_to_close = client_info.get("client")
        if client_to_close:
            try:
                await client_to_close.close()
                logger.info(f"Client {client_id} closed and removed")
            except Exception as e:
                logger.error(f"Error closing client {client_id}: {e}")

        if server_name:
            with self._lock:
                client_to_close = self.stateless_clients.pop(server_name, None)
        if client_to_close:
            try:
                await client_to_close.close()
                logger.info(f"Server {server_name} closed and removed")
            except Exception as e:
                logger.error(f"Error closing server {server_name}: {e}")

    def close_client(self, client_id: str = None, server_name: str = None):
        """Synchronous wrapper for the async close_client method"""
        future = asyncio.run_coroutine_threadsafe(
            self.close_client_async(client_id, server_name), self.loop
        )
        try:
            future.result()
        except Exception as e:
            logger.error(f"Error closing clients: {e}")

    async def _close_all_clients_async(self) -> None:
        """Close all clients in parallel on the event loop."""
        with self._lock:
            client_ids = list(self.clients.keys())
            server_names = list(self.stateless_clients.keys())
        results = await asyncio.gather(
            *[self.close_client_async(client_id=cid) for cid in client_ids],
            *[self.close_client_async(server_name=sn) for sn in server_names],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Error closing client: {r}")

    def close_all_clients(self):
        """Close all clients."""
        future = asyncio.run_coroutine_threadsafe(
            self._close_all_clients_async(), self.loop
        )
        try:
            future.result()
        except Exception as e:
            logger.error(f"Error closing clients: {e}")

    def load_scenario(
        self, client_id: str, scenario: dict | None = None, check: bool = False
    ):
        """Synchronous wrapper for the async call_tool method"""
        client, status = self.get_client(client_id)
        if not status and scenario:
            tool_args = {"scenario": scenario}
            future = asyncio.run_coroutine_threadsafe(
                self._call_tool_async("load_scenario", tool_args, client, client_id),
                self.loop,
            )
            try:
                result = future.result()
                if check:
                    saved_scenario = self.call_tool(
                        client_id=client_id,
                        tool_name="save_scenario",
                        tool_args={},
                    )
                    try:
                        scenario_normalized = json.loads(json.dumps(scenario))
                        if scenario_normalized == json.loads(saved_scenario):
                            self.set_status(client_id)
                    except Exception:
                        pass
                else:
                    self.set_status(client_id)
                return result
            except Exception as e:
                raise e
        return "This client is already initialized. Skipping..."

    def call_tool(self, tool_name: str, tool_args: dict | str, client_id: str):
        """Synchronous wrapper for the async call_tool method"""
        assert self.is_valid_client_id(
            client_id
        ), "The client_id should follow pattern {mcp_server}-{request_id} or {mcp_server}-{request_id}_test_{test_case_id}!"

        if "load_scenario" in tool_name:
            # Parse tool_args if it's a JSON string (from execute_mcp_tool wrapper)
            if isinstance(tool_args, str):
                try:
                    parsed_args = json.loads(tool_args)
                    if isinstance(parsed_args, dict) and "scenario" in parsed_args:
                        scenario = parsed_args["scenario"]
                    elif isinstance(parsed_args, dict):
                        scenario = parsed_args
                    else:
                        scenario = parsed_args
                except json.JSONDecodeError as e:
                    return f"{tool_name} fail: Invalid JSON in tool_args - {str(e)}"
            elif isinstance(tool_args, dict):
                scenario = tool_args.get("scenario", tool_args)

            return self.load_scenario(client_id=client_id, scenario=scenario)

        client, status = self.get_client(client_id)
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(tool_name, tool_args, client, client_id), self.loop
        )
        try:
            result = future.result()
            logger.debug(f"{tool_name} execute: {result}")
            return result
        except ToolError as e:
            logger.error(f"{tool_name} fail before execution: {e}")
            return f"{tool_name} fail before execution: {e}"
        except Exception as e:
            logger.error(f"{tool_name} raise an unexpected error: {e}")
            return f"{tool_name} raise an unexpected error: {e}"

    async def _call_tool_async(
        self, tool_name: str, tool_args: dict | str, client: Client, client_id: str
    ) -> str:
        assert self.is_valid_client_id(
            client_id
        ), "The client_id should follow pattern {mcp_server}-{request_id} or {mcp_server}-{request_id}_test_{test_case_id}!"

        tool_name = tool_name.split("-", 1)[-1]
        with self._lock:
            client_info = self.clients.get(client_id)
            need_connect = client_info is not None and not client_info.get("connected", False)
            if need_connect:
                client_info["connected"] = True
        if need_connect:
            await client._connect()

        if isinstance(tool_args, str):
            tool_args = json.loads(tool_args)
        
        result = await client.call_tool(tool_name, tool_args)

        all_texts = [item.text for item in result.content if hasattr(item, 'text')]
        tool_result = ','.join(all_texts) if all_texts else ''

        self.add_log(
            client_id=client_id,
            info={
                "tool": {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_result": tool_result,
                }
            },
        )

        return tool_result

    async def _save_all_scenario_async(self, client_id_list) -> dict:
        async def _save_one(client_id: str):
            tool_class = client_id.split("-", 1)[0]
            try:
                client, _ = self.get_client(client_id)
                raw = await self._call_tool_async("save_scenario", {}, client, client_id)
                return tool_class, json.loads(raw)
            except Exception:
                return tool_class, None

        results = await asyncio.gather(*[_save_one(cid) for cid in client_id_list])
        return dict(results)

    def save_all_scenario(self, client_id_list) -> dict:
        future = asyncio.run_coroutine_threadsafe(
            self._save_all_scenario_async(client_id_list), self.loop
        )
        return future.result()

    def shutdown(self, timeout=5):
        """Cleanup and shutdown the manager."""
        if not hasattr(self, "loop") or not self.loop.is_running():
            return
        self.close_all_clients()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=timeout)
        if self.loop_thread.is_alive():
            logger.warning(f"Event loop thread did not terminate within {timeout} seconds")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


MCPManager = MCPClientManager()
mcp_config_path = os.environ.get("MCP_CONFIG_PATH")
if mcp_config_path:
    MCPManager.init_config(mcp_config_path)