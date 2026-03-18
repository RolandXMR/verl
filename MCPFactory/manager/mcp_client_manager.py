import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.exceptions import ToolError

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

load_dotenv()

# Global Ray actor handle for distributed MCP access
_mcp_actor = None
_actor_initialized = False


class MCPClientManager:
    """Manages multiple concurrent MCP servers with connection pooling for performance.

    Uses transport connection pooling via `client.new()` to reuse persistent connections
    across isolated sessions, eliminating ~100-500ms connection overhead per operation.
    """

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
        
        # Log initialization for debugging singleton behavior
        import os
        pid = os.getpid()
        ray_initialized = ray.is_initialized() if RAY_AVAILABLE else False
        print(f"[MCPClientManager] Initializing in PID {pid}, "
              f"RAY_AVAILABLE={RAY_AVAILABLE}, ray_initialized={ray_initialized}")

        self._lock = threading.Lock()
        self._register_lock = asyncio.Lock()
        self._stateless_lock = asyncio.Lock()
        self._base_client_lock = asyncio.Lock()

        # Session clients: tracked but auto-clean via async-with exit
        self.clients: Dict[str, dict] = {}
        # Stateless clients: reused across all requests
        self.stateless_clients: Dict[str, Client] = {}
        # Base clients: persistent transport for session cloning
        self._base_clients: Dict[str, Client] = {}
        self.server_to_path_mapping: Dict[str, str] = {}
        self.tools: Dict[str, dict] = {}

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _ensure_base_client(self, server_name: str) -> Client:
        """Get or create a connected base client for transport reuse."""
        async with self._base_client_lock:
            if server_name in self._base_clients:
                return self._base_clients[server_name]

            client = Client(self.server_to_path_mapping[server_name])
            await client._connect()
            self._base_clients[server_name] = client
            return client

    async def _spawn_session(self, server_name: str) -> Client:
        """Create a new session client sharing the base transport."""
        base = await self._ensure_base_client(server_name)
        return base.new()

    def init_config(self, config_path, overwrite=False):
        """Initialize manager from MCP config file."""
        if self._initialized and not overwrite:
            return

        with open(Path(config_path).resolve(), "r", encoding="utf-8") as f:
            config = json.load(f)

        future = asyncio.run_coroutine_threadsafe(
            self._init_config_async(config, overwrite),
            self._loop,
        )
        return future.result(timeout=120)

    async def _init_config_async(self, config: dict, overwrite: bool = False):
        """Async initialization of all configured servers."""
        if self._initialized and not overwrite:
            return

        if overwrite:
            await self._close_all_clients_async()
            await self._close_all_base_clients_async()
            with self._lock:
                self.server_to_path_mapping.clear()
                self.tools.clear()

        tasks = [
            self.register_mcp_server_async(
                server_name, server_config["tool_path"],
                server_config.get("stateless", False)
            )
            for server_name, server_config in config.get("mcpServers", {}).items()
        ]
        await asyncio.gather(*tasks)
        self._initialized = True

    async def register_mcp_server_async(self, server_name: str, tool_path: str, is_stateless: bool = False):
        """Register an MCP server and extract its tool schemas."""
        client = Client(tool_path)
        await client._connect()

        tools = await client.list_tools()
        schemas = [
            (f"{server_name}-{tool.name}", {
                "type": "function",
                "function": {
                    "name": f"{server_name}-{tool.name}",
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            })
            for tool in tools
        ]

        async with self._register_lock:
            with self._lock:
                self.server_to_path_mapping[server_name] = tool_path
                for tool_name, schema in schemas:
                    self.tools[tool_name] = schema

                if is_stateless:
                    self.stateless_clients[server_name] = client
                else:
                    async with self._base_client_lock:
                        self._base_clients[server_name] = client

    def filter_tools(self, servers: Optional[List[str]] = None) -> List[dict]:
        """Filter tools by allowed server names."""
        with self._lock:
            if servers is None:
                return list(self.tools.values())

            allowed = frozenset(servers)
            result = []
            for name, schema in self.tools.items():
                server, _, short = name.partition("-")
                if server in allowed and short not in self._excluded_tools:
                    result.append(schema)
            return result

    @staticmethod
    def is_valid_client_id(client_id) -> bool:
        """Check if client_id uses '<server>-<request>' format."""
        if not isinstance(client_id, str):
            return False
        hyphen_idx = client_id.find("-")
        return 0 < hyphen_idx < len(client_id) - 1

    def get_client(self, client_id: str) -> Tuple[Client, bool]:
        """Get or create a client. Returns (client, is_initialized)."""
        assert self.is_valid_client_id(client_id), "client_id must use '<server>-<request>' format"
        server_name = client_id.split("-", 1)[0]

        with self._lock:
            if client_id in self.clients:
                info = self.clients[client_id]
                return info["client"], info["status"]

            if server_name in self.stateless_clients:
                return self.stateless_clients[server_name], True

        client = asyncio.run_coroutine_threadsafe(
            self._spawn_session(server_name), self._loop
        ).result(timeout=30)

        with self._lock:
            self.clients[client_id] = {"client": client, "status": False}
        return client, False

    def set_status(self, client_id: str):
        """Mark a session client as initialized."""
        with self._lock:
            if client_id in self.clients:
                self.clients[client_id]["status"] = True

    def load_scenario(self, client_id: str, scenario: Optional[dict] = None, check: bool = False):
        """Load a scenario into the client. Auto-initializes if successful."""
        client, initialized = self.get_client(client_id)
        if initialized or scenario is None:
            return "Client already initialized. Skipping..."

        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async("load_scenario", {"scenario": scenario}, client, client_id),
            self._loop,
        )
        result = future.result(timeout=30)

        if check:
            saved = self.call_tool(client_id, "save_scenario", {})
            try:
                if json.loads(saved) == scenario:
                    self.set_status(client_id)
            except Exception:
                pass
        else:
            self.set_status(client_id)

        return result

    def call_tool(self, client_id: str, tool_name: str, tool_args):
        """Execute a tool on the specified client."""
        assert self.is_valid_client_id(client_id), "client_id must use '<server>-<request>' format"

        if "load_scenario" in tool_name:
            scenario = tool_args.get("scenario", tool_args) if isinstance(tool_args, dict) else json.loads(tool_args)
            return self.load_scenario(client_id, scenario)

        client, _ = self.get_client(client_id)
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(tool_name, tool_args, client, client_id),
            self._loop,
        )
        try:
            return future.result(timeout=30)
        except TimeoutError:
            return f"{tool_name} timed out after 30 seconds"
        except ToolError as exc:
            return f"{tool_name} failed: {exc}"
        except Exception as exc:
            return f"{tool_name} error: {exc}"

    async def _call_tool_async(self, tool_name: str, tool_args, client: Client, client_id: str) -> str:
        """Execute tool with proper session handling."""
        short_name = tool_name.split("-", 1)[-1]
        args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args

        server_name = client_id.split("-", 1)[0]
        is_stateless = server_name in self.stateless_clients

        if is_stateless:
            async with self._stateless_lock:
                result = await client.call_tool(short_name, args)
        else:
            async with client:
                result = await client.call_tool(short_name, args)

        return ",".join(item.text for item in result.content if hasattr(item, "text"))

    def save_all_scenario(self, client_id_list: List[str]) -> Dict[str, Optional[dict]]:
        """Save scenarios from all specified clients."""
        future = asyncio.run_coroutine_threadsafe(
            self._save_all_scenario_async(client_id_list),
            self._loop,
        )
        return future.result(timeout=60)

    async def _save_all_scenario_async(self, client_id_list: List[str]) -> Dict[str, Optional[dict]]:
        """Async batch scenario save."""
        async def save_one(cid: str) -> Tuple[str, Optional[dict]]:
            server = cid.split("-", 1)[0]
            try:
                client, _ = self.get_client(cid)
                result = await self._call_tool_async("save_scenario", {}, client, cid)
                return server, json.loads(result)
            except Exception:
                return server, None

        results = await asyncio.gather(*[save_one(cid) for cid in client_id_list])
        return dict(results)

    def close_client(self, client_id: Optional[str] = None, server_name: Optional[str] = None):
        """Remove a session from tracking. Actual cleanup is automatic."""
        if client_id:
            with self._lock:
                self.clients.pop(client_id, None)

        if server_name:
            future = asyncio.run_coroutine_threadsafe(
                self._close_stateless_client(server_name),
                self._loop,
            )
            return future.result(timeout=10)

    async def _close_stateless_client(self, server_name: str):
        """Close a stateless client."""
        with self._lock:
            client = self.stateless_clients.pop(server_name, None)
        if client:
            await client.close()

    async def _close_all_clients_async(self):
        """Clear all session tracking."""
        with self._lock:
            self.clients.clear()

    async def _close_base_client(self, server_name: str):
        """Close a base client and remove from pool."""
        async with self._base_client_lock:
            client = self._base_clients.pop(server_name, None)
            if client:
                await client.close()

    async def _close_all_base_clients_async(self):
        """Close all base clients (for shutdown)."""
        async with self._base_client_lock:
            clients = list(self._base_clients.values())
            self._base_clients.clear()

        await asyncio.gather(*[c.close() for c in clients], return_exceptions=True)

    def close_all_clients(self):
        """Close all tracked clients."""
        future = asyncio.run_coroutine_threadsafe(
            self._close_all_clients_async(),
            self._loop,
        )
        return future.result(timeout=30)

    def shutdown(self, timeout: int = 5):
        """Gracefully shutdown all connections and the event loop."""
        if not self._loop.is_running():
            return

        asyncio.run_coroutine_threadsafe(
            self._close_all_base_clients_async(), self._loop
        ).result(timeout=timeout)

        self.close_all_clients()

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=timeout)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


# Global singleton instance - lazy initialization to avoid creating instances in every process
_MCPManager = None

def get_mcp_manager():
    """
    Lazy initialization of local MCPManager.
    
    This prevents creating unnecessary instances in every Ray worker process.
    Only creates an instance when explicitly called.
    """
    global _MCPManager
    if _MCPManager is None:
        _MCPManager = MCPClientManager()
        mcp_config_path = os.environ.get("MCP_CONFIG_PATH")
        if mcp_config_path:
            _MCPManager.init_config(mcp_config_path)
    return _MCPManager


# Backward compatibility alias - will be lazily evaluated
MCPManager = get_mcp_manager()


# =============================================================================
# Ray Distributed Support
# =============================================================================

if RAY_AVAILABLE:
    @ray.remote(num_cpus=1)
    class MCPManagerActor:
        """
        Ray Actor wrapper for MCPClientManager to enable distributed access.
        
        This actor runs as a singleton service that all Ray workers can call.
        Only one instance exists across all GPU processes.
        """
        
        def __init__(self, config_path: Optional[str] = None):
            """Initialize the MCPManager within the actor."""
            self._manager = MCPClientManager.__new__(MCPClientManager)
            # Reset bootstrap flag to allow initialization
            object.__setattr__(self._manager, '_bootstrapped', False)
            self._manager.__init__()
            
            if config_path:
                self._manager.init_config(config_path)
        
        def init_config(self, config_path: str, overwrite: bool = False) -> None:
            """Initialize MCP configuration."""
            self._manager.init_config(config_path, overwrite)
        
        def call_tool(self, client_id: str, tool_name: str, tool_args: Any) -> str:
            """Call a tool through the manager."""
            return self._manager.call_tool(client_id, tool_name, tool_args)
        
        def load_scenario(self, client_id: str, scenario: Optional[dict] = None, check: bool = False) -> str:
            """Load a scenario for a client."""
            return self._manager.load_scenario(client_id, scenario, check)
        
        def save_all_scenario(self, client_id_list: List[str]) -> Dict:
            """Save all scenarios for the given client IDs."""
            return self._manager.save_all_scenario(client_id_list)
        
        def close_client(self, client_id: Optional[str] = None, server_name: Optional[str] = None) -> Any:
            """Close a specific client or server."""
            return self._manager.close_client(client_id, server_name)
        
        def close_all_clients(self) -> Any:
            """Close all clients."""
            return self._manager.close_all_clients()
        
        def filter_tools(self, servers: Optional[List[str]] = None) -> List[dict]:
            """Filter tools by servers."""
            return self._manager.filter_tools(servers)
        
        @property
        def tools(self) -> Dict:
            """Get all tools."""
            return self._manager.tools
        
        @property
        def tool_schemas(self) -> List:
            """Get tool schemas."""
            return list(self._manager.tools.values())
        
        @property
        def server_to_path_mapping(self) -> Dict:
            """Get server to path mapping."""
            return self._manager.server_to_path_mapping
        
        @property
        def is_initialized(self) -> bool:
            """Check if the manager is initialized."""
            return getattr(self._manager, '_initialized', False)
        
        def shutdown(self, timeout: int = 5) -> None:
            """Shutdown the manager."""
            self._manager.shutdown(timeout)


def initialize_mcp_actor(config_path: Optional[str] = None, actor_name: str = "MCPManagerActor") -> Any:
    """
    Initialize the global MCPManager Ray actor.
    
    This should be called ONCE from the driver process (rank 0) before spawning workers.
    
    Args:
        config_path: Path to MCP configuration file
        actor_name: Name for the Ray actor (for retrieval)
    
    Returns:
        Ray actor handle
    
    Example:
        >>> # In main_ppo.py, before ray workers spawn
        >>> actor = initialize_mcp_actor("/path/to/mcp_config.json")
    """
    global _mcp_actor, _actor_initialized
    
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray is not available. Cannot create MCPManager actor.")
    
    if _actor_initialized and _mcp_actor is not None:
        return _mcp_actor
    
    # Check if we're in a Ray context
    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before creating MCPManager actor.")
    
    # Try to get existing actor first (in case it was created elsewhere)
    try:
        _mcp_actor = ray.get_actor(actor_name)
        _actor_initialized = True
        return _mcp_actor
    except ValueError:
        pass  # Actor doesn't exist yet, create it
    
    # Get config path from environment if not provided
    if config_path is None:
        config_path = os.environ.get("MCP_CONFIG_PATH")
    
    # Create the actor with a name for easy retrieval
    _mcp_actor = MCPManagerActor.options(
        name=actor_name,
        lifetime="detached"  # Keep alive even if creator dies
    ).remote(config_path)
    
    _actor_initialized = True
    return _mcp_actor


def get_mcp_actor(actor_name: str = "MCPManagerActor") -> Any:
    """
    Get the global MCPManager Ray actor.
    
    Can be called from any Ray worker to get the shared actor handle.
    
    Args:
        actor_name: Name of the Ray actor
    
    Returns:
        Ray actor handle
    
    Example:
        >>> # From any Ray worker
        >>> actor = get_mcp_actor()
        >>> result = ray.get(actor.call_tool.remote("client-1", "tool-name", args))
    """
    global _mcp_actor, _actor_initialized
    
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray is not available.")
    
    if _mcp_actor is not None:
        return _mcp_actor
    
    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before getting MCPManager actor.")
    
    # Try to get existing actor by name
    try:
        _mcp_actor = ray.get_actor(actor_name)
        _actor_initialized = True
        return _mcp_actor
    except ValueError as e:
        raise RuntimeError(
            f"MCPManager actor '{actor_name}' not found. "
            "Make sure to call initialize_mcp_actor() from the driver process first."
        ) from e


# Convenience functions for distributed access

def call_tool_distributed(client_id: str, tool_name: str, tool_args: Any, actor_name: str = "MCPManagerActor") -> str:
    """Call a tool through the distributed MCPManager actor."""
    actor = get_mcp_actor(actor_name)
    return ray.get(actor.call_tool.remote(client_id, tool_name, tool_args))


def load_scenario_distributed(client_id: str, scenario: Optional[dict] = None, check: bool = False, actor_name: str = "MCPManagerActor") -> str:
    """Load a scenario through the distributed MCPManager actor."""
    actor = get_mcp_actor(actor_name)
    return ray.get(actor.load_scenario.remote(client_id, scenario, check))


def save_all_scenario_distributed(client_id_list: List[str], actor_name: str = "MCPManagerActor") -> Dict:
    """Save all scenarios through the distributed MCPManager actor."""
    actor = get_mcp_actor(actor_name)
    return ray.get(actor.save_all_scenario.remote(client_id_list))


def close_client_distributed(client_id: Optional[str] = None, server_name: Optional[str] = None, actor_name: str = "MCPManagerActor") -> Any:
    """Close a client through the distributed MCPManager actor."""
    actor = get_mcp_actor(actor_name)
    return ray.get(actor.close_client.remote(client_id, server_name))


def get_tools_distributed(actor_name: str = "MCPManagerActor") -> Dict:
    """Get tools from the distributed MCPManager actor."""
    actor = get_mcp_actor(actor_name)
    return ray.get(actor.tools)


def get_tool_schemas_distributed(actor_name: str = "MCPManagerActor") -> List:
    """Get tool schemas from the distributed MCPManager actor."""
    actor = get_mcp_actor(actor_name)
    return ray.get(actor.tool_schemas)


def filter_tools_distributed(servers: Optional[List[str]] = None, actor_name: str = "MCPManagerActor") -> List:
    """Filter tools through the distributed MCPManager actor."""
    actor = get_mcp_actor(actor_name)
    return ray.get(actor.filter_tools.remote(servers))
