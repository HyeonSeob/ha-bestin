from __future__ import annotations

import ast
import asyncio as real_asyncio
from pathlib import Path
import socket
import time
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "custom_components" / "bestin"


class Logger:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class AsyncioProxy:
    CancelledError = real_asyncio.CancelledError
    TimeoutError = real_asyncio.TimeoutError
    Lock = real_asyncio.Lock
    Queue = real_asyncio.Queue
    Task = real_asyncio.Task
    gather = staticmethod(real_asyncio.gather)
    wait_for = staticmethod(real_asyncio.wait_for)

    def __init__(self):
        self.delays: list[float] = []

    async def sleep(self, delay):
        self.delays.append(delay)
        await real_asyncio.sleep(0)

    async def open_connection(self, *_args, **_kwargs):
        raise OSError("test connection failure")


def load_classes(path: Path, names: set[str], namespace: dict | None = None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [
        ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
        *(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names),
    ]
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    values = {
        "asyncio": real_asyncio,
        "re": __import__("re"),
        "socket": socket,
        "time": time,
        "serial_asyncio": types.SimpleNamespace(),
        "LOGGER": Logger(),
        "callback": lambda func: func,
    }
    if namespace:
        values.update(namespace)
    exec(compile(module, str(path), "exec"), values)
    return values


def method_node(path: Path, class_name: str, method_name: str) -> ast.AsyncFunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == method_name
    )


class FakeWriter:
    def __init__(self, closing: bool = False):
        self.closing = closing
        self.closed = False

    def is_closing(self):
        return self.closing or self.closed

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


class LiveReader:
    def at_eof(self):
        return False


class EofReader(LiveReader):
    async def read(self, _size):
        return b""


class ConnectionRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.asyncio = AsyncioProxy()
        values = load_classes(
            COMPONENT / "hub.py",
            {"ConnectionManager"},
            {"asyncio": self.asyncio},
        )
        self.ConnectionManager = values["ConnectionManager"]

    async def test_socket_eof_triggers_reconnect(self):
        connection = self.ConnectionManager("192.0.2.1:8899")
        connection.reader = EofReader()
        connection.writer = FakeWriter()
        calls = []

        async def reconnect(failed_writer=None):
            calls.append(failed_writer)
            return False

        connection.reconnect = reconnect
        self.assertIsNone(await connection.receive())
        self.assertEqual(calls, [connection.writer])

    async def test_closing_socket_writer_is_not_connected(self):
        connection = self.ConnectionManager("192.0.2.1:8899")
        connection.reader = LiveReader()
        connection.writer = FakeWriter(closing=True)
        self.assertFalse(connection.is_connected())

    async def test_reconnect_backoff_is_iterative_and_capped(self):
        connection = self.ConnectionManager("192.0.2.1:8899")

        async def fail_connect(*_args, **_kwargs):
            raise OSError("offline")

        connection.connect = fail_connect
        for _ in range(8):
            self.assertFalse(await connection.reconnect())

        self.assertEqual(self.asyncio.delays, [1, 2, 4, 8, 16, 32, 60, 60])

    async def test_concurrent_reconnect_uses_one_connection_attempt(self):
        connection = self.ConnectionManager("192.0.2.1:8899")
        failed_writer = FakeWriter()
        connection.reader = LiveReader()
        connection.writer = failed_writer
        attempts = 0

        async def connect(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            connection.reader = LiveReader()
            connection.writer = FakeWriter()
            return True

        connection.connect = connect
        results = await real_asyncio.gather(
            connection.reconnect(failed_writer),
            connection.reconnect(failed_writer),
        )
        self.assertEqual(results, [True, True])
        self.assertEqual(attempts, 1)


class ControllerRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.asyncio = AsyncioProxy()
        values = load_classes(
            COMPONENT / "controller.py",
            {"AsyncQueue", "BestinController"},
            {"asyncio": self.asyncio},
        )
        self.BestinController = values["BestinController"]

    async def test_empty_receive_yields_before_next_iteration(self):
        controller = object.__new__(self.BestinController)

        class Connection:
            def __init__(self):
                self.calls = 0

            def is_connected(self):
                return True

            async def receive(self):
                self.calls += 1
                if self.calls == 1:
                    return None
                raise real_asyncio.CancelledError

        controller.connection = Connection()
        controller.verify_checksum = lambda _data: True
        controller.log_packet_viewer = lambda *_args: None
        controller.handle_device_packet = lambda _data: None
        controller.queue = types.SimpleNamespace(size=lambda: None)

        with self.assertRaises(real_asyncio.CancelledError):
            await controller.process_incoming_data()
        self.assertIn(0.1, self.asyncio.delays)

    async def test_stop_waits_for_cancelled_tasks(self):
        controller = object.__new__(self.BestinController)
        task = real_asyncio.create_task(real_asyncio.sleep(60))
        controller.tasks = [task]
        await controller.stop()
        self.assertTrue(task.cancelled())
        self.assertEqual(controller.tasks, [])


class StartupAndUnloadSourceTests(unittest.TestCase):
    def test_startup_converts_connection_failures_to_retryable_error(self):
        tree = ast.parse((COMPONENT / "__init__.py").read_text(encoding="utf-8"))
        setup = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "async_setup_entry"
        )
        handled = {
            item.id
            for node in ast.walk(setup)
            if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Tuple)
            for item in node.type.elts
            if isinstance(item, ast.Name)
        }
        self.assertTrue({"OSError", "ConnectionError"}.issubset(handled))

    def test_close_does_not_require_connection_to_look_available(self):
        for method_name in ("async_close", "shutdown"):
            node = method_node(COMPONENT / "hub.py", "BestinHub", method_name)
            source = ast.unparse(node)
            self.assertNotIn("self.connection and self.available", source)
            self.assertIn("if self.connection", source)


if __name__ == "__main__":
    unittest.main()
