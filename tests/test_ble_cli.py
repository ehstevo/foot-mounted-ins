"""Verify BLE utility help paths without local configuration or hardware."""
import asyncio
import importlib.util
from pathlib import Path

import pytest


_SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    """Load one standalone script without executing its ``__main__`` block."""
    path = _SCRIPTS_DIRECTORY / f"{name}.py"
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script_name", ("ble_scan", "ble_inspect"))
def test_ble_help_exits_before_loading_local_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    script_name: str,
) -> None:
    """``--help`` prints usage without touching hardware-specific settings."""
    module = _load_script(script_name)

    def fail_if_loaded():
        pytest.fail("--help must not load local device configuration")

    monkeypatch.setattr(module, "_load_device_config", fail_if_loaded)

    with pytest.raises(SystemExit) as error:
        asyncio.run(module.main(["--help"]))

    assert error.value.code == 0
    assert "usage:" in capsys.readouterr().out
