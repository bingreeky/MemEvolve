#!/usr/bin/env python
# coding=utf-8

"""
Regression test for the MemEvolve auto-fix sync-ordering bug.

Bug summary
-----------
When a generated memory provider fails the fast simulation test, the auto-fix
repairs it inside an *isolated* copy of the codebase and the retry passes.
The fix then has to be copied back ("synced") to the real provider file.

In the released code the ordering is wrong:

* ``PhaseValidator._run_fast_validation`` deletes the isolated environment in
  its ``finally`` block as soon as the verdict is ``passed`` -- *before it
  returns*.
* ``PhaseValidator._validate_single_system`` only calls
  ``_sync_to_real_environment`` *after* ``_run_fast_validation`` returns, i.e.
  after the isolated copy (which holds the fix) has already been deleted.

The result: the sync silently fails ("Isolated environment not found, cannot
sync"), its ``False`` return value is ignored, the round reports the provider
as validated, and the deployed provider file on disk still contains the
original bug.

This test reproduces the failure deterministically and without any LLM calls:
it builds a tiny self-contained ``EvolveLab`` package whose deployed provider
raises ``NameError`` at ``initialize()`` time (a missing ``import time``), and
monkeypatches ``SWEAgentValidator.fix_memory_system`` to write the corrected
file into the isolated copy. After validation, the deployed provider file in
the real tree MUST contain the fix. On the buggy code it does not.
"""

import json
import sys
import textwrap
import types
from pathlib import Path

import pytest

from MemEvolve.phases.phase_validator import PhaseValidator
from MemEvolve.validators.swe_agent_validator import SWEAgentValidator


SYSTEM_NAME = "test_sync_memory"

# A deployed provider that raises at initialize() because `time` is never
# imported. This stands in for the kind of runtime error the auto-fix repairs.
BUGGY_PROVIDER = '''\
from EvolveLab.base_memory import BaseMemoryProvider
from EvolveLab.memory_types import MemoryType, MemoryResponse


class TestSyncProvider(BaseMemoryProvider):
    def __init__(self, config=None):
        super().__init__(MemoryType.TEST_SYNC, config)

    def initialize(self):
        self._ts = time.time()  # NameError: name 'time' is not defined
        return True

    def provide_memory(self, request):
        return MemoryResponse(
            memories=[], memory_type=MemoryType.TEST_SYNC, total_count=0
        )

    def take_in_memory(self, trajectory_data):
        return True, "stored"
'''

# The marker the fix introduces. Its presence in the *deployed* file is the
# user-facing invariant this test guards.
FIX_MARKER = "import time"


def _write_min_evolvelab(project_root: Path) -> None:
    """Create a minimal, self-contained EvolveLab package + storage dir."""
    evolvelab = project_root / "EvolveLab"
    providers = evolvelab / "providers"
    providers.mkdir(parents=True, exist_ok=True)
    (project_root / "storage").mkdir(parents=True, exist_ok=True)

    (evolvelab / "__init__.py").write_text("")
    (providers / "__init__.py").write_text("")

    (evolvelab / "memory_types.py").write_text(textwrap.dedent('''\
        from dataclasses import dataclass
        from enum import Enum
        from typing import Any, Dict, List, Optional


        class MemoryStatus(Enum):
            BEGIN = "begin"
            IN = "in"


        class MemoryType(Enum):
            TEST_SYNC = "test_sync_memory"


        PROVIDER_MAPPING = {
            MemoryType.TEST_SYNC: ("TestSyncProvider", "test_sync_memory_provider"),
        }


        class MemoryItemType(Enum):
            TEXT = "text"
            API = "api"


        @dataclass
        class MemoryRequest:
            query: str
            context: str
            status: MemoryStatus
            additional_params: Optional[Dict[str, Any]] = None


        @dataclass
        class MemoryItem:
            id: str
            content: Any
            metadata: Dict[str, Any]
            score: Optional[float] = None
            type: MemoryItemType = MemoryItemType.TEXT


        @dataclass
        class MemoryResponse:
            memories: List[MemoryItem]
            memory_type: MemoryType
            total_count: int
            request_id: Optional[str] = None


        @dataclass
        class TrajectoryData:
            query: str
            trajectory: List[Dict[str, Any]]
            result: Optional[Any] = None
            metadata: Optional[Dict[str, Any]] = None
    '''))

    (evolvelab / "base_memory.py").write_text(textwrap.dedent('''\
        from abc import ABC, abstractmethod
        from typing import Optional

        from .memory_types import MemoryType


        class BaseMemoryProvider(ABC):
            def __init__(self, memory_type: MemoryType, config: Optional[dict] = None):
                self.memory_type = memory_type
                self.config = config or {}

            @abstractmethod
            def provide_memory(self, request):
                ...

            @abstractmethod
            def take_in_memory(self, trajectory_data):
                ...

            @abstractmethod
            def initialize(self) -> bool:
                ...
    '''))

    (evolvelab / "config.py").write_text(textwrap.dedent('''\
        from .memory_types import MemoryType


        def get_memory_config(provider_type):
            return {}
    '''))

    (providers / f"{SYSTEM_NAME}_provider.py").write_text(BUGGY_PROVIDER)


@pytest.fixture
def validator_env(tmp_path, monkeypatch):
    """Build a temp project tree + wired-up PhaseValidator for the repro."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_min_evolvelab(project_root)

    # Config consumed by _validate_single_system: the stored `provider_code`
    # only needs to pass the static check; the deployed file is what matters.
    config = {
        "memory_type_info": {"enum_value": SYSTEM_NAME},
        "provider_code": {"code": "import time\n" + BUGGY_PROVIDER},
    }
    config_path = tmp_path / "configs.json"
    config_path.write_text(json.dumps([config]))

    # Fake FlashOAgents.OpenAIServerModel -> keeps the repro LLM-free.
    fake_flashoagents = types.ModuleType("FlashOAgents")

    class _FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_flashoagents.OpenAIServerModel = _FakeModel
    monkeypatch.setitem(sys.modules, "FlashOAgents", fake_flashoagents)

    # Auto-fix stand-in: write the corrected provider into the isolated copy,
    # exactly where the real SWE-agent fix would land, then report success.
    def fake_fix(self, memory_system_name, error_report):
        prov = (
            Path(self.isolated_env_dir)
            / "EvolveLab"
            / "providers"
            / f"{memory_system_name}_provider.py"
        )
        prov.write_text(FIX_MARKER + "\n" + prov.read_text())
        return True

    monkeypatch.setattr(SWEAgentValidator, "fix_memory_system", fake_fix)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    validator = PhaseValidator(
        work_dir=work_dir,
        configs_path=str(config_path),
        use_fast_validation=True,
        enable_auto_fix=True,
        max_fix_attempts=3,
        cleanup_temp=True,
    )
    # Point the validator at our self-contained temp tree.
    validator.project_root = project_root

    deployed_provider = (
        project_root / "EvolveLab" / "providers" / f"{SYSTEM_NAME}_provider.py"
    )
    return validator, deployed_provider


def test_autofix_fix_is_synced_to_deployed_provider(validator_env):
    """A provider repaired by auto-fix must be synced back to the real tree.

    Fails on the buggy code: the isolated environment (holding the fix) is
    deleted before _sync_to_real_environment runs, so the deployed file is
    never updated.
    """
    validator, deployed_provider = validator_env

    # Sanity: the deployed provider starts broken (no `import time`).
    assert FIX_MARKER not in deployed_provider.read_text()

    result = validator.run_validation([SYSTEM_NAME])

    # The fix must have reached the deployed provider file on disk.
    assert FIX_MARKER in deployed_provider.read_text(), (
        "Auto-fix repaired the provider but the fix was never synced to the "
        "deployed file; the real tree still holds the broken provider."
    )

    # And the system must not be reported as validated unless the sync worked.
    details = result["details"][SYSTEM_NAME]
    assert details["verdict"] == "passed", (
        f"Expected verdict 'passed' with a real sync, got {details['verdict']!r}"
    )
    assert SYSTEM_NAME in result["validated"]
