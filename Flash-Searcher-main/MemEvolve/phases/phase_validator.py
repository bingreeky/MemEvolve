#!/usr/bin/env python
# coding=utf-8

"""
Phase 4: Validation phase for memory evolution
Validates generated memory systems through static checks and simulation testing
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from ..config import DEFAULT_DATASETS, IMPORT_TIMEOUT_SEC
from ..validators.swe_agent_validator import SWEAgentValidator

load_dotenv(override=True)


class PhaseValidator:
    """
    Handles memory system validation phase

    Performs static checks and runtime tests without automatic fixes
    """

    def __init__(
        self,
        work_dir: Path,
        configs_path: str,
        dataset_name: Optional[str] = None,
        datasets_config: Optional[Dict[str, str]] = None,
        use_fast_validation: bool = True,
        enable_auto_fix: bool = True,
        max_fix_attempts: int = 3,
        cleanup_temp: bool = True,
    ):
        self.work_dir = work_dir
        self.configs_path = configs_path
        self.dataset_name = dataset_name
        self.datasets_config = datasets_config or DEFAULT_DATASETS
        self.use_fast_validation = use_fast_validation
        self.enable_auto_fix = enable_auto_fix
        self.max_fix_attempts = max_fix_attempts
        self.cleanup_temp = cleanup_temp
        
        self.project_root = Path(__file__).parent.parent.parent
        self.isolated_env_dir = None
        self.temp_dir = self.work_dir / ".temp_validation"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def run_validation(self, created_systems: List[str]) -> Dict:
        """
        Validate created systems through static checks and tests

        Args:
            created_systems: List of system names to validate

        Returns:
            Validation result with passed/failed systems
        """
        print(f"[Validate] Starting validation for {len(created_systems)} systems")

        if not created_systems:
            print("[Validate] No systems to validate")
            return {"success": True, "validated": [], "failed": [], "details": {}}

        validated_systems = []
        failed_systems = []
        validation_details = {}  # Changed from list to dict

        for system_name in created_systems:
            print(f"\n[Validate] System: {system_name}")

            validation_result = self._validate_single_system(system_name)
            validation_details[system_name] = validation_result  # Store as dict with system_name as key

            if validation_result["verdict"] == "passed":
                validated_systems.append(system_name)
                print(f"  PASSED")
            else:
                failed_systems.append(system_name)
                print(f"  FAILED: {validation_result['verdict']}")

        print(f"\n[Validate] Complete")
        print(f"  Passed: {len(validated_systems)}")
        print(f"  Failed: {len(failed_systems)}")
        
        old_reports_dir = self.work_dir / "validation_reports"
        if old_reports_dir.exists():
            try:
                shutil.rmtree(old_reports_dir)
                print(f"  Cleaned up old validation_reports folder")
            except Exception:
                pass

        return {
            "success": True,
            "validated": validated_systems,
            "failed": failed_systems,
            "details": validation_details  # Now returns dict instead of list
        }

    def _validate_single_system(self, system_name: str) -> Dict:
        """
        Validate a single system: static check -> test -> report

        Args:
            system_name: System name to validate

        Returns:
            Validation report
        """
        report = {
            "system": system_name,
            "timestamp": datetime.now().isoformat(),
            "static_check": {},
            "test_results": {},
            "logs": [],
            "error_logs": [],
            "verdict": "unknown"
        }

        with open(self.configs_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Handle both single config object and list of configs
        if isinstance(config_data, list):
            all_configs = config_data
        else:
            # Single config object - wrap in list
            all_configs = [config_data]

        system_config = None
        for cfg in all_configs:
            if cfg["memory_type_info"]["enum_value"] == system_name:
                system_config = cfg
                break

        if not system_config:
            report["verdict"] = "config_not_found"
            report["logs"].append("Configuration not found in config file")
            return report

        print(f"  [Static Check]")
        static_result = self._static_check(system_config["provider_code"]["code"])
        report["static_check"] = static_result

        if not static_result["passed"]:
            report["verdict"] = "failed_static"
            report["logs"].append(f"Static check failed with {len(static_result['issues'])} issues")
            for issue in static_result["issues"]:
                print(f"    - [{issue['severity']}] {issue['description']}")
            return report

        print(f"    Static check passed")

        # Select test mode
        if self.use_fast_validation:
            print(f"  [Simulation Test]")

            try:
                for attempt in range(self.max_fix_attempts):
                    if attempt > 0:
                        print(f"  [Simulation Test] Retry attempt {attempt + 1}")

                    # Only setup isolated environment on first attempt
                    # Subsequent attempts reuse the environment with fixes applied
                    setup_env = (attempt == 0)
                    fast_result = self._run_fast_validation(system_name, setup_env=setup_env)
                    report["fast_validation"] = fast_result

                    if fast_result.get("verdict") == "passed":
                        report["logs"].append("Simulation test passed: all core methods available")
                        print(f"    Simulation test passed")

                        if attempt > 0:
                            # The fix produced by auto-fix lives only in the
                            # isolated environment. It MUST be synced back to the
                            # real tree, otherwise the deployed provider still has
                            # the original bug and fails to load at task time. Do
                            # not report the system as passed unless the sync
                            # actually succeeds.
                            synced = self._sync_to_real_environment(system_name)
                            if synced:
                                report["verdict"] = "passed"
                                report["logs"].append(f"Fixed code synced to real environment after {attempt} fix attempts")
                            else:
                                report["verdict"] = "failed_sync"
                                report["logs"].append(f"Fix passed simulation after {attempt} fix attempt(s) but could NOT be synced to the real environment; deployed provider left unchanged")
                                print(f"    Failed to sync fixed code to real environment")
                        else:
                            report["verdict"] = "passed"

                        break
                    else:
                        report["verdict"] = "failed_simulation"
                        error_summary = "; ".join(fast_result.get("errors", [])[:3])
                        report["logs"].append(f"Simulation test failed: {error_summary}")
                        report["error_logs"] = fast_result.get("error_logs", [])
                        print(f"    Simulation test failed: {error_summary[:100]}")

                        if self.enable_auto_fix and attempt < self.max_fix_attempts - 1:
                            print(f"  [Auto-Fix] Attempting to fix errors...")

                            swe_validator = SWEAgentValidator(
                                work_dir=self.temp_dir,
                                isolated_env_dir=self.isolated_env_dir
                            )

                            fix_success = swe_validator.fix_memory_system(
                                memory_system_name=system_name,
                                error_report=fast_result
                            )

                            if fix_success:
                                print(f"  [Auto-Fix] Fix applied, retrying validation...")
                            else:
                                print(f"  [Auto-Fix] Fix failed")
                                break
                        else:
                            break
            finally:
                # Clean up the isolated environment exactly once, after the
                # retry/sync loop has finished. Doing it here (instead of inside
                # _run_fast_validation) guarantees a fix is never deleted before
                # it has been synced to the real tree.
                if self.cleanup_temp and self.isolated_env_dir and self.isolated_env_dir.exists():
                    try:
                        shutil.rmtree(self.isolated_env_dir)
                        print(f"    Cleaned up isolated environment")
                    except Exception as e:
                        print(f"    Warning: Failed to cleanup isolated environment: {e}")
        else:
            print(f"  [Real-World Task Test] (Not implemented with fast validation)")
            report["verdict"] = "skipped"
            report["logs"].append("Real-world task test skipped (use_fast_validation=True)")

        return report

    def _static_check(self, code: str) -> Dict:
        """
        Perform static checks on code

        Args:
            code: Python code to check

        Returns:
            Static check result
        """
        issues = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append({
                "type": "syntax_error",
                "description": f"Syntax error: {str(e)}",
                "severity": "high"
            })
            return {"passed": False, "issues": issues}

        if "BaseMemoryProvider" not in code:
            issues.append({
                "type": "missing_base_class",
                "description": "Must inherit from BaseMemoryProvider",
                "severity": "high"
            })

        if "def provide_memory" not in code:
            issues.append({
                "type": "missing_method",
                "description": "Must implement provide_memory method",
                "severity": "high"
            })

        if "def take_in_memory" not in code:
            issues.append({
                "type": "missing_method",
                "description": "Must implement take_in_memory method",
                "severity": "high"
            })

        if "def initialize" not in code:
            issues.append({
                "type": "missing_method",
                "description": "Must implement initialize method",
                "severity": "high"
            })

        required_imports = [
            ("BaseMemoryProvider", "EvolveLab.base_memory"),
            ("MemoryRequest", "EvolveLab.memory_types"),
            ("MemoryResponse", "EvolveLab.memory_types"),
        ]

        for symbol, module in required_imports:
            if symbol in code:
                import_pattern = f"from {module} import"
                if import_pattern not in code:
                    issues.append({
                        "type": "missing_import",
                        "description": f"Uses {symbol} but missing import from {module}",
                        "severity": "medium"
                    })

        return {
            "passed": len([i for i in issues if i["severity"] == "high"]) == 0,
            "issues": issues
        }

    def _extract_error_summary(self, test_report: Dict) -> str:
        """
        Extract error summary from test report

        Args:
            test_report: Test report dictionary

        Returns:
            Error summary string
        """
        errors = []
        for test_result in test_report.get("test_results", []):
            if test_result["status"] in ["error", "failed"]:
                error_msg = test_result.get('error_message', 'Unknown error')
                errors.append(f"{test_result['name']}: {error_msg}")
        return "\n".join(errors[:3]) if errors else "Unknown errors occurred"

    def _extract_test_summary(self, test_report: Dict) -> List[Dict]:
        """
        Extract test summary from test report

        Args:
            test_report: Test report dictionary

        Returns:
            List of test summaries
        """
        summaries = []
        for test_result in test_report.get("test_results", []):
            summaries.append({
                "name": test_result["name"],
                "status": test_result["status"],
                "execution_time": test_result.get("execution_time", 0),
                "error_message": test_result.get("error_message") if test_result["status"] != "passed" else None
            })
        return summaries

    def _setup_isolated_environment(self, system_name: str) -> bool:
        """Setup isolated environment for testing"""
        if self.isolated_env_dir and self.isolated_env_dir.exists():
            shutil.rmtree(self.isolated_env_dir, ignore_errors=True)
        
        self.isolated_env_dir = self.temp_dir / f"{system_name}_isolated_env"
        self.isolated_env_dir.mkdir(parents=True, exist_ok=True)
        
        def ignore_patterns(dir, names):
            ignored = set()
            for name in names:
                if name in ['.git', '.svn', '.hg', '__pycache__', '.pytest_cache', '.mypy_cache']:
                    ignored.add(name)
            return ignored
        
        dirs_to_copy = ["EvolveLab"]
        
        for dir_name in dirs_to_copy:
            src_dir = self.project_root / dir_name
            if src_dir.exists():
                dst_dir = self.isolated_env_dir / dir_name
                try:
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True, ignore=ignore_patterns)
                except Exception as e:
                    print(f"    Failed to copy {dir_name}: {e}")
                    return False
        
        real_storage = self.project_root / "storage"
        isolated_storage = self.isolated_env_dir / "storage"
        
        if real_storage.exists():
            try:
                shutil.copytree(real_storage, isolated_storage, dirs_exist_ok=True)
            except Exception as e:
                print(f"    Failed to copy storage: {e}")
                isolated_storage.mkdir(parents=True, exist_ok=True)
        else:
            isolated_storage.mkdir(parents=True, exist_ok=True)
        
        return True

    def _run_fast_validation(self, system_name: str, setup_env: bool = True) -> Dict:
        """
        Fast simulation-based validation in isolated environment
        
        Tests memory system by simulating actual usage:
        1. Import provider module (via PROVIDER_MAPPING)
        2. Get configuration (get_memory_config)
        3. Instantiate provider (provider_class(config=config))
        4. Call initialize() (fails if returns False)
        5. Simulate provide_memory() and take_in_memory() calls
        
        Args:
            system_name: System name to validate
            setup_env: Whether to setup isolated environment (default: True)
                      Set to False on retry attempts to preserve fixes
            
        Returns:
            Validation result with detailed error logs
        """
        result = {
            "verdict": "unknown",
            "tests": {},
            "errors": [],
            "error_logs": [],
            "captured_logs": [],  # Capture all logs for analysis
            "memory_details": {}  # Store memory operation details
        }
        
        if setup_env:
            print(f"    [Simulation] Setting up isolated environment...")
            if not self._setup_isolated_environment(system_name):
                result["verdict"] = "failed"
                result["errors"].append("Failed to setup isolated environment")
                result["error_logs"].append("Could not create isolated environment for testing")
                return result
        else:
            print(f"    [Simulation] Reusing existing isolated environment (preserving fixes)...")
            if not self.isolated_env_dir or not self.isolated_env_dir.exists():
                result["verdict"] = "failed"
                result["errors"].append("Isolated environment does not exist for retry")
                result["error_logs"].append("Expected to reuse isolated environment but it doesn't exist")
                return result
        
        original_cwd = os.getcwd()
        original_pythonpath = os.environ.get("PYTHONPATH", "")
        original_sys_path = sys.path.copy()
        
        # Setup log capture
        import io
        import logging
        log_capture = io.StringIO()
        log_handler = logging.StreamHandler(log_capture)
        log_handler.setLevel(logging.DEBUG)
        log_formatter = logging.Formatter('[%(levelname)s] %(name)s - %(message)s')
        log_handler.setFormatter(log_formatter)
        root_logger = logging.getLogger()
        original_log_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(log_handler)
        
        try:
            # Get absolute path before chdir to avoid relative path issues on retry
            isolated_env_abs = str(self.isolated_env_dir.absolute())
            os.chdir(isolated_env_abs)
            
            # Remove main codebase from sys.path to ensure we only load from isolated env
            main_codebase = str(Path(__file__).parent.parent.parent.absolute())
            sys.path = [p for p in sys.path if not p.startswith(main_codebase)]
            
            sys.path.insert(0, isolated_env_abs)
            os.environ["PYTHONPATH"] = f"{isolated_env_abs}:{original_pythonpath}"
            
            print(f"    [Simulation] Creating model instance...")
            try:
                from FlashOAgents import OpenAIServerModel
                
                custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
                model_config = {
                    "model_id": os.environ.get("DEFAULT_MODEL"),
                    "custom_role_conversions": custom_role_conversions,
                    "max_completion_tokens": 32768,
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "api_base": os.environ.get("OPENAI_API_BASE"),
                }
                model = OpenAIServerModel(**model_config)
                print(f"      Model created: {model_config['model_id']}")
            except Exception as e:
                result["verdict"] = "failed"
                error_msg = f"Failed to create model: {str(e)}"
                result["errors"].append(error_msg)
                result["error_logs"].append(f"Model creation error: {traceback.format_exc()}")
                result["tests"]["model_creation"] = {"status": "failed", "error": error_msg}
                print(f"      {error_msg}")
                return result
            
            print(f"    [Simulation] Test 1: Import module...")
            try:
                # Always clear EvolveLab module cache to ensure fresh import
                # This is critical because the main program may have already imported EvolveLab
                # with old module definitions before the new system was created
                modules_to_clear = [
                    key for key in list(sys.modules.keys())
                    if key == 'EvolveLab' or key.startswith('EvolveLab.')
                ]
                if modules_to_clear:
                    for mod_key in modules_to_clear:
                        del sys.modules[mod_key]
                    if not setup_env:
                        print(f"      Cleared {len(modules_to_clear)} cached modules for fresh import")
                        print(f"      sys.path[0] = {sys.path[0]}")
                
                from EvolveLab.memory_types import MemoryType, PROVIDER_MAPPING, MemoryRequest, MemoryResponse, MemoryStatus, TrajectoryData
                
                # Debug: verify we loaded from isolated env
                if not setup_env:
                    import EvolveLab
                    print(f"      EvolveLab loaded from: {EvolveLab.__file__}")
                
                memory_type = None
                for mt in MemoryType:
                    if mt.value == system_name:
                        memory_type = mt
                        break
                
                if not memory_type:
                    result["verdict"] = "failed"
                    result["errors"].append(f"MemoryType not found: {system_name}")
                    result["error_logs"].append(f"System name {system_name} not in MemoryType enum")
                    return result
                
                if memory_type not in PROVIDER_MAPPING:
                    result["verdict"] = "failed"
                    result["errors"].append(f"MemoryType {memory_type.value} not in PROVIDER_MAPPING")
                    result["error_logs"].append(f"PROVIDER_MAPPING missing {memory_type.value}")
                    return result
                
                class_name, module_name = PROVIDER_MAPPING[memory_type]
                result["tests"]["import"] = {"status": "checking", "class_name": class_name, "module_name": module_name}
                
                module_path = f"EvolveLab.providers.{module_name}"
                
                provider_module = __import__(module_path, fromlist=[class_name])
                provider_class = getattr(provider_module, class_name)
                
                # Debug: show where provider was loaded from
                if not setup_env:
                    print(f"      Provider module loaded from: {provider_module.__file__}")
                
                result["tests"]["import"]["status"] = "passed"
                print(f"      Import successful: {class_name}")
                
            except ImportError as e:
                result["verdict"] = "failed"
                error_msg = f"Import failed: {str(e)}"
                result["errors"].append(error_msg)
                result["error_logs"].append(f"ImportError: {traceback.format_exc()}")
                result["tests"]["import"] = {"status": "failed", "error": error_msg}
                print(f"      {error_msg}")
                return result
            except Exception as e:
                result["verdict"] = "failed"
                error_msg = f"Import error: {str(e)}"
                result["errors"].append(error_msg)
                result["error_logs"].append(f"Exception during import: {traceback.format_exc()}")
                result["tests"]["import"] = {"status": "failed", "error": error_msg}
                print(f"      {error_msg}")
                return result
            
            print(f"    [Simulation] Test 2: Get config and instantiate provider...")
            provider = None
            
            from EvolveLab.config import get_memory_config
            try:
                config = get_memory_config(memory_type)
                # Inject model into config (as done in run_flash_searcher_mm_gaia.py)
                if model is not None:
                    try:
                        config["model"] = model
                    except Exception:
                        pass
            except Exception as e:
                result["verdict"] = "failed"
                error_msg = f"Get config failed: {str(e)}"
                result["errors"].append(error_msg)
                result["error_logs"].append(f"get_memory_config error: {traceback.format_exc()}")
                result["tests"]["get_config"] = {"status": "failed", "error": error_msg}
                print(f"      {error_msg}")
                return result
            
            try:
                provider = provider_class(config=config)
                
                # Check captured logs for ERROR level messages
                captured = log_capture.getvalue()
                error_lines = [line for line in captured.split('\n') if '[ERROR]' in line]
                
                if error_lines:
                    result["verdict"] = "failed"
                    error_msg = f"Instantiation produced ERROR logs: {len(error_lines)} error(s)"
                    result["errors"].append(error_msg)
                    result["error_logs"].append(f"ERROR logs during instantiation:\n" + "\n".join(error_lines))
                    result["tests"]["instantiation"] = {
                        "status": "failed", 
                        "error": error_msg,
                        "error_logs": error_lines
                    }
                    print(f"      {error_msg}")
                    print(f"         {error_lines[0][:100]}...")
                    return result
                
                result["tests"]["instantiation"] = {"status": "passed"}
                print(f"      Instantiation successful (with model)")
            except Exception as e:
                result["verdict"] = "failed"
                error_msg = f"Instantiation failed: {str(e)}"
                result["errors"].append(error_msg)
                result["error_logs"].append(f"Instantiation error: {traceback.format_exc()}")
                result["tests"]["instantiation"] = {"status": "failed", "error": error_msg}
                print(f"      {error_msg}")
                return result
            
            print(f"    [Simulation] Test 3: Call initialize()...")
            try:
                init_success = provider.initialize()
                if not init_success:
                    result["verdict"] = "failed"
                    error_msg = "initialize() returned False"
                    result["errors"].append(error_msg)
                    result["error_logs"].append("initialize() method returned False indicating initialization failure")
                    result["tests"]["initialize"] = {"status": "failed", "returned": False, "error": error_msg}
                    print(f"      {error_msg}")
                    return result
                else:
                    result["tests"]["initialize"] = {"status": "passed", "returned": True}
                    print(f"      initialize() successful")
            except Exception as e:
                result["verdict"] = "failed"
                error_msg = f"initialize() call failed: {str(e)}"
                result["errors"].append(error_msg)
                result["error_logs"].append(f"initialize() error: {traceback.format_exc()}")
                result["tests"]["initialize"] = {"status": "failed", "error": error_msg}
                print(f"      {error_msg}")
                return result
            
            print(f"    [Simulation] Test 4: Call provide_memory()...")
            try:
                test_request = MemoryRequest(
                    query="test query for memory retrieval simulation",
                    context="test context",
                    status=MemoryStatus.BEGIN
                )
                response = provider.provide_memory(test_request)
                
                if not isinstance(response, MemoryResponse):
                    result["tests"]["provide_memory"] = {
                        "status": "warning",
                        "error": f"Return type error: expected MemoryResponse, got {type(response)}"
                    }
                    result["errors"].append("provide_memory() returned incorrect type")
                else:
                    # Record detailed memory content
                    memories_content = []
                    if response.memories:
                        for idx, mem in enumerate(response.memories[:5]):  # First 5 memories
                            memories_content.append({
                                "index": idx,
                                "content_preview": str(mem)[:200] if mem else "Empty",
                                "type": type(mem).__name__
                            })
                    
                    result["memory_details"]["provide_memory"] = {
                        "count": len(response.memories) if response.memories else 0,
                        "memories": memories_content,
                        "response_type": type(response).__name__
                    }
                    
                    result["tests"]["provide_memory"] = {
                        "status": "passed",
                        "memories_count": len(response.memories) if response.memories else 0
                    }
                    print(f"      provide_memory() successful (returned {len(response.memories) if response.memories else 0} memories)")
                    if memories_content:
                        print(f"         Memory preview: {memories_content[0]['content_preview'][:80]}...")
                    
            except Exception as e:
                result["verdict"] = "failed"
                error_msg = f"provide_memory() call failed: {str(e)}"
                result["errors"].append(error_msg)
                result["error_logs"].append(f"provide_memory() error: {traceback.format_exc()}")
                result["tests"]["provide_memory"] = {"status": "failed", "error": error_msg}
                print(f"      {error_msg}")
                return result
            
            print(f"    [Simulation] Test 5: Call take_in_memory()...")
            
            # Capture storage state before take_in_memory
            storage_dir = self.isolated_env_dir / "storage" / system_name
            storage_before = {}
            if storage_dir.exists():
                for storage_file in storage_dir.glob("*.json"):
                    try:
                        with open(storage_file, 'r') as f:
                            content = json.load(f)
                            storage_before[storage_file.name] = {
                                "size": len(json.dumps(content)),
                                "keys": list(content.keys()) if isinstance(content, dict) else None,
                                "item_count": len(content) if isinstance(content, (list, dict)) else None
                            }
                    except Exception:
                        pass
            
            try:
                test_trajectory = TrajectoryData(
                    query="test query for memory ingestion simulation",
                    trajectory=[
                        {"role": "user", "content": "test query"},
                        {"role": "assistant", "content": "test response"}
                    ],
                    result="test result - successful task completion",
                    metadata={"test": True, "is_correct": True, "status": "success"}
                )
                success, description = provider.take_in_memory(test_trajectory)
                
                # Capture storage state after take_in_memory
                storage_after = {}
                if storage_dir.exists():
                    for storage_file in storage_dir.glob("*.json"):
                        try:
                            with open(storage_file, 'r') as f:
                                content = json.load(f)
                                storage_after[storage_file.name] = {
                                    "size": len(json.dumps(content)),
                                    "keys": list(content.keys()) if isinstance(content, dict) else None,
                                    "item_count": len(content) if isinstance(content, (list, dict)) else None,
                                    "content_preview": str(content)[:300] if content else None
                                }
                        except Exception:
                            pass
                
                # Analyze storage changes
                storage_changes = {}
                for filename in set(list(storage_before.keys()) + list(storage_after.keys())):
                    before = storage_before.get(filename, {})
                    after = storage_after.get(filename, {})
                    if before != after:
                        storage_changes[filename] = {
                            "before": before,
                            "after": after,
                            "size_change": after.get("size", 0) - before.get("size", 0)
                        }
                
                result["memory_details"]["take_in_memory"] = {
                    "success": success,
                    "description": description,
                    "storage_before": storage_before,
                    "storage_after": storage_after,
                    "storage_changes": storage_changes
                }
                
                if not isinstance(success, bool):
                    result["tests"]["take_in_memory"] = {
                        "status": "warning",
                        "error": f"Return type error: expected (bool, str), got {type(success)}"
                    }
                    result["errors"].append("take_in_memory() returned incorrect type")
                else:
                    result["tests"]["take_in_memory"] = {
                        "status": "passed",
                        "success": success,
                        "description": description[:100] if description else "",
                        "storage_changes": storage_changes
                    }
                    print(f"      take_in_memory() successful (success={success})")
                    if storage_changes:
                        print(f"         Storage changes: {list(storage_changes.keys())}")
                        for filename, change in list(storage_changes.items())[:2]:
                            print(f"         - {filename}: size changed by {change['size_change']} bytes")
                    else:
                        print(f"         No storage changes detected")
                    
            except Exception as e:
                result["verdict"] = "failed"
                error_msg = f"take_in_memory() call failed: {str(e)}"
                result["errors"].append(error_msg)
                result["error_logs"].append(f"take_in_memory() error: {traceback.format_exc()}")
                result["tests"]["take_in_memory"] = {"status": "failed", "error": error_msg}
                print(f"      {error_msg}")
                return result
            
            # Final check: Look for ERROR logs in any phase (except instantiation which was already checked)
            if result["verdict"] == "unknown":
                captured = log_capture.getvalue()
                all_error_lines = [line for line in captured.split('\n') if '[ERROR]' in line]
                
                # Filter out errors already captured during instantiation
                instantiation_errors = result.get("error_logs", [])
                new_error_lines = []
                for line in all_error_lines:
                    if not any(line in inst_err for inst_err in instantiation_errors):
                        new_error_lines.append(line)
                
                if new_error_lines:
                    result["verdict"] = "failed"
                    error_msg = f"Runtime produced ERROR logs: {len(new_error_lines)} error(s)"
                    result["errors"].append(error_msg)
                    # For errors containing "Raw:", include more lines to show full JSON content
                    error_log_content = "\n".join(new_error_lines[:5])
                    if "Raw:" in error_log_content:
                        # Include more lines to capture the full JSON response
                        error_log_content = "\n".join(new_error_lines[:30])  # Increased from 5 to 30
                    result["error_logs"].append(f"ERROR logs during runtime:\n" + error_log_content)
                    print(f"    {error_msg}")
                    for err_line in new_error_lines[:2]:
                        print(f"       {err_line[:120]}...")
                else:
                    result["verdict"] = "passed"
                    print(f"    [Simulation] All tests passed")
            
        finally:
            # Capture all logs
            all_logs = log_capture.getvalue()
            result["captured_logs"] = all_logs.split('\n') if all_logs else []
            
            root_logger.removeHandler(log_handler)
            root_logger.setLevel(original_log_level)
            log_handler.close()
            
            os.chdir(original_cwd)
            sys.path = original_sys_path
            os.environ["PYTHONPATH"] = original_pythonpath

            # NOTE: the isolated environment is intentionally NOT cleaned up
            # here. On a retry that passes, it still holds the auto-fixed
            # provider that must be synced to the real tree first. Cleanup is
            # handled once, in _validate_single_system, after syncing.

        return result

    def _sync_to_real_environment(self, system_name: str) -> bool:
        """Sync fixed code from isolated environment to real environment"""
        if not self.isolated_env_dir or not self.isolated_env_dir.exists():
            print("    Isolated environment not found, cannot sync")
            return False
        
        isolated_provider = self.isolated_env_dir / "EvolveLab" / "providers" / f"{system_name}_provider.py"
        real_provider = self.project_root / "EvolveLab" / "providers" / f"{system_name}_provider.py"
        
        if isolated_provider.exists():
            if real_provider.exists():
                backup_file = real_provider.with_suffix(f".py.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                shutil.copy2(real_provider, backup_file)
                print(f"    Backup created: {backup_file.name}")
            
            real_provider.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(isolated_provider, real_provider)
            print(f"    Code synced to real environment: {real_provider.name}")
            return True
        else:
            print(f"    Provider file not found in isolated environment: {isolated_provider}")
            return False

