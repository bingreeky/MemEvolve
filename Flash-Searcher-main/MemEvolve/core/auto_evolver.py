#!/usr/bin/env python
# coding=utf-8

"""
AutoEvolver

Outer-loop orchestrator for multi-round memory evolution on a dataset.

"""

import json
import os
import random
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List

from dotenv import load_dotenv

from .memory_evolver import MemoryEvolver
from ..utils.trajectory_tools import TrajectoryFeedbackAggregator
from ..config import (
    DEFAULT_DATASETS,
    EVOLVE_TASK_BATCH_X,
    EVOLVE_EXTRA_SAMPLE_Y,
    EVOLVE_TOP_T,
    EVOLVE_GENERATED_M,
    EVOLVE_RANDOM_SEED,
)

load_dotenv(override=True)


class AutoEvolver:
    """
    Multi-round orchestrator for memory evolution.

    Typical usage:
        auto = AutoEvolver(
            analysis_model_id="gpt-5",
            gen_model_id="gpt-5",
            work_root="runs/auto",
            dataset_name="gaia",
            run_provider=run_provider,
        )
        auto.run(num_rounds=1, eval_limit=50)
    """

    def __init__(
        self,
        analysis_model_id: Optional[str] = None,
        gen_model_id: Optional[str] = None,
        work_root: str = "",
        dataset_name: str = "",
        run_provider: Optional[Callable[[str, str, List[int], Path, Path], Path]] = None,
        default_provider: str = "agent_kb",
        num_systems: int = EVOLVE_GENERATED_M,
        creativity_index: float = 0.5,
        task_batch_x: int = EVOLVE_TASK_BATCH_X,
        top_t: int = EVOLVE_TOP_T,
        extra_sample_y: int = EVOLVE_EXTRA_SAMPLE_Y,
        datasets_config: Optional[Dict[str, str]] = None,
        max_workers: int = 3,
        use_pareto_selection: bool = False,
        clear_storage_per_round: bool = True,
    ):
        if analysis_model_id is None:
            analysis_model_id = os.getenv("ANALYSIS_MODEL", os.getenv("DEFAULT_MODEL", "gpt-5"))
        
        if gen_model_id is None:
            gen_model_id = os.getenv("GENERATION_MODEL", os.getenv("DEFAULT_MODEL", "gpt-5"))
        
        self.analysis_model_id = analysis_model_id
        self.gen_model_id = gen_model_id
        if not work_root:
            raise ValueError("work_root is required")
        if not dataset_name:
            raise ValueError("dataset_name is required")
        if run_provider is None:
            raise ValueError("run_provider is required")
        
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.datasets_config = datasets_config or DEFAULT_DATASETS
        if dataset_name not in self.datasets_config:
            raise KeyError(f"Dataset {dataset_name} not in config: {list(self.datasets_config.keys())}")
        self.dataset_name = dataset_name
        self.dataset_path = Path(self.datasets_config[dataset_name])
        self.run_provider = run_provider
        self.default_provider = default_provider
        self.num_systems = num_systems
        self.creativity_index = creativity_index
        self.task_batch_x = task_batch_x
        self.top_t = top_t
        self.extra_sample_y = extra_sample_y
        self.max_workers = max_workers
        self.use_pareto_selection = use_pareto_selection
        self.clear_storage_per_round = clear_storage_per_round
        self.state = self._load_state()

    def _state_file(self) -> Path:
        return self.work_root / "evolve_state.json"

    def _checkpoint_file(self, round_num: int) -> Path:
        return self.work_root / f"round_{round_num:02d}" / "checkpoint.json"

    def _load_state(self) -> Dict[str, Any]:
        if self._state_file().exists():
            with open(self._state_file(), "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "round": 0,
            "dataset_name": self.dataset_name,
            "dataset_cursor": 0,
            "best_provider": self.default_provider,
            "history": []
        }

    def _save_state(self):
        with open(self._state_file(), "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def _load_checkpoint(self, round_num: int) -> Optional[Dict[str, Any]]:
        """Load round checkpoint if exists"""
        checkpoint_file = self._checkpoint_file(round_num)
        if checkpoint_file.exists():
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _get_existing_created_systems(self, round_dir: Path) -> list[str]:
        """Get list of already created systems from created_system.json"""
        created_file = round_dir / "created_system.json"
        if created_file.exists():
            try:
                with open(created_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("created", [])
            except Exception:
                pass
        return []
    
    def _get_existing_validated_systems(self, round_dir: Path) -> list[str]:
        """Get list of already validated systems from validated_systems.json"""
        validated_file = round_dir / "validated_systems.json"
        if validated_file.exists():
            try:
                with open(validated_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("validated", [])
            except Exception:
                pass
        return []
    
    def _save_checkpoint(self, round_num: int, checkpoint_data: Dict[str, Any]):
        """Save round checkpoint"""
        checkpoint_file = self._checkpoint_file(round_num)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

    def _delete_checkpoint(self, round_num: int):
        """Delete checkpoint after round completes"""
        checkpoint_file = self._checkpoint_file(round_num)
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    def _run_memory_evolver(
        self,
        round_dir: Path,
        checkpoint: Optional[Dict[str, Any]] = None,
        base_provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run memory evolution pipeline with checkpoint support.
        
        For each system to be generated:
        1. Re-run analysis (fresh perspective on the same task logs)
        2. Generate ONE new memory system configuration
        3. Create the memory system files
        4. Validate the created system
        
        This ensures each system is generated independently with its own analysis insights,
        rather than reusing a single analysis report for multiple systems.
        
        Args:
            base_provider: the reigning champion that produced this round's base_logs.
                The diagnosis prompt, mutation template source and inspected memory
                storage all derive from it. Must be passed explicitly (it is NOT the
                original --provider, which becomes stale from round 1 onward).
        
        Returns dict with keys: analysis, generation, creation, validation, validated_systems
        """
        if not base_provider or not base_provider.strip():
            raise ValueError(
                "base_provider is required: must be the current champion "
                "(state['best_provider']) that produced this round's base_logs, "
                "not the initial --provider"
            )
        
        evolver = MemoryEvolver(
            work_dir=str(round_dir),
            analysis_model_id=self.analysis_model_id,
            gen_model_id=self.gen_model_id,
        )
        
        result = {
            "analysis": None,
            "generation": [],
            "creation": [],
            "validation": [],
            "validated_systems": []
        }
        
        # Load checkpoint state and check for manual CLI fixes
        systems_to_generate = self.num_systems
        validated_from_checkpoint = []
        
        if checkpoint and "validated_systems" in checkpoint:
            validated_from_checkpoint = checkpoint["validated_systems"].copy()
        
        # Check if user manually fixed via CLI (check MemoryEvolver state)
        validated_from_state = []
        if evolver.state.get("phases", {}).get("validate", {}).get("completed"):
            validated_from_state = evolver.state["phases"]["validate"].get("validated_systems", [])
            if validated_from_state:
                print(f"[CLI Fix Detected] Found {len(validated_from_state)} validated systems from manual CLI commands")
        
        # Use the union of both (prefer CLI state as it's more recent)
        all_validated = list(set(validated_from_checkpoint + validated_from_state))
        if all_validated:
            existing_count = len(all_validated)
            systems_to_generate = max(0, self.num_systems - existing_count)
            print(f"[Checkpoint] Found {existing_count} validated systems (checkpoint: {len(validated_from_checkpoint)}, CLI: {len(validated_from_state)})")
            print(f"[Checkpoint] Need to generate {systems_to_generate} more systems")
            result["validated_systems"] = all_validated
        
        if systems_to_generate <= 0:
            print(f"[Checkpoint] Skipping generation (already have {len(result['validated_systems'])} validated systems)")
            # Still load the previous results from state if they exist
            if evolver.state.get("phases", {}).get("analyze", {}).get("completed"):
                result["analysis"] = {"success": True, "report_path": evolver.state["phases"]["analyze"].get("output")}
            if evolver.state.get("phases", {}).get("generate", {}).get("completed"):
                result["generation"] = [{"success": True, "config_path": evolver.state["phases"]["generate"].get("output")}]
            if evolver.state.get("phases", {}).get("create", {}).get("completed"):
                result["creation"] = [{"success": True, "created": evolver.state["phases"]["create"].get("created_systems", [])}]
            if evolver.state.get("phases", {}).get("validate", {}).get("completed"):
                result["validation"] = [{"success": True, "validated": evolver.state["phases"]["validate"].get("validated_systems", [])}]
            return result
        
        # Check for existing generated/created/validated systems
        existing_generated = evolver._get_all_generated_system_paths()
        existing_created = self._get_existing_created_systems(round_dir)
        existing_validated = self._get_existing_validated_systems(round_dir)
        
        print(f"\n[Resume Check]")
        print(f"  Existing generated: {len(existing_generated)} file(s)")
        print(f"  Existing created: {len(existing_created)} system(s)")
        print(f"  Existing validated: {len(existing_validated)} system(s)")
        
        # Determine starting point
        start_idx = len(result['validated_systems'])
        if start_idx > 0:
            print(f"  Resuming from system {start_idx + 1}")
        
        # ===================================================================
        # PHASE 1: Generation - Generate all systems first
        # ===================================================================
        print(f"\n{'='*70}")
        print(f"[PHASE 1: GENERATION] Generating {systems_to_generate - start_idx} system(s)")
        print(f"{'='*70}")
        
        for system_idx in range(start_idx, systems_to_generate):
            print(f"\n[Generation] System {system_idx + 1}/{self.num_systems}")
            
            # Analysis (only if not already done or if checkpoint says to skip)
            if system_idx == 0:
                # First system: run analysis
                if checkpoint and checkpoint.get("evolution_completed", {}).get("analysis"):
                    print(f"[Checkpoint] Skipping analysis (already completed)")
                    result["analysis"] = checkpoint.get("evolution_results", {}).get("analysis", {"success": True})
                else:
                    try:
                        result["analysis"] = evolver.analyze(
                            task_logs_dir=str(round_dir / "base_logs"),
                            default_provider=base_provider,
                        )
                        if not result["analysis"].get("success"):
                            print(f"Warning: Analysis failed, continuing with limited info")
                    except Exception as e:
                        print(f"Error in analysis: {e}")
                        result["analysis"] = {"success": False, "error": str(e)}
                        # Continue anyway, might have partial analysis
            else:
                # Subsequent systems: re-run analysis for fresh perspective
                try:
                    result["analysis"] = evolver.analyze(
                        task_logs_dir=str(round_dir / "base_logs"),
                        default_provider=base_provider,
                    )
                    if not result["analysis"].get("success"):
                        print(f"Warning: Analysis failed, continuing with limited info")
                except Exception as e:
                    print(f"Error in analysis: {e}")
                    result["analysis"] = {"success": False, "error": str(e)}
            
            # Generation (single system)
            # Check if system was already generated
            all_generated = evolver._get_all_generated_system_paths()
            if len(all_generated) > system_idx:
                # System already generated, load it
                gen_config_path = all_generated[system_idx]
                print(f"[Checkpoint] System {system_idx + 1} already generated: {gen_config_path.name}")
                with open(gen_config_path, 'r', encoding='utf-8') as f:
                    gen_config = json.load(f)
                gen_result = {
                    "success": True,
                    "config_path": str(gen_config_path),
                    "config": gen_config
                }
                result["generation"].append(gen_result)
            else:
                # Generate new system
                try:
                    gen_result = evolver.generate(
                        creativity_index=self.creativity_index,
                    )
                    result["generation"].append(gen_result)
                    if not gen_result.get("success"):
                        error_msg = f"Generation failed for system {system_idx + 1}"
                        print(f"\nError: {error_msg}")
                        print(f"\n" + "="*70)
                        print(f"Manual fix options:")
                        print(f"  1. Review analysis: {round_dir}/analysis_report.json")
                        print(f"  2. Manually regenerate:")
                        print(f"     python evolve_cli.py --work-dir {round_dir} generate --creativity {self.creativity_index}")
                        print(f"  3. After fix, resume: Re-run the same command")
                        print(f"="*70 + "\n")
                        raise RuntimeError(error_msg)
                except RuntimeError:
                    raise
                except Exception as e:
                    error_msg = f"Generation error for system {system_idx + 1}: {e}"
                    print(f"\nError: {error_msg}")
                    result["generation"].append({"success": False, "error": str(e)})
                    print(f"\n" + "="*70)
                    print(f"Manual fix options:")
                    print(f"  1. Review analysis: {round_dir}/analysis_report.json")
                    print(f"  2. Manually regenerate:")
                    print(f"     python evolve_cli.py --work-dir {round_dir} generate --creativity {self.creativity_index}")
                    print(f"  3. After fix, resume: Re-run the same command")
                    print(f"="*70 + "\n")
                    raise RuntimeError(error_msg)
        
        print(f"[PHASE 1: GENERATION] Completed generating {len(result['generation'])} system(s)")
        
        # ===================================================================
        # PHASE 2: Creation - Create all generated systems
        # ===================================================================
        print(f"\n{'='*70}")
        print(f"[PHASE 2: CREATION] Creating {len(result['generation'])} system(s)")
        print(f"{'='*70}")
        
        for system_idx in range(len(result['generation'])):
            gen_result = result['generation'][system_idx]
            print(f"\n[Creation] System {system_idx + 1}/{len(result['generation'])}")
            
            # Extract system name from generated config
            gen_config = gen_result.get("config", {})
            system_name = gen_config.get("memory_type_info", {}).get("enum_value", "")
            
            # Check if system was already created
            if system_name in existing_created:
                print(f"[Checkpoint] System {system_idx + 1} ({system_name}) already created")
                create_result = {
                    "success": True,
                    "created": [system_name],
                    "failed": []
                }
                result["creation"].append(create_result)
            else:
                # Create system
                try:
                    create_result = evolver.create(config_path=gen_result.get("config_path"))
                    result["creation"].append(create_result)
                    if not create_result.get("success"):
                        error_msg = f"Creation failed for system {system_idx + 1}"
                        print(f"\nError: {error_msg}")
                        print(f"\n" + "="*70)
                        print(f"Manual fix options:")
                        print(f"  1. Review generated config: {round_dir}/generated_system.json")
                        print(f"  2. Fix conflicts or regenerate:")
                        print(f"     python evolve_cli.py --work-dir {round_dir} generate --creativity {self.creativity_index}")
                        print(f"  3. Manually create system:")
                        print(f"     python evolve_cli.py --work-dir {round_dir} create")
                        print(f"  4. After fix, resume: Re-run the same command")
                        print(f"="*70 + "\n")
                        raise RuntimeError(error_msg)
                except RuntimeError:
                    raise
                except Exception as e:
                    error_msg = f"Creation error for system {system_idx + 1}: {e}"
                    print(f"\nError: {error_msg}")
                    result["creation"].append({"success": False, "error": str(e)})
                    print(f"\n" + "="*70)
                    print(f"Manual fix options:")
                    print(f"  1. Review error above and generated config: {round_dir}/generated_system.json")
                    print(f"  2. Manually create after fixing issues:")
                    print(f"     python evolve_cli.py --work-dir {round_dir} create")
                    print(f"  3. After fix, resume: Re-run the same command")
                    print(f"="*70 + "\n")
                    raise RuntimeError(error_msg)
        
        print(f"[PHASE 2: CREATION] Completed creating {len(result['creation'])} system(s)")
        
        # ===================================================================
        # PHASE 3: Validation - Validate all created systems
        # ===================================================================
        print(f"\n{'='*70}")
        print(f"[PHASE 3: VALIDATION] Validating {len(result['creation'])} system(s)")
        print(f"{'='*70}")
        
        for system_idx in range(len(result['creation'])):
            create_result = result['creation'][system_idx]
            gen_result = result['generation'][system_idx]
            print(f"\n[Validation] System {system_idx + 1}/{len(result['creation'])}")
            
            created_systems = create_result.get("created", [])
            
            if not created_systems:
                error_msg = "No systems created"
                print(f"\nError: {error_msg}")
                raise RuntimeError(error_msg)
            
            system_to_validate = created_systems[0]
            
            # Check if system was already validated
            if system_to_validate in existing_validated:
                print(f"[Checkpoint] System {system_idx + 1} ({system_to_validate}) already validated")
                validation_result = {
                    "success": True,
                    "validated": [system_to_validate],
                    "failed": [],
                    "details": {}
                }
                result["validation"].append(validation_result)
                validated_systems = [system_to_validate]
            else:
                # Validate system
                try:
                    validation_result = evolver.validate(
                        dataset_name=self.dataset_name,
                        datasets_config=self.datasets_config,
                        config_path=gen_result.get("config_path"),
                        created_systems=created_systems
                    )
                    result["validation"].append(validation_result)
                    validated_systems = validation_result.get("validated", [])
                    
                    if not validated_systems:
                        failed_info = validation_result.get("failed", [])
                        error_msg = f"Validation failed for system {system_idx + 1}"
                        print(f"\nError: {error_msg}")
                        if failed_info:
                            print(f"Failure details: {failed_info}")
                        created = create_result.get("created", [])
                        if created:
                            failed_system = created[0]
                            provider_file = f"EvolveLab/providers/{failed_system}_provider.py"
                            print(f"\n" + "="*70)
                            print(f"Manual fix options:")
                            print(f"")
                            print(f"Option A: Fix the existing code directly")
                            print(f"  1. Edit provider code: {provider_file}")
                            print(f"  2. Revalidate the fixed system:")
                            print(f"     python evolve_cli.py --work-dir {round_dir} validate")
                            print(f"  3. Resume: Re-run the same command")
                            print(f"")
                            print(f"Option B: Delete and regenerate")
                            print(f"  1. Delete failed system:")
                            print(f"     python evolve_cli.py delete --memory-type {failed_system.upper()} --yes")
                            print(f"  2. Regenerate with new config:")
                            print(f"     python evolve_cli.py --work-dir {round_dir} generate --creativity {self.creativity_index}")
                            print(f"     python evolve_cli.py --work-dir {round_dir} create")
                            print(f"     python evolve_cli.py --work-dir {round_dir} validate")
                            print(f"  3. Resume: Re-run the same command")
                            print(f"="*70 + "\n")
                        else:
                            print(f"\n" + "="*70)
                            print(f"To resume: Re-run the same command to retry from checkpoint")
                            print(f"="*70 + "\n")
                        raise RuntimeError(error_msg)
                except RuntimeError:
                    raise
                except Exception as e:
                    error_msg = f"Validation error for system {system_idx + 1}: {e}"
                    print(f"\nError: {error_msg}")
                    result["validation"].append({"success": False, "error": str(e)})
                    print(f"\n" + "="*70)
                    print(f"Manual fix options:")
                    print(f"  1. Review error above and validation status:")
                    print(f"     python evolve_cli.py --work-dir {round_dir} status")
                    print(f"  2. Manually validate after fixing issues:")
                    print(f"     python evolve_cli.py --work-dir {round_dir} validate")
                    print(f"  3. After fix, resume: Re-run the same command")
                    print(f"="*70 + "\n")
                    raise RuntimeError(error_msg)
            
            result["validated_systems"].extend(validated_systems)
        
        print(f"\n[PHASE 3: VALIDATION] Completed validating {len(result['validation'])} system(s)")
        print(f"[Evolution] Total validated systems: {len(result['validated_systems'])}")
        
        return result

    def _evaluate_providers_parallel(
        self, 
        providers: List[str], 
        tasks: List[int], 
        round_dir: Path,
        checkpoint: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple providers in parallel with checkpoint support.
        """
        round_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        
        # Load completed evaluations from checkpoint
        completed_providers = set()
        if checkpoint and "eval_results" in checkpoint:
            results = checkpoint["eval_results"].copy()
            completed_providers = set(results.keys())
            print(f"[Checkpoint] Found {len(completed_providers)} completed evaluations")
        
        # Filter out already completed providers
        providers_to_run = [p for p in providers if p not in completed_providers]
        
        if not providers_to_run:
            print(f"[Checkpoint] All providers already evaluated")
            return results
        
        print(f"[Parallel] Evaluating {len(providers_to_run)} providers with {self.max_workers} workers")
        
        def evaluate_single_provider(provider: str) -> tuple[str, Dict[str, Any]]:
            """Evaluate a single provider"""
            try:
                logs_dir = self.run_provider(
                    self.dataset_name, 
                    provider, 
                    tasks, 
                    self.dataset_path, 
                    round_dir / provider
                )
                aggregator = TrajectoryFeedbackAggregator(str(logs_dir))
                agg = aggregator.aggregate()
                return provider, {
                    "logs_dir": str(logs_dir),
                    "summary": agg.get("summary", {}),
                    "per_task": agg.get("per_task", []),
                    "success": True
                }
            except Exception as e:
                print(f"Error evaluating {provider}: {e}")
                traceback.print_exc()
                return provider, {
                    "logs_dir": str(round_dir / provider),
                    "summary": {"accuracy": 0, "tokens": {"total_tokens": 999999}},
                    "per_task": [],
                    "success": False,
                    "error": str(e)
                }
        
        # Run evaluations in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_provider = {
                executor.submit(evaluate_single_provider, provider): provider 
                for provider in providers_to_run
            }
            
            for future in as_completed(future_to_provider):
                provider, result = future.result()
                results[provider] = result
                print(f"[Parallel] Completed evaluation for {provider}")
        
        return results

    def _compute_avg_execution_time(self, eval_result: Dict[str, Any]) -> float:
        """
        Compute average execution time from evaluation results.
        Reads from log files where each task result JSON contains metrics.elapsed_time.
        
        Expected JSON structure in log files:
        {
            "metrics": {
                "elapsed_time": <float>,  # execution time in seconds
                ...
            },
            ...
        }
        
        Returns:
            Average execution time in seconds, or 0.0 if not available
        """
        import json
        
        # Read from log files (TrajectoryFeedbackAggregator doesn't extract elapsed_time)
        logs_dir = eval_result.get("logs_dir")
        if logs_dir and Path(logs_dir).exists():
            try:
                task_files = list(Path(logs_dir).glob("*.json"))
                if not task_files:
                    return 0.0
                
                execution_times = []
                for task_file in task_files:
                    try:
                        with open(task_file, "r", encoding="utf-8") as f:
                            task_data = json.load(f)
                            # Extract elapsed_time from metrics.elapsed_time
                            metrics = task_data.get("metrics", {})
                            if isinstance(metrics, dict):
                                elapsed_time = metrics.get("elapsed_time")
                                # Only include valid positive values
                                if isinstance(elapsed_time, (int, float)) and elapsed_time > 0:
                                    execution_times.append(float(elapsed_time))
                    except (json.JSONDecodeError, IOError, KeyError, ValueError) as e:
                        # Skip files that can't be read or don't have valid data
                        continue
                
                if execution_times:
                    return sum(execution_times) / len(execution_times)
            except Exception:
                pass
        
        return 0.0

    def _pareto_select_top(self, eval_results: Dict[str, Any], k: int) -> List[str]:
        """
        Select top k memory systems using Pareto optimality-based sorting.
        
        Multi-objective optimization:
        - Primary objective: Task success rate (accuracy, higher is better)
        - Secondary objectives: Computational cost (total_tokens, lower is better), 
          execution time (elapsed_time, lower is better)
        
        Steps:
        1. Extract multi-dimensional metric vectors for each architecture
        2. Assign Pareto ranks (non-dominated sorting)
        3. For architectures with the same rank, use scalarized score (weighted composite score) to break ties
        4. Select Top-K architectures
        """
        # Extract multi-dimensional metrics
        candidates = []
        for provider, res in eval_results.items():
            if not res.get("success", True):
                continue
            summary = res.get("summary", {})
            
            # Extract metrics (for minimization objectives, we keep original values)
            accuracy = summary.get("accuracy", 0.0)  # Higher is better
            total_tokens = summary.get("tokens", {}).get("total_tokens", 999999)
            
            # Calculate average execution time from per_task data or original log files
            avg_execution_time = self._compute_avg_execution_time(res)
            
            candidates.append({
                "provider": provider,
                "accuracy": accuracy,  # Higher is better
                "total_tokens": total_tokens,  # Lower is better
                "execution_time": avg_execution_time,  # Lower is better
            })
        
        if not candidates:
            return []
        
        # Compute Pareto ranks (non-dominated sorting)
        def dominates(candidate1: Dict, candidate2: Dict) -> bool:
            """
            Determine if candidate1 dominates candidate2.
            Dominance relationship: candidate1 is no worse than candidate2 on all objectives,
            and strictly better on at least one objective.
            """
            # For "higher is better" objectives (accuracy), require candidate1 >= candidate2
            # For "lower is better" objectives (tokens, execution_time), require candidate1 <= candidate2
            acc1, tok1, time1 = candidate1["accuracy"], candidate1["total_tokens"], candidate1["execution_time"]
            acc2, tok2, time2 = candidate2["accuracy"], candidate2["total_tokens"], candidate2["execution_time"]
            
            # candidate1 is no worse than candidate2 on all objectives
            not_worse = (acc1 >= acc2 and tok1 <= tok2 and time1 <= time2)
            # candidate1 is strictly better than candidate2 on at least one objective
            strictly_better = (acc1 > acc2 or tok1 < tok2 or time1 < time2)
            
            return not_worse and strictly_better
        
        # Assign Pareto ranks
        pareto_ranks = {}
        remaining = candidates.copy()
        current_rank = 1
        
        while remaining:
            # Find current rank's Pareto front (candidates not dominated by any other candidate)
            front = []
            for c1 in remaining:
                is_dominated = False
                for c2 in remaining:
                    if c1 != c2 and dominates(c2, c1):
                        is_dominated = True
                        break
                if not is_dominated:
                    front.append(c1)
            
            # Assign rank to candidates in the front
            for c in front:
                pareto_ranks[c["provider"]] = current_rank
                remaining.remove(c)
            
            current_rank += 1
        
        # For candidates with the same rank, use scalarized score for sorting
        def compute_scalarized_score(candidate: Dict) -> float:
            """
            Compute weighted composite score for breaking ties.
            Weight settings:
            - accuracy: 0.6 (primary objective)
            - token efficiency: 0.25 (cost)
            - execution time efficiency: 0.15 (latency)
            """
            accuracy = candidate["accuracy"]
            tokens = candidate["total_tokens"]
            execution_time = candidate["execution_time"]
            
            # Get max and min values for all candidates for min-max normalization
            all_tokens = [c["total_tokens"] for c in candidates]
            all_times = [c["execution_time"] for c in candidates]
            min_tokens = min(all_tokens) if all_tokens else 0
            max_tokens = max(all_tokens) if all_tokens else 1
            min_time = min(all_times) if all_times else 0
            max_time = max(all_times) if all_times else 1
            
            # Min-max normalization: for "lower is better" objectives, normalize and invert
            # token_score: tokens lower is better, normalize to [0,1], lower values get higher scores
            if max_tokens > min_tokens:
                token_normalized = (max_tokens - tokens) / (max_tokens - min_tokens)
            else:
                token_normalized = 1.0  # All values are the same, give full score
            token_score = token_normalized
            
            # time_score: execution time lower is better
            if max_time > min_time:
                time_normalized = (max_time - execution_time) / (max_time - min_time)
            else:
                time_normalized = 1.0  # All values are the same, give full score
            time_score = time_normalized
            
            # Weighted composite score
            score = 0.6 * accuracy + 0.25 * token_score + 0.15 * time_score
            return score
        
        # Sort by rank and composite score
        ranked_candidates = []
        for candidate in candidates:
            rank = pareto_ranks.get(candidate["provider"], 999)
            score = compute_scalarized_score(candidate)
            ranked_candidates.append({
                "provider": candidate["provider"],
                "pareto_rank": rank,
                "scalarized_score": score,
            })
        
        # Sort: first by Pareto rank (lower is better), then by composite score (higher is better)
        ranked_candidates.sort(key=lambda x: (x["pareto_rank"], -x["scalarized_score"]))
        
        # Select Top-K
        selected = [c["provider"] for c in ranked_candidates[:k]]
        return selected

    def _select_top(self, eval_results: Dict[str, Any], k: int) -> List[str]:
        """
        Select top k memory systems.
        If use_pareto_selection is True, use Pareto sorting; otherwise use traditional accuracy+token efficiency sorting.
        """
        if self.use_pareto_selection:
            return self._pareto_select_top(eval_results, k)
        
        # Traditional sorting method (original logic)
        scored = []
        for provider, res in eval_results.items():
            if not res.get("success", True):
                continue
            summary = res.get("summary", {})
            accuracy = summary.get("accuracy", 0)
            total_tokens = summary.get("tokens", {}).get("total_tokens", 999999)
            scored.append((provider, accuracy, total_tokens))
        
        scored.sort(key=lambda x: (-x[1], x[2]))
        return [p for p, _, _ in scored[:k]]

    def _select_tasks(self, start_idx: int, count: int) -> List[int]:
        """Select tasks from dataset"""
        return list(range(start_idx, start_idx + count))

    def run(self, num_rounds: int = 1, eval_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run multi-round evolution with checkpoint support.
        """
        history = []
        current_provider = self.state.get("best_provider", self.default_provider)
        dataset_cursor = self.state.get("dataset_cursor", 0)
        round_start = self.state.get("round", 0)

        random.seed(EVOLVE_RANDOM_SEED)

        for r in range(round_start, round_start + num_rounds):
            print(f"\n{'='*70}")
            print(f"Round {r}")
            print(f"{'='*70}")
            
            round_dir = self.work_root / f"round_{r:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            
            # Load checkpoint if exists
            checkpoint = self._load_checkpoint(r)
            if checkpoint:
                print(f"[Checkpoint] Resuming round {r} from checkpoint")
            
            try:
                # Step 0: Rename base provider storage for fair evolution
                # Convention: storage path is ./storage/{provider_name}/
                if self.clear_storage_per_round:
                    if checkpoint and checkpoint.get("storage_renamed"):
                        print(f"[Checkpoint] Storage already renamed for this round")
                    else:
                        print(f"\n[Step 0: Storage Cleanup]")
                        print(f"[Fairness] Renaming base provider storage to ensure fair evolution")
                        
                        storage_dir = Path(f"./storage/{current_provider}")
                        if storage_dir.exists():
                            # Rename with timestamp to preserve evolution data
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            backup_name = f"{current_provider}_round{r:02d}_{timestamp}"
                            backup_dir = storage_dir.parent / backup_name
                            
                            shutil.move(str(storage_dir), str(backup_dir))
                            print(f"[Fairness] Renamed {storage_dir} -> {backup_dir}")
                            print(f"[Fairness] Provider will start with empty storage")
                        else:
                            print(f"[Fairness] No existing storage for {current_provider}")
                        
                        checkpoint = checkpoint or {}
                        checkpoint["storage_renamed"] = True
                        self._save_checkpoint(r, checkpoint)
                else:
                    print(f"\n[Step 0: Storage Cleanup]")
                    print(f"[Mode] Storage clearing disabled - providers keep previous knowledge")
                
                # Step 1: Run base provider on x tasks
                batch_indices = self._select_tasks(dataset_cursor, self.task_batch_x)
                
                if checkpoint and checkpoint.get("step_completed", 0) >= 1:
                    print(f"[Checkpoint] Skipping Step 1 (base logs exist)")
                else:
                    print(f"\nStep 1: Running base provider on {len(batch_indices)} tasks")
                    base_logs = self.run_provider(
                        self.dataset_name, 
                        current_provider, 
                        batch_indices, 
                        self.dataset_path, 
                        round_dir / "base_logs"
                    )
                    checkpoint = checkpoint or {}
                    checkpoint["step_completed"] = 1
                    checkpoint["batch_indices"] = batch_indices
                    self._save_checkpoint(r, checkpoint)

                # Step 2: Evolve from base logs
                if checkpoint and checkpoint.get("step_completed", 0) >= 2:
                    print(f"[Checkpoint] Step 2 marked complete, checking for updates...")
                    # Even if step 2 is complete, check for CLI manual fixes
                    evolution = self._run_memory_evolver(round_dir, checkpoint, base_provider=current_provider)
                    validated_systems = evolution.get("validated_systems", [])
                    
                    # If validated_systems changed (due to CLI fixes), update checkpoint
                    checkpoint_validated = checkpoint.get("validated_systems", [])
                    if set(validated_systems) != set(checkpoint_validated):
                        print(f"[CLI Fix Detected] Updating checkpoint with new validated systems: {validated_systems}")
                        checkpoint["validated_systems"] = validated_systems
                        # Ensure evolution_results exists before updating
                        if "evolution_results" not in checkpoint:
                            checkpoint["evolution_results"] = {}
                        checkpoint["evolution_results"]["validation"] = evolution.get("validation", [])
                        self._save_checkpoint(r, checkpoint)
                else:
                    print(f"\nStep 2: Evolving memory systems")
                    evolution = self._run_memory_evolver(round_dir, checkpoint, base_provider=current_provider)
                    validated_systems = evolution.get("validated_systems", [])
                    
                    checkpoint = checkpoint or {}
                    checkpoint["step_completed"] = 2
                    checkpoint["evolution_results"] = {
                        "analysis": evolution.get("analysis"),
                        "generation": evolution.get("generation"),
                        "creation": evolution.get("creation"),
                        "validation": evolution.get("validation"),
                    }
                    checkpoint["validated_systems"] = validated_systems
                    self._save_checkpoint(r, checkpoint)
                
                # Use validated systems instead of created systems
                providers_to_eval = [current_provider] + validated_systems
                print(f"[Systems] Evaluating {len(providers_to_eval)} providers: {providers_to_eval}")

                # Step 3: Evaluate base + validated systems on same batch
                if checkpoint and checkpoint.get("step_completed", 0) >= 3:
                    print(f"[Checkpoint] Resuming Step 3 (batch evaluation)")
                    eval_results = checkpoint.get("eval_results", {})
                else:
                    print(f"\nStep 3: Evaluating {len(providers_to_eval)} providers on batch")
                    eval_results = {}
                
                eval_results = self._evaluate_providers_parallel(
                    providers=providers_to_eval,
                    tasks=batch_indices,
                    round_dir=round_dir / "eval_batch",
                    checkpoint={"eval_results": eval_results} if eval_results else None
                )
                
                checkpoint["step_completed"] = 3
                checkpoint["eval_results"] = eval_results
                self._save_checkpoint(r, checkpoint)
                
                top_candidates = self._select_top(eval_results, k=self.top_t)
                print(f"[Selection] Top {len(top_candidates)} candidates: {top_candidates}")

                # Step 4: Finalists run on (y sampled + new x)
                sampled_tasks = random.sample(batch_indices, min(self.extra_sample_y, len(batch_indices)))
                new_tasks = self._select_tasks(dataset_cursor + self.task_batch_x, self.task_batch_x)
                finalist_tasks = sampled_tasks + new_tasks
                finalists = top_candidates[:self.top_t]
                
                if checkpoint and checkpoint.get("step_completed", 0) >= 4:
                    print(f"[Checkpoint] Resuming Step 4 (finalist evaluation)")
                    finalist_results = checkpoint.get("finalist_results", {})
                else:
                    print(f"\nStep 4: Evaluating {len(finalists)} finalists on {len(finalist_tasks)} tasks")
                    finalist_results = {}
                
                finalist_results = self._evaluate_providers_parallel(
                    providers=finalists,
                    tasks=finalist_tasks,
                    round_dir=round_dir / "eval_finalists",
                    checkpoint={"eval_results": finalist_results} if finalist_results else None
                )
                
                checkpoint["step_completed"] = 4
                checkpoint["finalist_results"] = finalist_results
                self._save_checkpoint(r, checkpoint)
                
                winner = self._select_top(finalist_results, k=1)[0] if finalist_results else current_provider
                print(f"[Winner] {winner}")

                # Advance dataset cursor
                dataset_cursor += self.task_batch_x * 2

                # Create round summary
                summary = {
                    "round": r,
                    "timestamp": datetime.now().isoformat(),
                    "dataset": self.dataset_name,
                    "batch_tasks": batch_indices,
                    "finalist_tasks": finalist_tasks,
                    "evolution": {
                        "analysis": evolution.get("analysis"),
                        "generation": evolution.get("generation"),
                        "creation": evolution.get("creation"),
                        "validation": evolution.get("validation"),
                        "validated_systems": validated_systems,
                    },
                    "eval_results": eval_results,
                    "finalist_results": finalist_results,
                    "winner": winner,
                    "best_provider_before": current_provider,
                    "best_provider_after": winner,
                    "dataset_cursor_after": dataset_cursor,
                }
                history.append(summary)

                # Save round summary
                summary_path = round_dir / "round_summary.json"
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)

                # Update global state
                current_provider = winner
                self.state.update({
                    "round": r + 1,
                    "dataset_cursor": dataset_cursor,
                    "best_provider": current_provider,
                })
                self.state["history"].append({
                    "round": r,
                    "winner": winner,
                    "cursor": dataset_cursor,
                })
                self._save_state()
                
                # Delete checkpoint after successful completion
                self._delete_checkpoint(r)
                print(f"\n[Round {r}] Completed successfully")
                
            except Exception as e:
                print(f"\n[Error] Round {r} failed: {e}")
                traceback.print_exc()
                print(f"[Checkpoint] Progress saved. Resume by running the same command.")
                raise

        # Save complete history
        history_path = self.work_root / "auto_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        return {"history_path": str(history_path), "rounds": len(history), "history": history}