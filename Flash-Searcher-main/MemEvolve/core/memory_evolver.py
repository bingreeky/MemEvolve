#!/usr/bin/env python
# coding=utf-8

"""
Memory Evolution Engine
Core orchestrator for analyzing, generating, creating, and validating memory systems
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

from ..phases.memory_creator import MemorySystemCreator
from ..phases.phase_analyzer import PhaseAnalyzer
from ..phases.phase_generator import PhaseGenerator
from ..phases.phase_validator import PhaseValidator
from ..config import ANALYSIS_MAX_STEPS, CREATIVITY_INDEX

load_dotenv(override=True)


class MemoryEvolver:
    """
    Core memory evolution orchestrator

    Coordinates four phases:
    1. Analyze - analyze task trajectories with AnalysisAgent
    2. Generate - generate ONE new memory system configuration
    3. Create - create actual provider files for the generated system
    4. Validate - static checks and testing without automatic fixes
    
    """

    def __init__(self, work_dir: str, analysis_model_id: Optional[str] = None, gen_model_id: Optional[str] = None):
        """
        Initialize memory evolver

        Args:
            work_dir: Working directory for all outputs
            analysis_model_id: Model ID for analysis phase (optional, defaults to ANALYSIS_MODEL or DEFAULT_MODEL env var)
            gen_model_id: Model ID for generation phase (optional, defaults to GENERATION_MODEL or DEFAULT_MODEL env var)
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        if analysis_model_id is None:
            analysis_model_id = os.getenv("ANALYSIS_MODEL", os.getenv("DEFAULT_MODEL", "gpt-5"))
        self.analysis_model_id = analysis_model_id
        
        if gen_model_id is None:
            gen_model_id = os.getenv("GENERATION_MODEL", os.getenv("DEFAULT_MODEL", "gpt-5"))
        self.gen_model_id = gen_model_id
        
        # Initialize OpenAI client for generation
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        
        # Load or initialize state
        self.state_file = self.work_dir / "state.json"
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """
        Load state from file or create new
        
        Returns:
            State dictionary
        """
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {
            "analysis_model_id": self.analysis_model_id,
            "gen_model_id": self.gen_model_id,
            "created_at": datetime.now().isoformat(),
            "phases": {
                "analyze": {"completed": False},
                "generate": {"completed": False},
                "create": {"completed": False},
                "validate": {"completed": False}
            }
        }
    
    def _save_state(self):
        """Save current state to file"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2)
    
    def _get_next_generated_system_path(self) -> Path:
        """
        Get next available generated_system_N.json path
        
        Returns:
            Path to next available generated system file (generated_system_1.json, generated_system_2.json, etc.)
        """
        # Check for existing generated_system_*.json files
        existing_files = list(self.work_dir.glob("generated_system_*.json"))
        
        if not existing_files:
            # No files exist, start with generated_system_1.json
            return self.work_dir / "generated_system_1.json"
        
        # Find highest number
        numbers = []
        for f in existing_files:
            try:
                # Extract number from filename like "generated_system_2.json"
                stem = f.stem  # e.g., "generated_system_2"
                if stem.startswith("generated_system_"):
                    num_str = stem.split('_')[-1]  # Get the last part after underscore
                    if num_str:  # If there's a number
                        numbers.append(int(num_str))
            except ValueError:
                continue
        
        next_num = max(numbers) + 1 if numbers else 1
        return self.work_dir / f"generated_system_{next_num}.json"
    
    def _get_all_generated_system_paths(self) -> list[Path]:
        """
        Get all generated system configuration files
        
        Returns:
            List of paths to generated system files, sorted by number
        """
        # Get all generated_system_*.json files
        all_files = list(self.work_dir.glob("generated_system_*.json"))
        
        # Sort by number in filename
        def get_number(p: Path) -> int:
            try:
                stem = p.stem  # e.g., "generated_system_2"
                if stem.startswith("generated_system_"):
                    num_str = stem.split('_')[-1]
                    return int(num_str) if num_str else 0
            except ValueError:
                pass
            return 0
        
        all_files.sort(key=get_number)
        
        return all_files
    
    def analyze(self, task_logs_dir: str, default_provider: Optional[str] = "agent_kb", max_steps: int = ANALYSIS_MAX_STEPS) -> Dict:
        """
        Analyze task trajectories using AnalysisAgent
        
        Args:
            task_logs_dir: Path to directory containing task execution logs
            default_provider: Provider to use as template reference (required)
        
        Returns:
            Analysis result with report path
        """
        if not default_provider or default_provider.strip() == "":
            raise ValueError("default_provider is required and cannot be empty")
        
        print(f"[Analyze] Starting trajectory analysis")
        print(f"  Task logs: {task_logs_dir}")
        print(f"  Template provider: {default_provider}")
        
        analyzer = PhaseAnalyzer(
            analysis_model_id=self.analysis_model_id,
            task_logs_dir=task_logs_dir,
            default_provider=default_provider
        )
        
        # Run analysis
        result = analyzer.run_analysis(max_steps=max_steps)
        
        # Save analysis report
        report_path = self.work_dir / "analysis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Update state
        self.state["phases"]["analyze"] = {
            "completed": True,
            "output": str(report_path),
            "task_logs_dir": task_logs_dir,
            "default_provider": default_provider,
            "timestamp": datetime.now().isoformat()
        }
        self._save_state()
        
        print(f"[Analyze] Complete. Report saved to: {report_path}")
        
        return {
            "success": True,
            "report_path": str(report_path),
            "stats": result.get("stats", {})
        }
    
    def generate(self, creativity_index: float = CREATIVITY_INDEX) -> Dict:
        """
        Generate a single memory system configuration
        
        Args:
            creativity_index: Innovation level (0-1). 0=conservative, 1=highly creative
        
        Returns:
            Generation result with config path
        """
        # Validate creativity_index
        creativity_index = max(0.0, min(1.0, creativity_index))
        
        print(f"[Generate] Generating memory system")
        print(f"  Creativity index: {creativity_index:.2f}")
        
        # Load analysis report
        if not self.state["phases"]["analyze"]["completed"]:
            raise ValueError("Analysis phase not completed. Run analyze() first.")
        
        report_path = self.state["phases"]["analyze"]["output"]
        with open(report_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        # Get default_provider from analyze phase
        default_provider = self.state["phases"]["analyze"].get("default_provider")
        if not default_provider or default_provider.strip() == "":
            raise ValueError(
                "default_provider missing in analyze state — analyze() must run first; "
                "refusing to silently fall back to the initial provider"
            )
        print(f"  Using template provider: {default_provider}")
        
        generator = PhaseGenerator(
            openai_client=self.openai_client,
            model_id=self.gen_model_id,
            work_dir=self.work_dir,
            default_provider=default_provider
        )
        
        # Run generation
        result = generator.run_generation(
            analysis_data=analysis_data,
            creativity_index=creativity_index
        )
        
        if not result.get("success") or not result.get("config"):
            return {
                "success": False,
                "error": "Failed to generate system"
            }
        
        # Validate configuration before saving
        config = result["config"]
        config_updates = config.get("config_updates", {})
        
        if config_updates:
            print(f"\n[Validation] Validating generated configuration...")
            from MemEvolve.phases.memory_creator import MemorySystemCreator
            validation_result = MemorySystemCreator._validate_config_updates(config_updates)
            
            # Display warnings
            if validation_result["warnings"]:
                print(f"[WARNING] Configuration Warnings:")
                for warning in validation_result["warnings"]:
                    print(f"  - {warning}")
            
            # Display errors but don't fail (allow user to proceed)
            if not validation_result["success"]:
                print(f"[WARNING] Configuration Validation Issues Detected:")
                for error in validation_result["errors"]:
                    print(f"  - {error}")
                print(f"\n[NOTE] These issues will need to be fixed before creating the system.")
            else:
                print(f"[OK] Configuration validation passed!")
        
        # Save configuration - use incremental naming if file exists
        configs_path = self._get_next_generated_system_path()
        with open(configs_path, 'w', encoding='utf-8') as f:
            json.dump(result["config"], f, indent=2, ensure_ascii=False)
        
        # Update state
        self.state["phases"]["generate"] = {
            "completed": True,
            "output": str(configs_path),
            "creativity_index": creativity_index,
            "timestamp": datetime.now().isoformat()
        }
        self._save_state()
        
        print(f"\n[Generate] Complete. Generated 1 system")
        print(f"  Config saved to: {configs_path}")
        
        return {
            "success": True,
            "config_path": str(configs_path),
            "config": result["config"]
        }
    
    def create(self, config_path: Optional[str] = None) -> Dict:
        """
        Create memory system files using MemorySystemCreator
        
        Args:
            config_path: Optional path to generated system config. If not provided, uses state.
        
        Returns:
            Creation result with created system name
        """
        print(f"[Create] Creating memory system files")
        
        # Load generated config
        if config_path is None:
            if not self.state["phases"]["generate"]["completed"]:
                raise ValueError("Generation phase not completed. Run generate() first.")
            config_path = self.state["phases"]["generate"]["output"]
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Create system
        enum_value = config["memory_type_info"]["enum_value"]
        print(f"[Create] System: {enum_value}")
        
        result = MemorySystemCreator.create_memory_system(config, base_dir=".")
        
        created_systems = []
        failed_systems = []
        
        if result["success"]:
            created_systems.append(enum_value)
            print(f"  Success: {enum_value}")
        else:
            failed_systems.append({
                "system": enum_value,
                "error": result.get("error", "Unknown error")
            })
            print(f"  Failed: {result.get('error')}")
        
        # Save creation results (incremental update)
        results_path = self.work_dir / "created_system.json"
        
        # Load existing results if file exists
        existing_created = []
        existing_failed = []
        if results_path.exists():
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_created = existing_data.get("created", [])
                    existing_failed = existing_data.get("failed", [])
            except Exception:
                pass
        
        # Merge with new results (avoid duplicates)
        all_created = list(set(existing_created + created_systems))
        all_failed = existing_failed + [f for f in failed_systems if f not in existing_failed]
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                "created": all_created,
                "failed": all_failed
            }, f, indent=2)
        
        # Update state
        self.state["phases"]["create"] = {
            "completed": True,
            "output": str(results_path),
            "created_systems": created_systems,
            "failed_systems": failed_systems,
            "timestamp": datetime.now().isoformat()
        }
        self._save_state()
        
        print(f"[Create] Complete. Created {len(created_systems)} system")
        if failed_systems:
            print(f"  Warning: {len(failed_systems)} system failed")
        
        return {
            "success": True,
            "created": created_systems,
            "failed": failed_systems,
            "results_path": str(results_path)
        }
    
    def validate(self, dataset_name: Optional[str] = None, datasets_config: Optional[Dict[str, str]] = None, 
                 config_path: Optional[str] = None, created_systems: Optional[list] = None) -> Dict:
        """
        Validate created systems through static checks and testing
        
        Args:
            dataset_name: Dataset name for validation
            datasets_config: Dataset configuration
            config_path: Optional path to generated system config
            created_systems: Optional list of system names to validate

        Returns:
            Validation result with passed systems
        """
        print(f"[Validate] Starting validation")

        # Load created systems
        if created_systems is None:
            if not self.state["phases"]["create"]["completed"]:
                raise ValueError("Create phase not completed. Run create() first.")

            results_path = self.state["phases"]["create"]["output"]
            with open(results_path, 'r', encoding='utf-8') as f:
                creation_data = json.load(f)

            created_systems = creation_data["created"]

        if not created_systems:
            print("[Validate] No systems to validate")
            return {"success": True, "validated": [], "failed": [], "details": {}}

        # Create validator
        if config_path is None:
            config_path = self.state["phases"]["generate"]["output"]
        validator = PhaseValidator(
            work_dir=self.work_dir,
            configs_path=config_path,
            dataset_name=dataset_name,
            datasets_config=datasets_config
        )

        # Run validation
        result = validator.run_validation(created_systems=created_systems)

        # Save validation results (incremental update)
        validation_path = self.work_dir / "validated_systems.json"
        
        # Load existing results if file exists
        existing_validated = []
        existing_failed = []
        existing_details = {}
        if validation_path.exists():
            try:
                with open(validation_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_validated = existing_data.get("validated", [])
                    existing_failed = existing_data.get("failed", [])
                    existing_details = existing_data.get("details", {})
            except Exception:
                pass
        
        # Merge with new results (avoid duplicates)
        all_validated = list(set(existing_validated + result["validated"]))
        all_failed = existing_failed + [f for f in result["failed"] if f not in existing_failed]
        all_details = {**existing_details, **result["details"]}
        
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump({
                "validated": all_validated,
                "failed": all_failed,
                "details": all_details
            }, f, indent=2, ensure_ascii=False)

        # Update state
        self.state["phases"]["validate"] = {
            "completed": True,
            "output": str(validation_path),
            "validated_systems": result["validated"],
            "failed_systems": result["failed"],
            "timestamp": datetime.now().isoformat()
        }
        self._save_state()

        print(f"  Results saved to: {validation_path}")

        return {
            "success": True,
            "validated": result["validated"],
            "failed": result["failed"],
            "details": result["details"]
        }
