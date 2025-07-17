#!/usr/bin/env python3
"""
Results organization and sharing script for ICL project.
Usage: uv run python scripts/organize_results.py [command]
"""

import json
import shutil
import tarfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse


def load_config(config_path: Path) -> Dict:
    """Load experiment configuration."""
    with open(config_path) as f:
        return json.load(f)


def create_summary_report(results_dir: Path) -> Dict:
    """Create a summary report of all experiments."""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_experiments": 0,
        "by_task": {},
        "experiments": []
    }
    
    for task_dir in results_dir.iterdir():
        if not task_dir.is_dir():
            continue
            
        task_name = task_dir.name
        task_experiments = []
        
        for exp_dir in task_dir.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith("train_"):
                continue
                
            config_path = exp_dir / "config.json"
            if not config_path.exists():
                continue
                
            config = load_config(config_path)
            
            # Extract key metrics
            exp_info = {
                "id": exp_dir.name,
                "path": str(exp_dir.relative_to(results_dir)),
                "config": config,
                "has_checkpoints": (exp_dir / "checkpoints").exists(),
                "has_plots": bool(list(exp_dir.rglob("*.png"))),
                "plot_count": len(list(exp_dir.rglob("*.png"))),
                "log_exists": (exp_dir / "log.json").exists()
            }
            
            task_experiments.append(exp_info)
            summary["total_experiments"] += 1
            
        summary["by_task"][task_name] = {
            "count": len(task_experiments),
            "experiments": task_experiments
        }
    
    return summary


def create_archive(results_dir: Path, output_path: Path, 
                  include_checkpoints: bool = False) -> None:
    """Create compressed archive of results."""
    with tarfile.open(output_path, "w:gz") as tar:
        for task_dir in results_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            for exp_dir in task_dir.iterdir():
                if not exp_dir.is_dir() or not exp_dir.name.startswith("train_"):
                    continue
                    
                # Always include config, logs, and plots
                for pattern in ["config.json", "log.json", "*.png", "*.pkl"]:
                    for file in exp_dir.rglob(pattern):
                        tar.add(file, arcname=file.relative_to(results_dir))
                
                # Optionally include checkpoints
                if include_checkpoints:
                    checkpoint_dir = exp_dir / "checkpoints"
                    if checkpoint_dir.exists():
                        tar.add(checkpoint_dir, 
                               arcname=checkpoint_dir.relative_to(results_dir))


def clean_old_experiments(results_dir: Path, keep_recent: int = 5) -> None:
    """Keep only the most recent experiments per task."""
    for task_dir in results_dir.iterdir():
        if not task_dir.is_dir():
            continue
            
        experiments = [d for d in task_dir.iterdir() 
                      if d.is_dir() and d.name.startswith("train_")]
        
        # Sort by modification time
        experiments.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old experiments
        for exp_dir in experiments[keep_recent:]:
            print(f"Removing old experiment: {exp_dir}")
            shutil.rmtree(exp_dir)


def create_paper_figures(results_dir: Path, output_dir: Path) -> None:
    """Extract and organize key figures for papers/presentations."""
    output_dir.mkdir(exist_ok=True)
    
    # Collect all PNG files with meaningful names
    for task_dir in results_dir.iterdir():
        if not task_dir.is_dir():
            continue
            
        task_name = task_dir.name
        task_output = output_dir / task_name
        task_output.mkdir(exist_ok=True)
        
        # Copy standalone plots (not in experiment folders)
        for png_file in task_dir.glob("*.png"):
            shutil.copy2(png_file, task_output / png_file.name)
        
        # Copy representative plots from experiments
        for exp_dir in task_dir.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith("train_"):
                continue
                
            plots_dir = exp_dir / "plots"
            if plots_dir.exists():
                for png_file in plots_dir.glob("*.png"):
                    new_name = f"{exp_dir.name}_{png_file.name}"
                    shutil.copy2(png_file, task_output / new_name)


def main():
    parser = argparse.ArgumentParser(description="Organize ICL experiment results")
    parser.add_argument("command", choices=["summary", "archive", "clean", "figures"])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, help="Output path")
    parser.add_argument("--keep-recent", type=int, default=5, 
                       help="Number of recent experiments to keep")
    parser.add_argument("--include-checkpoints", action="store_true",
                       help="Include model checkpoints in archive")
    
    args = parser.parse_args()
    
    if args.command == "summary":
        summary = create_summary_report(args.results_dir)
        output_path = args.output or Path("results_summary.json")
        
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary report saved to {output_path}")
        
    elif args.command == "archive":
        output_path = args.output or Path(f"icl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz")
        create_archive(args.results_dir, output_path, args.include_checkpoints)
        print(f"Archive created: {output_path}")
        
    elif args.command == "clean":
        clean_old_experiments(args.results_dir, args.keep_recent)
        print(f"Cleaned old experiments, keeping {args.keep_recent} recent per task")
        
    elif args.command == "figures":
        output_path = args.output or Path("paper_figures")
        create_paper_figures(args.results_dir, output_path)
        print(f"Figures organized in {output_path}")


if __name__ == "__main__":
    main()