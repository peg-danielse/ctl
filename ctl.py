"""
CTL (Closing The Loop) - Main orchestration module.

This module orchestrates the continuous optimization workflow for Knative services
using baseline load testing, anomaly detection, and LLM-driven configuration generation.
"""

import time
import argparse
import os
from functools import wraps

from util.workflow import WorkflowOrchestrator


def report_time(func):
    """Decorator to report execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper


@report_time
def main(label=None, run_baseline=True, llm_provider='local', runtime_minutes=30):
    """
    Main entry point for the CTL system.
    
    This function orchestrates the complete workflow:
    1. Baseline load test to train isolation forest (optional)
    2. Continuous monitoring with anomaly detection
    3. LLM-driven configuration generation
    4. Configuration application and evolution
    
    Args:
        label (str): Experiment label for output organization
        run_baseline (bool): Whether to run baseline phase before adaptation
        llm_provider (str): LLM provider to use ('local' or 'openai')
        runtime_minutes (int): Runtime duration in minutes for load tests and monitoring
    """
    # Get label from environment or use default
    if label is None:
        label = os.getenv('CTL_LABEL', 'default')
    
    print(f"🚀 Starting CTL with label: '{label}'")
    print(f"📊 Baseline phase: {'Enabled' if run_baseline else 'Disabled'}")
    print(f"🤖 LLM Provider: {llm_provider.upper()}")
    print(f"⏱️  Runtime: {runtime_minutes} minutes")
    print("="*60)
    
    # Initialize the workflow orchestrator with LLM provider and runtime
    orchestrator = WorkflowOrchestrator(label=label, llm_provider=llm_provider, runtime_minutes=runtime_minutes)
    
    # Set up the experiment
    orchestrator.setup_experiment()
    
    if run_baseline:
        # Run baseline phase first
        print("🔄 Running baseline phase...")
        orchestrator.run_baseline_phase()
        print("✅ Baseline phase completed")
    
    # Run the adaptation phase
    print("🔄 Running adaptation phase...")
    orchestrator.run_adaptation_phase()
    print("✅ Adaptation phase completed")
    
    print("🎉 CTL workflow completed successfully!")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CTL (Closing The Loop) - Continuous optimization for Knative services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ctl.py                           # Use default label from CTL_LABEL env var
  python ctl.py --label my-experiment     # Use custom label
  python ctl.py --no-baseline             # Skip baseline phase
  python ctl.py --llm-provider openai     # Use OpenAI instead of local LLM
  python ctl.py --runtime 60              # Run for 60 minutes instead of default 30
  python ctl.py --label test --llm-provider openai --no-baseline --runtime 45 # Custom label with OpenAI and 45min runtime
  CTL_LABEL=production python ctl.py      # Use environment variable
        """
    )
    
    parser.add_argument(
        '--label', '-l',
        type=str,
        default=None,
        help='Experiment label for output organization (default: from CTL_LABEL env var or "default")'
    )
    
    parser.add_argument(
        '--no-baseline',
        action='store_true',
        help='Skip the baseline phase and run only adaptation phase'
    )
    
    parser.add_argument(
        '--baseline-only',
        action='store_true',
        help='Run only the baseline phase (for training anomaly detector)'
    )
    
    parser.add_argument(
        '--adaptation-only',
        action='store_true',
        help='Run only the adaptation phase (equivalent to --no-baseline)'
    )
    
    parser.add_argument(
        '--llm-provider',
        choices=['local', 'openai'],
        default='local',
        help='Choose LLM provider for configuration generation (default: local)'
    )
    
    parser.add_argument(
        '--runtime',
        type=int,
        default=30,
        help='Runtime duration in minutes for load tests and monitoring (default: 30)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine if baseline should run
    run_baseline = not (args.no_baseline or args.adaptation_only)
    
    # Handle baseline-only mode
    if args.baseline_only:
        print("🔄 Running baseline phase only...")
        label = args.label or os.getenv('CTL_LABEL', 'baseline')
        orchestrator = WorkflowOrchestrator(label=label, llm_provider=args.llm_provider, runtime_minutes=args.runtime)
        orchestrator.setup_experiment()
        orchestrator.run_baseline_phase()
        print("✅ Baseline phase completed")
    else:
        # Run the main optimization workflow
        main(label=args.label, run_baseline=run_baseline, llm_provider=args.llm_provider, runtime_minutes=args.runtime)