"""
SSM Framework - Drift Detection Benchmark
=========================================
Compares SSM Agent vs Baseline (Vanilla Gemini) on constraint adherence
across 30 prompts with embedded "noise" to test drift resilience.
"""

import os
import sys
import json
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import create_agent
from eval.curated_benchmark import BENCHMARK_MODES, estimate_cost, MT_BENCH_PROMPTS, CAB_PROMPTS, HYBRID_PROMPTS

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("⚠️  GEMINI_API_KEY not found in .env")

# =============================================================================
# DEFINE THE RULES (The "Anchor")
# =============================================================================

RULES = {
    "language": "Java",
    "framework": "TensorFlow",
    "accelerator": "NVIDIA GPU",
    "tone": "technical",
    "constraint": "Real-time processing"
}

# =============================================================================
# DEFINE THE 30 PROMPTS - Three Phases
# =============================================================================

PROMPTS = [
    # Phase 1: Project Setup (Prompts 1-5) - Establish constraints
    ("I'm building a fish detection system. Use Java, TensorFlow for ML, NVIDIA GPU acceleration, keep tone technical, and ensure real-time processing.", "setup"),
    ("What is the best approach for real-time fish detection?", "setup"),
    ("How should I structure the Java backend for this?", "setup"),
    ("What TensorFlow Java API should I use?", "setup"),
    ("Explain GPU optimization for real-time video streams.", "setup"),
    
    # Phase 2: The Drift Storm - Noise (Prompts 6-25) - Off-topic to test resilience
    ("What is the best recipe for making pizza?", "drift"),
    ("Who won the FIFA World Cup in 1998?", "drift"),
    ("Tell me a story about a space explorer named Captain Orion.", "drift"),
    ("What are the ingredients in a Caesar salad?", "drift"),
    ("Explain the plot of the movie Interstellar.", "drift"),
    ("How do I grow tomatoes in my backyard?", "drift"),
    ("What's the distance between Earth and the Sun?", "drift"),
    ("Who painted the Mona Lisa?", "drift"),
    ("Give me 5 tips for better sleep.", "drift"),
    ("What's the history of the Great Wall of China?", "drift"),
    ("How do I fix a flat tire on a bicycle?", "drift"),
    ("What's the capital of Australia?", "drift"),
    ("Tell me a joke about a programmer and a rubber duck.", "drift"),
    ("What are the best tourist spots in Tokyo?", "drift"),
    ("How do you brew a perfect cup of coffee?", "drift"),
    ("What are the rules of American football?", "drift"),
    ("Tell me about the history of the Roman Empire.", "drift"),
    ("How do I make homemade ice cream?", "drift"),
    ("What's the difference between alligators and crocodiles?", "drift"),
    ("Explain the concept of quantum computing.", "drift"),
    
    # Phase 3: Final Check (Prompts 26-30) - Return to core task
    ("Write a function to detect fish in a real-time stream.", "final"),
    ("How should I handle database connections in Java?", "final"),
    ("Design a class for real-time frame capture from GPU.", "final"),
    ("Summarize all our project requirements so far.", "final"),
    ("FINAL TASK: Create a Java class called 'FishDetectionEngine' with real-time GPU processing.", "final"),
]


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_response(text, phase):
    """
    Fair and Hard: Only grade on technical phases.
    
    - During DRIFT PHASE (6-25): Return 10 (we don't expect Java/GPU discussion during pizza talk)
    - During SETUP/FINAL (1-5, 26-30): Return score based on technical keywords
    
    This reveals if the agent REMEMBERS constraints after noise injection.
    """
    # During the "Drift Storm", don't grade on technical keywords
    # Just give it a 10 if it responds (handling off-topic gracefully)
    if phase == "drift":
        return 10
    
    # For setup and final phases, grade on constraint adherence
    score = 0
    text_lower = text.lower()
    
    # Rule 1: Language (Java)
    if "java" in text_lower or "public class" in text_lower or "class " in text_lower:
        score += 2
    
    # Rule 2: Framework (TensorFlow)
    if "tensorflow" in text_lower or "tf." in text_lower:
        score += 2
    
    # Rule 3: Accelerator (GPU/CUDA)
    if "gpu" in text_lower or "cuda" in text_lower or "nvidia" in text_lower:
        score += 2
    
    # Rule 4: Real-time constraint mentioned
    if "real-time" in text_lower or "realtime" in text_lower or "latency" in text_lower:
        score += 2
    
    # Rule 5: Tone (Technical - avoid casual language)
    casual_words = ["hey", "cool", "sure", "happy", "feel free"]
    if not any(word in text_lower for word in casual_words):
        score += 2
    
    return min(score, 10)  # Cap at 10


# =============================================================================
# BENCHMARK EXECUTION
# =============================================================================

def run_benchmark(mode="original", custom_prompts=None):
    """
    Run benchmark with selected mode.
    
    Args:
        mode: "original" (30-turn), "mt-bench", "cab", "hybrid", or "custom"
        custom_prompts: List of (prompt, phase) tuples for custom mode
    
    Returns:
        DataFrame with results
    """
    
    # Select prompt set based on mode
    if mode == "mt-bench":
        prompts_to_use = MT_BENCH_PROMPTS
        mode_name = "MT-Bench Subset (10 prompts)"
    elif mode == "cab":
        prompts_to_use = CAB_PROMPTS
        mode_name = "CodeAssistBench Subset (8 prompts)"
    elif mode == "hybrid":
        prompts_to_use = HYBRID_PROMPTS
        mode_name = "Hybrid Benchmark (15 prompts)"
    elif mode == "custom" and custom_prompts:
        prompts_to_use = custom_prompts
        mode_name = "Custom Benchmark"
    else:
        prompts_to_use = PROMPTS
        mode_name = "Original 30-Turn Benchmark (Fish Detection)"
    
    print("\n" + "="*70)
    print(f"  FAIR BUT HARD BENCHMARK - {mode_name.upper()}")
    print("="*70)
    print(f"  Rules Anchor: {RULES}")
    print(f"  \n  ⚖️  SETUP:")
    print(f"  - Baseline: Gemini with NATIVE CHAT MEMORY (one continuous session)")
    print(f"  - SSM Agent: With constraint anchoring + state management")
    print(f"  - Benchmark Mode: {mode_name}")
    print(f"  - Total Prompts: {len(prompts_to_use)}")
    
    # Show cost estimate
    if mode in BENCHMARK_MODES:
        cost_info = estimate_cost(mode)
        if cost_info:
            print(f"  \n  💰 ESTIMATED COST: ${cost_info['estimated_cost']:.2f}")
            print(f"     (Savings: ${cost_info['savings_vs_original']:.2f} vs original)")
    
    print(f"  \n  SCORING LOGIC:")
    print(f"  - Technical Phases: Grade on technical keywords")
    print(f"  - Drift Phase: Always score 10 (not grading on tech terms)")
    print(f"  - Final Phase: Grade on technical keywords ← CRITICAL TEST")
    print("="*70 + "\n")
    
    try:
        ssm_agent = create_agent()
        print("✅ SSM Agent initialized (with constraint anchoring)")
    except Exception as e:
        print(f"❌ SSM Agent failed: {e}")
        return None
    
    try:
        baseline_model = genai.GenerativeModel('gemini-2.5-flash')
        # THIS IS THE KEY: One continuous chat session for baseline
        baseline_chat = baseline_model.start_chat(history=[])
        print("✅ Baseline Gemini initialized (with continuous chat memory)\n")
    except Exception as e:
        print(f"❌ Baseline chat initialization failed: {e}\n")
        return None
    
    results = []
    
    # Run benchmark with selected prompts
    for i, (prompt, phase) in enumerate(prompts_to_use):
        turn = i + 1
        print(f"[{turn:2d}/{len(prompts_to_use)}] {phase:8s} | {prompt[:40]:40s}", end=" ")
        
        # --- BASELINE: Using NATIVE CHAT MEMORY (continuous session) ---
        baseline_score = 0
        try:
            # send_message() maintains chat history automatically
            base_response = baseline_chat.send_message(prompt).text
            baseline_score = evaluate_response(base_response, phase)
            print(f"| Base: {baseline_score:2d}/10 ", end="")
        except Exception as e:
            baseline_score = 0
            errorDetail = str(e)[:80]
            print(f"| Base:  0/10 [ERROR: {errorDetail}] ", end="")
        
        # --- SSM AGENT: With constraint anchoring ---
        ssm_score = 0
        try:
            ssm_response = ssm_agent.process(prompt)
            response_text = ssm_response.get("response", "")
            ssm_score = evaluate_response(response_text, phase)
            print(f"| SSM: {ssm_score:2d}/10")
        except Exception as e:
            ssm_score = 0
            print(f"| SSM: ERROR")
        
        results.append({
            "Turn": turn,
            "Phase": phase,
            "Baseline": baseline_score,
            "SSM_Agent": ssm_score,
            "Difference": ssm_score - baseline_score
        })
        
        time.sleep(0.5)  # Rate limiting
    
    return pd.DataFrame(results)


# =============================================================================
# PLOTTING AND RESULTS EXPORT
# =============================================================================

def plot_results(df, mode="original"):
    """Create comparison plot."""
    
    plt.figure(figsize=(14, 7))
    
    # Plot both lines
    plt.plot(df['Turn'], df['Baseline'], 
             label='Baseline (Vanilla Gemini)', 
             color='red', marker='o', linestyle='--', linewidth=2, markersize=6)
    plt.plot(df['Turn'], df['SSM_Agent'], 
             label='SSM Agent (Constraint Anchoring)', 
             color='green', marker='s', linewidth=2.5, markersize=6)
    
    # Styling
    mode_name = BENCHMARK_MODES.get(mode, {}).get('name', 'Benchmark')
    plt.title(f'Constraint Adherence Analysis: {mode_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Turn Number', fontsize=12)
    plt.ylabel('Constraint Adherence Score (0-10)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    plt.ylim(-0.5, 10.5)
    
    # Highlight the Drift Storm phase if 30+ turns
    if len(df) >= 25:
        plt.axvspan(6, 25, color='gray', alpha=0.15, label='Noise Injection Phase')
        plt.text(15, 9.5, 'Drift Storm →', fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_filename = f"drift_analysis_graph_{mode}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved: {plot_path}")
    plt.close()
    
    return output_dir


def print_summary(df):
    """Print benchmark summary with extended metrics."""
    print("\n" + "="*70)
    print("  BENCHMARK SUMMARY - EXTENDED METRICS")
    print("="*70)
    print(f"Total Turns: {len(df)}\n")
    
    # === BASIC STATISTICS ===
    print("📊 BASIC STATISTICS")
    print("-" * 70)
    print(f"Baseline (Vanilla Gemini):")
    print(f"  - Average Score:       {df['Baseline'].mean():.2f}/10")
    print(f"  - Std Dev (Consistency): {df['Baseline'].std():.2f}")
    print(f"  - Median Score:        {df['Baseline'].median():.2f}/10")
    print(f"  - Max Score:           {df['Baseline'].max()}/10")
    print(f"  - Min Score:           {df['Baseline'].min()}/10")
    
    print(f"\nSSM Agent (Constraint Anchoring):")
    print(f"  - Average Score:       {df['SSM_Agent'].mean():.2f}/10")
    print(f"  - Std Dev (Consistency): {df['SSM_Agent'].std():.2f}")
    print(f"  - Median Score:        {df['SSM_Agent'].median():.2f}/10")
    print(f"  - Max Score:           {df['SSM_Agent'].max()}/10")
    print(f"  - Min Score:           {df['SSM_Agent'].min()}/10")
    
    # === PHASE BREAKDOWN ===
    print(f"\n📈 PERFORMANCE BY PHASE")
    print("-" * 70)
    setup_df = df[df['Phase'] == "setup"]
    drift_df = df[df['Phase'] == "drift"]
    final_df = df[df['Phase'] == "final"]
    
    for phase_name, phase_data in [("SETUP (1-5)", setup_df), ("DRIFT (6-25)", drift_df), ("FINAL (26-30)", final_df)]:
        print(f"\n{phase_name}:")
        print(f"  Baseline:  Avg {phase_data['Baseline'].mean():.2f}, Std {phase_data['Baseline'].std():.2f}")
        print(f"  SSM Agent: Avg {phase_data['SSM_Agent'].mean():.2f}, Std {phase_data['SSM_Agent'].std():.2f}")
    
    # === RECOVERY RATE ===
    print(f"\n🔄 RECOVERY RATE")
    print("-" * 70)
    setup_baseline = setup_df['Baseline'].mean()
    final_baseline = final_df['Baseline'].mean()
    baseline_recovery = ((final_baseline - drift_df['Baseline'].mean()) / (setup_baseline - drift_df['Baseline'].mean() + 0.01)) * 100 if setup_baseline > drift_df['Baseline'].mean() else 0
    
    setup_ssm = setup_df['SSM_Agent'].mean()
    final_ssm = final_df['SSM_Agent'].mean()
    ssm_recovery = ((final_ssm - drift_df['SSM_Agent'].mean()) / (setup_ssm - drift_df['SSM_Agent'].mean() + 0.01)) * 100 if setup_ssm > drift_df['SSM_Agent'].mean() else 0
    
    print(f"  Baseline Recovery Rate:  {baseline_recovery:.1f}%")
    print(f"  SSM Agent Recovery Rate: {ssm_recovery:.1f}%")
    
    # === ATTRIBUTE RETENTION RATIO (ARR) ===
    print(f"\n🎯 ATTRIBUTE RETENTION RATIO (ARR)")
    print("-" * 70)
    # ARR = (final_phase_avg / setup_phase_avg) * 100
    baseline_arr = (final_baseline / setup_baseline * 100) if setup_baseline > 0 else 0
    ssm_arr = (final_ssm / setup_ssm * 100) if setup_ssm > 0 else 0
    
    print(f"  Baseline ARR:  {baseline_arr:.1f}% (Final Avg / Setup Avg)")
    print(f"  SSM Agent ARR: {ssm_arr:.1f}% (Final Avg / Setup Avg)")
    
    # === PRECISION AT TURN 30 ===
    print(f"\n🎯 PRECISION AT TURN 30 (Final Turn)")
    print("-" * 70)
    turn_30 = df[df['Turn'] == 30]
    if len(turn_30) > 0:
        baseline_t30 = turn_30['Baseline'].values[0]
        ssm_t30 = turn_30['SSM_Agent'].values[0]
        print(f"  Baseline Score:  {baseline_t30}/10")
        print(f"  SSM Agent Score: {ssm_t30}/10")
        print(f"  SSM Advantage:   {ssm_t30 - baseline_t30:+.0f} points")
    
    # === CONSISTENCY SCORE ===
    print(f"\n📊 CONSISTENCY SCORE (Lower is Better - Less Variance)")
    print("-" * 70)
    drift_baseline_std = drift_df['Baseline'].std()
    setup_final_baseline_std = pd.concat([setup_df['Baseline'], final_df['Baseline']]).std()
    baseline_consistency = (drift_baseline_std / (setup_final_baseline_std + 0.01)) * 100
    
    drift_ssm_std = drift_df['SSM_Agent'].std()
    setup_final_ssm_std = pd.concat([setup_df['SSM_Agent'], final_df['SSM_Agent']]).std()
    ssm_consistency = (drift_ssm_std / (setup_final_ssm_std + 0.01)) * 100
    
    print(f"  Baseline: Drift Variance / Setup-Final Variance = {baseline_consistency:.1f}%")
    print(f"  SSM Agent: Drift Variance / Setup-Final Variance = {ssm_consistency:.1f}%")
    print(f"  (Lower = More consistent during drift)")
    
    # === WIN RATE ===
    print(f"\n🏆 WIN RATE (Turns Where SSM > Baseline)")
    print("-" * 70)
    win_count = (df['SSM_Agent'] > df['Baseline']).sum()
    win_rate = (win_count / len(df)) * 100
    print(f"  SSM wins: {win_count}/{len(df)} turns ({win_rate:.1f}%)")
    
    # === OVERALL IMPROVEMENT ===
    print(f"\n📈 OVERALL IMPROVEMENT")
    print("-" * 70)
    improvement = df['SSM_Agent'].mean() - df['Baseline'].mean()
    improvement_pct = (improvement / (df['Baseline'].mean() + 0.01)) * 100
    print(f"  Absolute: {improvement:+.2f} points")
    print(f"  Relative: {improvement_pct:+.1f}%")
    
    print("="*70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(benchmark_mode="original"):
    """
    Main benchmark execution.
    
    Args:
        benchmark_mode: "original", "mt-bench", "cab", "hybrid", or "custom"
    """
    
    try:
        # Display available modes
        print("\n" + "="*70)
        print("  AVAILABLE BENCHMARK MODES")
        print("="*70)
        for mode, info in BENCHMARK_MODES.items():
            cost_info = estimate_cost(mode)
            print(f"\n  {mode.upper():10s}: {info['name']}")
            print(f"  - Description: {info['description']}")
            print(f"  - Size: {cost_info['turns']} prompts")
            print(f"  - Est. Cost: ${cost_info['estimated_cost']:.2f} (Save ${cost_info['savings_vs_original']:.2f})")
        print("\n" + "="*70)
        
        print(f"\n🔄 Running benchmark with mode: {benchmark_mode}")
        print(f"🔄 Running {BENCHMARK_MODES.get(benchmark_mode, BENCHMARK_MODES['original'])['size']}-prompt benchmark...\n")
        
        # Run benchmark
        df = run_benchmark(mode=benchmark_mode)
        
        if df is None:
            print("❌ Benchmark failed to complete")
            return
        
        # Save results to CSV with mode suffix
        output_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(output_dir, exist_ok=True)
        
        csv_filename = f"benchmark_results_{benchmark_mode}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"\n✅ CSV saved: {csv_path}")
        
        # Create visualization
        plot_results(df, mode=benchmark_mode)
        
        # Print summary
        print_summary(df)
        
        print("\n" + "="*70)
        print("✅ BENCHMARK COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Allow selecting benchmark mode via command line
    import sys
    
    benchmark_mode = "original"  # default
    
    if len(sys.argv) > 1:
        benchmark_mode = sys.argv[1].lower()
    
    # Validate mode
    if benchmark_mode not in ["original", "mt-bench", "cab", "hybrid"]:
        print(f"❌ Invalid mode: {benchmark_mode}")
        print(f"Available modes: original, mt-bench, cab, hybrid")
        sys.exit(1)
    
    main(benchmark_mode=benchmark_mode)
