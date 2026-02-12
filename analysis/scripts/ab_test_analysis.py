#!/usr/bin/env python3
"""
A/B Test Statistical Analysis
Performs statistical significance tests on control vs treatment groups
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not available, using basic statistics only", file=sys.stderr)

import numpy as np


def calculate_metrics(data: List[Dict]) -> Dict:
    """Calculate metrics from data points"""
    if not data:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'count': 0,
            'min': 0.0,
            'max': 0.0
        }
    
    values = [d.get('value', 0) for d in data if 'value' in d]
    
    if not values:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'count': 0,
            'min': 0.0,
            'max': 0.0
        }
    
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'count': len(values),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }


def t_test(control: List[float], treatment: List[float]) -> Dict:
    """Perform independent samples t-test"""
    if not HAS_SCIPY:
        # Basic comparison without significance test
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        diff = treatment_mean - control_mean
        percent_change = (diff / control_mean * 100) if control_mean != 0 else 0.0
        
        return {
            'control_mean': float(control_mean),
            'treatment_mean': float(treatment_mean),
            'difference': float(diff),
            'percent_change': float(percent_change),
            'p_value': None,
            'significant': None,
            'note': 'scipy not available for significance testing'
        }
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(treatment, control)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(control) - 1) * np.var(control) + (len(treatment) - 1) * np.var(treatment)) /
        (len(control) + len(treatment) - 2)
    )
    cohens_d = (np.mean(treatment) - np.mean(control)) / pooled_std if pooled_std > 0 else 0.0
    
    control_mean = np.mean(control)
    treatment_mean = np.mean(treatment)
    diff = treatment_mean - control_mean
    percent_change = (diff / control_mean * 100) if control_mean != 0 else 0.0
    
    return {
        'control_mean': float(control_mean),
        'treatment_mean': float(treatment_mean),
        'difference': float(diff),
        'percent_change': float(percent_change),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significant': p_value < 0.05,
        'confidence_level': '95%'
    }


def analyze_ab_test(control_data: Dict, treatment_data: Dict) -> Dict:
    """Analyze A/B test results"""
    # Extract metrics
    control_metrics = {}
    treatment_metrics = {}
    
    # Time to completion
    if 'time_to_completion' in control_data and 'time_to_completion' in treatment_data:
        control_times = [d.get('value', 0) for d in control_data['time_to_completion']]
        treatment_times = [d.get('value', 0) for d in treatment_data['time_to_completion']]
        time_analysis = t_test(control_times, treatment_times)
        control_metrics['time_to_completion'] = calculate_metrics(control_data['time_to_completion'])
        treatment_metrics['time_to_completion'] = calculate_metrics(treatment_data['time_to_completion'])
    else:
        time_analysis = None
    
    # Context Precision
    if 'context_precision' in control_data and 'context_precision' in treatment_data:
        control_cp = [d.get('value', 0) for d in control_data['context_precision']]
        treatment_cp = [d.get('value', 0) for d in treatment_data['context_precision']]
        cp_analysis = t_test(control_cp, treatment_cp)
        control_metrics['context_precision'] = calculate_metrics(control_data['context_precision'])
        treatment_metrics['context_precision'] = calculate_metrics(treatment_data['context_precision'])
    else:
        cp_analysis = None
    
    # Success rate
    if 'success_rate' in control_data and 'success_rate' in treatment_data:
        control_success = control_data['success_rate'].get('value', 0)
        treatment_success = treatment_data['success_rate'].get('value', 0)
        success_improvement = treatment_success - control_success
        success_percent_change = (success_improvement / control_success * 100) if control_success > 0 else 0.0
    else:
        control_success = None
        treatment_success = None
        success_improvement = None
        success_percent_change = None
    
    # Overall conclusion
    significant_metrics = []
    if time_analysis and time_analysis.get('significant'):
        significant_metrics.append('time_to_completion')
    if cp_analysis and cp_analysis.get('significant'):
        significant_metrics.append('context_precision')
    
    conclusion = 'Statistically significant' if significant_metrics else 'Not statistically significant'
    
    return {
        'control': control_metrics,
        'treatment': treatment_metrics,
        'time_to_completion': time_analysis,
        'context_precision': cp_analysis,
        'success_rate': {
            'control': control_success,
            'treatment': treatment_success,
            'improvement': success_improvement,
            'percent_change': success_percent_change
        },
        'improvement': {
            'time_to_completion_percent': time_analysis['percent_change'] if time_analysis else None,
            'context_precision_percent': cp_analysis['percent_change'] if cp_analysis else None,
            'success_rate_percent': success_percent_change
        },
        'significant_metrics': significant_metrics,
        'conclusion': conclusion
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze A/B test results')
    parser.add_argument('--input', type=str, default=None,
                       help='Input JSON file with control and treatment data (default: stdin)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (default: stdout)')
    
    args = parser.parse_args()
    
    # Load data
    if args.input:
        with open(args.input, 'r') as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)
    
    # Extract control and treatment
    control_data = data.get('control', {})
    treatment_data = data.get('treatment', {})
    
    if not control_data or not treatment_data:
        print("[ERROR] Both 'control' and 'treatment' data required", file=sys.stderr)
        sys.exit(1)
    
    # Analyze
    result = analyze_ab_test(control_data, treatment_data)
    
    # Output
    output_json = json.dumps(result, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
    else:
        print(output_json)


if __name__ == '__main__':
    main()

