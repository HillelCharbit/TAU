"""
Comprehensive QA test for TAU Clustering weighted/unweighted flag combinations.

Tests all 18 combinations of:
- Actual file format: adjlist (unweighted) vs edgelist (weighted)
- Config is_weighted: True, False, None
- TAUClustering is_weighted: True, False, None

All combinations should now work after the auto-detection fix.
"""

import sys
# Add local source to path BEFORE any other imports
sys.path.insert(0, '/home/espl/TAU/src')

import networkx as nx
from tau_community_detection import TauClustering, TauConfig
import igraph as ig
from itertools import product

# Test parameters
POPULATION = 10
GENERATIONS = 2
RANDOM_SEED = 42

# Graph paths
WEIGHTED_PATH = '/home/espl/TAU/tests/S_cerevisiae.graph'        # edgelist with weights
UNWEIGHTED_PATH = '/home/espl/TAU/tests/S_cerevisiae_unweighted.graph'  # adjlist


def run_tau_test(graph_path, config_is_weighted, clustering_is_weighted):
    """Run a single TAU clustering test."""
    result = {
        'success': False,
        'error': None,
        'modularity': None,
        'communities': None,
        'warning': None
    }
    
    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Build config
            config_kwargs = {'random_seed': RANDOM_SEED, 'stopping_generations': GENERATIONS}
            if config_is_weighted is not None:
                config_kwargs['is_weighted'] = config_is_weighted
            config = TauConfig(**config_kwargs)
            
            # Build TAUClustering
            tau_kwargs = {
                'population_size': POPULATION,
                'max_generations': GENERATIONS,
                'config': config,
            }
            if clustering_is_weighted is not None:
                tau_kwargs['is_weighted'] = clustering_is_weighted
            
            tau = TauClustering(graph_path, **tau_kwargs)
            membership, modularity_history = tau.run()
            
            result['success'] = True
            result['modularity'] = modularity_history[-1] if modularity_history else float('nan')
            result['communities'] = len(set(membership))
            
            # Capture any warnings
            if w:
                result['warning'] = str(w[-1].message)
                
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        if len(error_msg) > 150:
            error_msg = error_msg[:150] + "..."
        result['error'] = error_msg
    
    return result


def val_str(v):
    if v is None:
        return "None"
    return str(v)


def main():
    print("=" * 80)
    print("TAU Clustering - Full QA Test (All 18 Combinations)")
    print("=" * 80)
    print()
    
    # Define all test combinations
    file_types = [
        ('adjlist', UNWEIGHTED_PATH),
        ('edgelist', WEIGHTED_PATH)
    ]
    flag_values = [True, False, None]
    
    results = []
    
    for (file_type, file_path), config_val, clustering_val in product(file_types, flag_values, flag_values):
        case_num = len(results) + 1
        
        print(f"Case {case_num:2d}: File={file_type:<8} | Config={val_str(config_val):<5} | Clust={val_str(clustering_val):<5}")
        sys.stdout.flush()
        
        result = run_tau_test(file_path, config_val, clustering_val)
        
        if result['success']:
            warn_str = " ⚠️" if result['warning'] else ""
            print(f"         ✅ PASS | mod={result['modularity']:.4f} | comm={result['communities']}{warn_str}")
        else:
            print(f"         ❌ FAIL | {result['error']}")
        
        if result['warning']:
            print(f"         Warning: {result['warning'][:60]}...")
        
        sys.stdout.flush()
        
        results.append({
            'case': case_num,
            'file_type': file_type,
            'config': config_val,
            'clustering': clustering_val,
            **result
        })
        print()
    
    # Summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    print(f"{'#':<3} {'File':<8} {'Config':<6} {'Clust':<6} {'Status':<8} {'Mod':<8} {'Comm':<5} {'Warn':<4}")
    print("-" * 65)
    
    for r in results:
        status = "PASS" if r['success'] else "FAIL"
        mod_str = f"{r['modularity']:.4f}" if r['modularity'] is not None else "N/A"
        comm_str = str(r['communities']) if r['communities'] is not None else "N/A"
        warn_str = "Yes" if r['warning'] else ""
        
        print(f"{r['case']:<3} {r['file_type']:<8} {val_str(r['config']):<6} {val_str(r['clustering']):<6} {status:<8} {mod_str:<8} {comm_str:<5} {warn_str:<4}")
    
    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    passed = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    warned = sum(1 for r in results if r['warning'])
    
    print(f"Passed:   {passed}/18")
    print(f"Failed:   {failed}/18")
    print(f"Warnings: {warned}/18")
    print()
    
    if failed > 0:
        print("Failed cases:")
        for r in results:
            if not r['success']:
                print(f"  Case {r['case']}: {r['file_type']} + config={val_str(r['config'])} + clust={val_str(r['clustering'])}")
                print(f"    Error: {r['error']}")
        print()
    
    if warned > 0:
        print("Cases with warnings (conflict between config and clustering param):")
        for r in results:
            if r['warning']:
                print(f"  Case {r['case']}: config={val_str(r['config'])} vs clust={val_str(r['clustering'])}")
        print()
    
    if passed == 18:
        print("🎉 All 18 combinations passed! Auto-detection fix is working correctly.")
    else:
        print(f"⚠️  {failed} combinations still failing - investigate errors above.")


if __name__ == "__main__":
    main()