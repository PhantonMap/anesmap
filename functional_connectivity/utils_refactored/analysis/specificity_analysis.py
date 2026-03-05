
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..statistics.tests import get_top_indices_2d, perform_kruskal, check_is_unique
from ..plotting.specificity import plot_shared_matrix_heatmap


def _norm_pair(rc):
    """Normalize (i, j) to an undirected-edge representation to avoid treating (i, j) and (j, i) as different connections."""
    r, c = rc
    return (r, c) if r < c else (c, r)

def parse_threshold_mode(threshold_mode, matrix_size=None):

    if isinstance(threshold_mode, int):
        return threshold_mode
    
    threshold_mode_str = str(threshold_mode).lower()
    
    
    if threshold_mode_str.startswith('top_'):
        try:
            return int(threshold_mode_str.split('_')[1])
        except (IndexError, ValueError):
            raise ValueError(
                "When threshold_mode='top_N', you must provide the value of N."
            )
    

    return int(threshold_mode_str)


def create_and_save_shared_matrix(specificity_results, brain_regions, output_dir, 
                                    matrix_size, group_label='group', 
                                    brain_regions_abbr=None):

    print("\n" + "="*60)
    print(f"[ANALYSIS] Computing shared connections across {group_label}s...")
    print("="*60)
    
    # Count how many groups mark each connection as unique
    connection_count = {}
    for name, result in specificity_results.items():
        for coord in result['unique_coords']:
            if coord not in connection_count:
                connection_count[coord] = {'count': 0, 'groups': []}
            connection_count[coord]['count'] += 1
            connection_count[coord]['groups'].append(name)
    
    # Connections with count >= 2 are considered shared
    shared_coords = [coord for coord, info in connection_count.items() if info['count'] >= 2]
    
    print(f"[OK] Found {len(shared_coords)} shared connections (unique in ≥2 {group_label}s)")
    
    if len(shared_coords) == 0:
        print("⚠️ No shared connections found")
        return [], None
    
    # Create the shared matrix (2D matrix)
    shared_matrix = np.zeros((matrix_size, matrix_size))
    
    for row, col in shared_coords:
        shared_matrix[row, col] = 1
        shared_matrix[col, row] = 1  # symmetric
    
    # Save the shared matrix
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    df_shared_matrix = pd.DataFrame(
        shared_matrix,
        index=[brain_regions.get(i, f'R{i}') for i in range(matrix_size)],
        columns=[brain_regions.get(i, f'R{i}') for i in range(matrix_size)]
    )
    df_shared_matrix.to_csv(
        os.path.join(output_dir, "shared_connections_matrix.xlsx"),
        encoding='utf-8-sig'
    )
    
    # Save detailed list of shared connections
    shared_report_data = []
    for coord in shared_coords:
        row, col = coord
        info = connection_count[coord]
        shared_report_data.append({
            'Region_Row': brain_regions.get(row, f'R{row}'),
            'Region_Col': brain_regions.get(col, f'R{col}'),
            f'Num_{group_label.capitalize()}s': info['count'],
            f'{group_label.capitalize()}s': ', '.join([str(g).upper() for g in info['groups']])
        })
    
    df_shared_report = pd.DataFrame(shared_report_data)
    df_shared_report = df_shared_report.sort_values(by=f'Num_{group_label.capitalize()}s', ascending=False)
    df_shared_report.to_csv(
        os.path.join(output_dir, "shared_connections_detail.csv"),
        index=False, encoding='utf-8-sig'
    )
    
    # Plot heatmap of the shared matrix
    plot_shared_matrix_heatmap(
        shared_matrix, shared_coords, brain_regions,
        output_dir, brain_regions_abbr=brain_regions_abbr
    )
    
    print(f"[OK] Shared matrix and report saved to {output_dir}")
    
    return shared_coords, shared_matrix


def perform_specificity_test(current_data, other_data_dict, significant_indices, 
                               matrix_size, alpha=0.05, desc="Specificity test"):

    
    unique_coords = []
    is_unique_list = []
    stats_list = []
    
    upper_tri_indices = np.triu_indices(matrix_size, k=1)
    
    # Check input format: coordinate pairs or linear indices
    is_coord_pair = isinstance(significant_indices[0], tuple) if significant_indices else False
    
    for item in tqdm(significant_indices, desc=desc, leave=False):
        if is_coord_pair:
            # Input is a coordinate pair (row, col)
            row, col = item
            # Convert coordinates to the linear index of the flattened upper triangle
            # Find the position of this coordinate in the upper triangle
            idx = np.where((upper_tri_indices[0] == row) & (upper_tri_indices[1] == col))[0][0]
        else:
            print("Input is a linear index")
            idx = item
            row = upper_tri_indices[0][idx]
            col = upper_tri_indices[1][idx]
        
        # Values of this connection in the current group
        curr_vals = current_data[:, idx]
        
        # Values of this connection in other groups
        other_vals_list = [other_data_dict[key][:, idx] for key in other_data_dict.keys()]
        
        # Prepare data for Kruskal: current group first, others follow
        group_data = [curr_vals] + other_vals_list
        
        # Kruskal-Wallis test first
        kruskal_result = perform_kruskal(group_data, alpha=alpha)
        
        # Initialize statistics dictionary
        stats = {
            'kruskal_h': kruskal_result['stat'],
            'kruskal_p': kruskal_result['p_val'],
            'kruskal_sig': kruskal_result['significant'],
            'dunn_pvalues': [],
            'dunn_min_p': 1.0
        }
        
        unique_flag = False
        if kruskal_result['significant']:
            # If there is an overall difference, run post-hoc Dunn test to decide uniqueness
            # Current group is group 0; all groups are indexed as [0, 1, 2, ...]
            valid_groups = np.arange(len(group_data))
            dunn_result = check_is_unique(group_data, group_idx=0, valid_groups=valid_groups, alpha=alpha)
            unique_flag = dunn_result['is_unique']
            stats['dunn_pvalues'] = dunn_result['dunn_pvalues']
            stats['dunn_min_p'] = dunn_result['min_pvalue']
        
        is_unique_list.append(unique_flag)
        stats_list.append(stats)
        
        if unique_flag:
            unique_coords.append((row, col))
    
    return unique_coords, is_unique_list, stats_list


def generate_truly_unique_connections_table(specificity_results, brain_regions, 
                                             output_name, group_types):

    print("\n" + "="*60)
    print("[ANALYSIS] Generating truly unique connections table...")
    print("="*60)
    
    # 1. Collect unique connections from all groups
    all_unique_coords = {}
    for group_name in group_types:
        if group_name in specificity_results:
            all_unique_coords[group_name] = set(
                _norm_pair(coord) for coord in specificity_results[group_name]['unique_coords']
            )
    
    # 2. Find each group's "truly unique" connections (not shared with any other group)
    truly_unique_per_group = {}
    for group_name in group_types:
        if group_name not in all_unique_coords:
            continue
        
        current_unique = all_unique_coords[group_name]
        other_unique = set()
        
        for other_group in group_types:
            if other_group != group_name and other_group in all_unique_coords:
                other_unique.update(all_unique_coords[other_group])
        
        # Truly unique = current unique - union of all other groups' unique
        truly_unique_per_group[group_name] = current_unique - other_unique
        
        print(f"[OK] {group_name}: {len(truly_unique_per_group[group_name])} truly unique connections")
    
    # 3. Generate CSV table
    table_data = []
    
    for group_name in group_types:
        if group_name not in truly_unique_per_group:
            continue
        
        result = specificity_results[group_name]
        truly_unique_coords = truly_unique_per_group[group_name]
        
        if len(truly_unique_coords) == 0:
            continue
        
        # Get statistics
        significant_indices = result['significant_indices']
        stats_list = result['stats_list']
        
        # Map coordinates to statistics
        coord_to_stats = {}
        for k, coord in enumerate(significant_indices):
            norm_coord = _norm_pair(coord)
            coord_to_stats[norm_coord] = stats_list[k]
        
        # Create a row for each truly unique connection
        for coord in truly_unique_coords:
            if coord not in coord_to_stats:
                continue
            
            stats = coord_to_stats[coord]
            row, col = coord
            
            # Basic info
            row_data = {
                'Anesthetic': group_name.upper(),
                'Unique_FC': f"{brain_regions.get(col, f'R{col}')} - {brain_regions.get(row, f'R{row}')}",
                'Kruskal_Wallis_p': stats['kruskal_p']
            }
            
            # Add Dunn-test p-value columns (in the order of group_types)
            dunn_pvalues = stats.get('dunn_pvalues', [])
            
            # Other groups in order (excluding the current group)
            other_groups = [g for g in group_types if g != group_name]
            
            for i, other_group in enumerate(group_types):
                col_name = f'Dunn_p_vs_{other_group.upper()}'
                
                if other_group == group_name:
                    # Do not compare a group to itself
                    row_data[col_name] = np.nan
                else:
                    # Find the index of this group in dunn_pvalues
                    # dunn_pvalues order: current group (0) vs other groups (1, 2, ...)
                    # other_groups lists groups excluding current group in the original order
                    if other_group in other_groups:
                        dunn_idx = other_groups.index(other_group)
                        if dunn_idx < len(dunn_pvalues):
                            row_data[col_name] = dunn_pvalues[dunn_idx]
                        else:
                            row_data[col_name] = np.nan
                    else:
                        row_data[col_name] = np.nan
            
            table_data.append(row_data)
    
    # 4. Create DataFrame and save
    if len(table_data) == 0:
        print("⚠️ No truly unique connections found")
        return None
    
    df_truly_unique = pd.DataFrame(table_data)
    
    # Save Excel
    df_truly_unique.to_excel(output_name, index=False)
    print(f"[OK] Truly unique connections table saved to: {output_name}")
    print(f"[OK] Total truly unique connections: {len(df_truly_unique)}")
    
    return df_truly_unique
def analyze_cross_anesthetic_specificity(raw_data_store, importance_matrices_dict, brain_regions,
                                         base_output_dir, alpha=0.05, threshold_mode='top_10', 
                                         ):
    """
    Cross-anesthetic specificity analysis:
    For each anesthetic, perform statistical tests on its significant connections
    to determine whether they are anesthetic-specific.

    Parameters:
        raw_data_store: dict, {anes_type: np.array(n_samples, n_features)}, raw data (without SMOTE)
        importance_matrices_dict: dict, {anes_type: importance_matrix (n_regions x n_regions)}
        brain_regions: dict, mapping from brain-region index to region name
        base_output_dir: output directory
        alpha: significance level
        threshold_mode: str, filtering mode for selecting features ('top_20', 'top_50')

    Returns:
        specificity_results: dict,
            {anes_type: {'importance_matrix': ..., 'unique_coords': ..., 'shared_coords': ...}}
    """
    print("\n" + "="*60)
    print(f"[ANALYSIS] Cross-anesthetic specificity analysis ({threshold_mode} mode)...")
    print("="*60)
    
    specificity_results = {}
    anesthetic_types = list(raw_data_store.keys())
    
    for anes_type in anesthetic_types:
        print(f"\nAnalyzing specificity for {anes_type.upper()}...")
            
        importance_matrix = importance_matrices_dict[anes_type]
        matrix_size = importance_matrix.shape[0]
        
        # Use threshold_mode to select top_N connections from the importance matrix
        top_n = parse_threshold_mode(threshold_mode)
        significant_indices = get_top_indices_2d(importance_matrix, top_n=top_n)
        print(f"[INFO] Using {threshold_mode} filter: {len(significant_indices)} connections")
    
        if len(significant_indices) == 0:
            print(f"⚠️ No features meet the criteria for {anes_type}; skipping.")
            continue
        
        # Prepare the list of other anesthetics
        other_anes = [k for k in anesthetic_types if k != anes_type]
        
        # Build the data dict for other anesthetics
        other_data_dict = {oa: raw_data_store[oa] for oa in other_anes}
        
        # Use a unified specificity testing function
        print(f"Testing specificity for {len(significant_indices)} significant connections...")
        unique_coords, is_unique_list, stats_list = perform_specificity_test(
            current_data=raw_data_store[anes_type],
            other_data_dict=other_data_dict,
            significant_indices=significant_indices,
            matrix_size=matrix_size,
            alpha=alpha,
            desc="Specificity test"
        )
        
        print(f"[OK] {anes_type}: Unique connections = {len(unique_coords)}")
        
        # Save results
        specificity_results[anes_type] = {
            'importance_matrix': importance_matrix,
            'unique_coords': unique_coords,
            'significant_indices': significant_indices,
            'is_unique_list': is_unique_list,
            'stats_list': stats_list
        }
        
        
    threshold_mode = threshold_mode.replace("_", "")
    output_name = os.path.join(base_output_dir, f"Specificity_among_three_receptors-{threshold_mode}.xlsx")

    generate_truly_unique_connections_table(
        specificity_results, 
        brain_regions, 
        output_name, 
        anesthetic_types,
    )
    
    print("\n" + "="*60)
    print("[SUCCESS] Cross-anesthetic specificity analysis completed!")
    print("="*60 + "\n")
    
    return specificity_results




def analyze_receptor_specificity(specificity_results, raw_data_store, importance_matrices_dict, 
                                 brain_regions, base_output_dir, alpha=0.05, threshold_mode='top_20',
                                 brain_regions_abbr=None):

    print("\n" + "="*60)
    print("🧬 Receptor-group specificity analysis (2D heatmaps)...")
    print("="*60)
    
    receptor_groups = {
        "NMDA": ["ketamine", "n2o"],
        "GABA": ["iso", "propofol"],
        "a2": ["dex"]
    }
    
    receptor_output_dir = os.path.join(base_output_dir, "receptor_specificity_comparison")
    os.makedirs(receptor_output_dir, exist_ok=True)
    
    summary_data = []
    receptor_specificity_results = {}
    
    # ========== Part 1: anesthetic-level statistics (summary table only) ==========
    for receptor, drug_list in receptor_groups.items():
        available_drugs = [d for d in drug_list if d in specificity_results]
        if not available_drugs:
            continue
        
        print(f"\n{receptor} receptor ({', '.join([d.upper() for d in available_drugs])}):")
        
        for drug in available_drugs:
            data = specificity_results[drug]
            n_unique = len(data['unique_coords'])
            n_total = len(data['significant_indices'])
            
            unique_ratio = n_unique / n_total if n_total > 0 else 0
            
            print(f"  {drug.upper()}: specificity={n_unique}/{n_total} ({unique_ratio:.1%})")
            
            summary_data.append({
                'Receptor': receptor,
                'Drug': drug.upper(),
                'Unique_Connections': n_unique,
                'Total_Significant': n_total,
                'Specificity_Ratio': unique_ratio
            })
    
    # Save anesthetic-level summary table
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(
            os.path.join(receptor_output_dir, "anesthetic_level_summary.xlsx"),
            index=False, encoding='utf-8-sig'
        )
        print("[OK] Anesthetic-level summary saved")
    
    # ========== Part 2: receptor-level 2D cross-group specificity analysis ==========
    print("\n" + "="*60)
    print("[ANALYSIS] Receptor-level cross-group specificity analysis (2D)...")
    print("="*60)
    
    # Prepare receptor-level data by aggregating all anesthetics under the same receptor
    receptor_data_store = {}             # {receptor: np.array(n_samples_total, n_features)}
    receptor_importance_matrices = {}    # {receptor: mean_importance_matrix}
    receptor_significant_indices = {}    # {receptor: top-N indices from mean importance matrix}
    
    for receptor, drug_list in receptor_groups.items():
        available_drugs = [d for d in drug_list if d in raw_data_store]
        if not available_drugs:
            continue
        
        # Aggregate data: concatenate samples from all anesthetics
        all_samples = []
        for drug in available_drugs:
            all_samples.append(raw_data_store[drug])
        receptor_data_store[receptor] = np.vstack(all_samples)
        
        # Aggregate importance matrices: take the mean
        all_imp_matrices = [importance_matrices_dict[d] for d in available_drugs]
        receptor_importance_matrices[receptor] = np.mean(all_imp_matrices, axis=0)
        
        # ========== [Modified] select top_N from the mean importance matrix ==========
        top_n = parse_threshold_mode(threshold_mode)
        receptor_significant_indices[receptor] = get_top_indices_2d(
            receptor_importance_matrices[receptor], top_n=top_n
        )
        # ========================================================================
        
        print(f"[OK] {receptor}: {len(available_drugs)} drugs, "
              f"{receptor_data_store[receptor].shape[0]} samples, "
              f"{len(receptor_significant_indices[receptor])} candidate connections ({threshold_mode})")
    
    # Specificity analysis for each receptor
    if len(receptor_data_store) < 2:
        print("⚠️ Fewer than 2 receptors available; cross-receptor specificity analysis cannot be performed.")
    else:
        receptor_list = list(receptor_data_store.keys())
        
        for receptor in receptor_list:
            print(f"\nAnalyzing specificity for {receptor} receptor...")
            
            if receptor not in receptor_significant_indices or len(receptor_significant_indices[receptor]) == 0:
                print(f"⚠️ No significant features for {receptor}; skipping.")
                continue
            
            importance_matrix = receptor_importance_matrices[receptor]
            significant_indices = receptor_significant_indices[receptor]
            matrix_size = importance_matrix.shape[0]
            
            # Prepare other receptors
            other_receptors = [r for r in receptor_list if r != receptor]
            
            # Build data dict for other receptors
            other_data_dict = {r: receptor_data_store[r] for r in other_receptors}
            
            # Unified specificity test
            print(f"Testing specificity for {len(significant_indices)} significant connections...")
            unique_coords, is_unique_list, stats_list = perform_specificity_test(
                current_data=receptor_data_store[receptor],
                other_data_dict=other_data_dict,
                significant_indices=significant_indices,
                matrix_size=matrix_size,
                alpha=alpha,
                desc=f"{receptor} specificity test"
            )
            
            print(f"[OK] {receptor}: Unique connections = {len(unique_coords)}")
            
            # Save results
            receptor_specificity_results[receptor] = {
                'importance_matrix': importance_matrix,
                'unique_coords': unique_coords,
                'significant_indices': significant_indices,
                'is_unique_list': is_unique_list,
                'stats_list': stats_list
            }
            
            # Save detailed report directory
            output_dir = os.path.join(receptor_output_dir, f"receptor_{receptor}")
            os.makedirs(output_dir, exist_ok=True)
            

        
        # ========== After all receptors are analyzed: compute shared connections ==========
        first_matrix = list(receptor_importance_matrices.values())[0]
        matrix_size = first_matrix.shape[0]
        shared_output_dir_receptor = os.path.join(receptor_output_dir, "shared_connections")
        
        create_and_save_shared_matrix(
            receptor_specificity_results, brain_regions, shared_output_dir_receptor,
            matrix_size, group_label='receptor',
            brain_regions_abbr=brain_regions_abbr
        )
        
        # ========== Generate table for "truly unique" functional connections ==========
        generate_truly_unique_connections_table(
            receptor_specificity_results, 
            brain_regions, 
            receptor_output_dir, 
            receptor_list,
        )
    
    print("\n" + "="*60)
    print("[SUCCESS] Receptor specificity analysis completed (with 2D heatmaps)!")
    print("="*60 + "\n")



def analyze_within_receptor_comparison(raw_data_store, importance_matrices_dict, 
                                       brain_regions, base_output_dir, alpha=0.05, 
                                       threshold_mode='top_50'):

    print("\n" + "="*70)
    print("[ANALYSIS] Within-receptor binary comparison (GABA: iso vs propofol, NMDA: ketamine vs n2o)")
    print("="*70)
    
    # Define within-receptor pairs
    receptor_pairs = {
        "GABA": ["iso", "propofol"],
        "NMDA": ["ketamine", "n2o"]
    }
    
    all_results = {}
    
    for receptor_name, drug_list in receptor_pairs.items():
        # Check data availability
        available_drugs = [d for d in drug_list if d in raw_data_store and d in importance_matrices_dict]
        if len(available_drugs) != 2:
            print(f"⚠️ Incomplete data for {receptor_name} receptor "
                  f"(need 2 anesthetics, but found {len(available_drugs)}); skipping.")
            continue
        
        drug1, drug2 = available_drugs
        print(f"\n{'='*60}")
        print(f"[ANALYSIS] Analyzing {receptor_name} receptor: {drug1.upper()} vs {drug2.upper()}")
        print(f"{'='*60}")
        
        # Get matrix size
        imp_matrix1 = importance_matrices_dict[drug1]
        matrix_size = imp_matrix1.shape[0]
        top_n = parse_threshold_mode(threshold_mode)
        
        # Perform specificity analysis for each anesthetic
        for current_drug, other_drug in [(drug1, drug2), (drug2, drug1)]:
            print(f"\nAnalyzing specificity of {current_drug.upper()} relative to {other_drug.upper()}...")
            
            current_data = raw_data_store[current_drug]
            other_data = raw_data_store[other_drug]
            current_imp_matrix = importance_matrices_dict[current_drug]
            
            # Get candidate connections (using threshold_mode)
            significant_indices = get_top_indices_2d(current_imp_matrix, top_n=top_n)
            print(f"  - Number of candidate connections: {len(significant_indices)} ({threshold_mode})")
            
            # Build comparison dict (only one counterpart)
            other_data_dict = {other_drug: other_data}
            
            # Unified specificity test
            unique_coords, is_unique_list, stats_list = perform_specificity_test(
                current_data=current_data,
                other_data_dict=other_data_dict,
                significant_indices=significant_indices,
                matrix_size=matrix_size,
                alpha=alpha,
                desc=f"{current_drug} specificity test"
            )
            
            print(f"[OK] {current_drug.upper()}: Unique connections = {len(unique_coords)}")
            
            # Save results
            all_results[f"{receptor_name}_{current_drug}"] = {
                'importance_matrix': current_imp_matrix,
                'unique_coords': unique_coords,
                'significant_indices': significant_indices,
                'is_unique_list': is_unique_list,
                'stats_list': stats_list,
                'comparison_with': other_drug
            }
        
        # ========== Generate table for "truly unique" functional connections ==========
        within_receptor_specificity_results = {
            drug1: all_results[f"{receptor_name}_{drug1}"],
            drug2: all_results[f"{receptor_name}_{drug2}"]
        }

        # Output filename
        output_name = os.path.join(base_output_dir, f"{receptor_name}_unique_{threshold_mode}.xlsx")

        generate_truly_unique_connections_table(
            within_receptor_specificity_results, 
            brain_regions, 
            output_name, 
            [drug1, drug2],
        )
    
    print("\n" + "="*70)
    print("[SUCCESS] Within-receptor binary comparison analysis completed!")
    print("="*70 + "\n")
    
    return all_results

def analyze_gender_specificity(anesthetic_type, base_input_dir, brain_regions, brain_regions_abbr, 
                               base_output_dir, alpha=0.05, threshold_mode='top_20'):

    print("=" * 70)
    print(f"📊 Loading MALE and FEMALE {anesthetic_type.upper()} anesthetic data (threshold={threshold_mode})...")
    print("=" * 70)

    # Store gender-specific data
    gender_data_store = {}           # {gender: raw_data}
    gender_importance_matrices = {}  # {gender: importance_matrix}
    thresholdmode = threshold_mode.replace('_', '')

    # Create output directory based on anesthetic type
    dir_prefix = 'FC-ALL anesthetics gender-specific high-contribution specificity' if anesthetic_type == 'all' \
                 else 'FC-per anesthetic gender-specific high-contribution specificity'

    gender_specificity_dir = os.path.join(base_output_dir, dir_prefix, f'{thresholdmode}')
    os.makedirs(gender_specificity_dir, exist_ok=True)

    # Output file name depends on anesthetic_type
    xlsx_name = f'{thresholdmode}.xlsx' if anesthetic_type == 'all' else f'{anesthetic_type}.xlsx'
    gender_specificity_name = os.path.join(gender_specificity_dir, xlsx_name)

    for gender in ['male', 'female']:
        print(f"\nProcessing {gender.upper()} group...")

        if anesthetic_type == 'all':
            matrix_path = os.path.join(
                base_output_dir,
                "ALL anesthetics gender-specific high contribution",
                gender,
                f"{anesthetic_type}_rf_feature_importance_matrix.csv"
            )
        else:
            matrix_path = os.path.join(
                base_output_dir,
                "Per-anesthetic gender-specific high contribution",
                gender,
                anesthetic_type,
                f"{anesthetic_type}_rf_feature_importance_matrix.csv"
            )

        if not os.path.exists(matrix_path):
            print(f"⚠️ Importance matrix for {gender} {anesthetic_type.upper()} not found: {matrix_path}")
            continue

        importance_matrix = pd.read_csv(matrix_path, header=None).values
        gender_importance_matrices[gender] = importance_matrix
        print(f"✅ Loaded importance matrix for {gender}: {importance_matrix.shape}")

        # 2. Load raw data (including positive and negative samples)
        data_list = []

        # Load positive samples (anesthetic_{anesthetic_type})
        pos_data_folder = os.path.join(base_input_dir, f'anesthetic_{anesthetic_type}', gender)
        if os.path.exists(pos_data_folder):
            from ..data_processing.data_loader import read_excel_data
            from tqdm import tqdm
            files = sorted([f for f in os.listdir(pos_data_folder) if f.endswith('.xlsx')])
            for file in tqdm(files, desc=f"Loading {gender} {anesthetic_type.upper()} positive samples"):
                file_path = os.path.join(pos_data_folder, file)
                try:
                    features, _, _ = read_excel_data(file_path, 1)
                    data_list.append(features)
                except Exception:
                    continue
        pos_count = len(data_list)

        # Load negative samples (non_anesthetic_{anesthetic_type})
        neg_data_folder = os.path.join(base_input_dir, f'non_anesthetic_{anesthetic_type}', gender)
        if os.path.exists(neg_data_folder):
            files = sorted([f for f in os.listdir(neg_data_folder) if f.endswith('.xlsx')])
            for file in tqdm(files, desc=f"Loading {gender} {anesthetic_type.upper()} negative samples"):
                file_path = os.path.join(neg_data_folder, file)
                try:
                    features, _, _ = read_excel_data(file_path, 0)
                    data_list.append(features)
                except Exception:
                    continue
        neg_count = len(data_list) - pos_count

        if data_list:
            gender_data_store[gender] = np.array(data_list)
            print(f"✅ Loaded raw data for {gender} {anesthetic_type.upper()}: {len(data_list)} samples "
                  f"(positive={pos_count}, negative={neg_count})")
        else:
            print(f"⚠️ Failed to load data for {gender} {anesthetic_type.upper()}")

    # Check data completeness
    if len(gender_data_store) != 2 or len(gender_importance_matrices) != 2:
        print(f"\n❌ Incomplete gender data for {anesthetic_type.upper()}; specificity analysis cannot proceed.")
        return

    print("\n" + "=" * 70)
    print(f"✅ Data loaded. Starting analysis with threshold_mode = {threshold_mode}")
    print("=" * 70)

    print(f"\n📁 Output directory: {gender_specificity_dir}")

    # Perform specificity analysis for each gender
    gender_list = ['male', 'female']
    specificity_results = {}

    for gender in gender_list:
        # Check if this gender has already been analyzed
        output_dir = os.path.join(gender_specificity_dir, f"specificity_{gender}")
        report_file = os.path.join(output_dir, f"{gender}_connection_specificity_report.csv")

        if os.path.exists(report_file):
            print("\n" + "=" * 70)
            print(f"✅ Specificity analysis for {gender.upper()} already completed; skipping... (threshold={thresholdmode})")
            print(f"   Result file: {report_file}")
            print("=" * 70)
            continue

        print("\n" + "=" * 70)
        print(f"🔬 Analyzing specificity for {gender.upper()}... (threshold={thresholdmode})")
        print("=" * 70)

        current_data = gender_data_store[gender]
        other_gender = 'female' if gender == 'male' else 'male'
        other_data = gender_data_store[other_gender]

        importance_matrix = gender_importance_matrices[gender]
        matrix_size = importance_matrix.shape[0]

        # Get significant connection indices (using threshold_mode)
        top_n = parse_threshold_mode(threshold_mode, matrix_size=matrix_size)
        significant_indices = get_top_indices_2d(importance_matrix, top_n=top_n)
        print(f"📌 Filtered by {threshold_mode}: {len(significant_indices)} connections (top_n={top_n})")

        # Build the comparison data dict (only one comparator gender)
        other_data_dict = {other_gender: other_data}

        # Use the unified specificity testing function
        print(f"Testing specificity for {len(significant_indices)} significant connections...")
        unique_coords, is_unique_list, stats_list = perform_specificity_test(
            current_data=current_data,
            other_data_dict=other_data_dict,
            significant_indices=significant_indices,
            matrix_size=matrix_size,
            alpha=alpha,
            desc=f"{gender} specificity test"
        )

        # Compute shared_coords (significant but not specific connections)
        shared_coords = []
        for i, is_unique in enumerate(is_unique_list):
            if not is_unique:
                row, col = significant_indices[i]
                shared_coords.append((row, col))

        print(f"✅ {gender.upper()}: specific connections = {len(unique_coords)}, shared connections = {len(shared_coords)}")

        # Save results
        specificity_results[gender] = {
            'importance_matrix': importance_matrix,
            'unique_coords': unique_coords,
            'shared_coords': shared_coords,
            'significant_indices': significant_indices,
            'is_unique_list': is_unique_list,
            'stats_list': stats_list
        }

    # Generate a statistical test table for "truly unique" functional connections
    generate_truly_unique_connections_table(
        specificity_results,
        brain_regions,
        gender_specificity_name,
        gender_list
    )

    print(f"\n✅ {anesthetic_type.upper()} gender specificity analysis completed (threshold={thresholdmode})!")
    print("=" * 70)

