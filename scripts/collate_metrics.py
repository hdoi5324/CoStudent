import os
import re
import argparse

from collections import defaultdict
from glob import glob
import ast
import json
import pandas as pd
from datetime import datetime



def main(args):
    base_dir = args.output_dir
    grouped_files = defaultdict(list)
    
    file_paths = glob(os.path.join(base_dir, '*/metrics.json'))
    # Walk through the directory tree
    for file_path in file_paths:
        path, file = os.path.split(file_path)
        if file == 'metrics.json':
            path_split = os.path.split(path)[1].split('_')
            path_split = path_split[:-1] if path_split[-1].isdigit() else path_split
            prefix = '_'.join(path_split)
            grouped_files[prefix].append(file_path)
    
    # Print the grouped file paths
    data = []
    for prefix, paths in grouped_files.items():
        print(f"Group: {prefix}")
        for file_path in paths:
            print(f"  {file_path}")
            seed = os.path.split(file_path)[0].split('_')[-1]
            seed = seed if seed.isdigit() else 999999
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:  # skip empty lines
                        try:
                            line_cleaned = line.replace('NaN', 'null')
                            results = json.loads(line_cleaned)
                            results['seed'] = seed
                            results['prefix'] = prefix
                            data.append(results)
                        except (ValueError, SyntaxError) as e:
                            print(f"Error parsing line: {line}\n{e}")
    
    data = [line for line in data if 'bbox/AP50' in line]
    if len(data) == 0:
        print(f"No metrics found in {base_dir}")
        return None
    df = pd.DataFrame(data)

    grouped = df.groupby(['prefix', 'seed'])
    #max_ap50 = grouped['bbox/AP50'].max().reset_index().rename(columns={'bbox/AP50': 'max_AP50'})
    
    max_ap50_rows = grouped.apply(lambda g: g[g['bbox/AP50'] == g['bbox/AP50'].max()]).reset_index(drop=True)
    max_ap50_rows = max_ap50_rows[['prefix', 'seed', 'bbox/AP50', 'iteration']].rename(columns={'bbox/AP50': 'max_AP50', 'iteration': 'iteration_at_max_AP50'})

    #last_ap50 = grouped.apply(lambda g: g[g['iteration'] == g['iteration'].max()][['prefix', 'seed', 'bbox/AP50']])
    #last_ap50 = last_ap50.reset_index(drop=True).rename(columns={'bbox/AP50': 'last_AP50'})
    
    last_ap50 = (
        df.loc[df.groupby(['prefix', 'seed'])['iteration'].idxmax()][['prefix', 'seed', 'bbox/AP50', 'iteration']]
        .rename(columns={'bbox/AP50': 'last_AP50'})
        .rename(columns={'iteration': 'last_iteration'})
        .reset_index(drop=True)
    )

    
    result = pd.merge(max_ap50_rows, last_ap50, on=['prefix', 'seed'])
    

    
    summary = result.groupby('prefix').agg(
        avg_max_AP50=('max_AP50', 'mean'),
        std_max_AP50=('max_AP50', 'std'),
        avg_last_AP50=('last_AP50', 'mean'),
        std_last_AP50=('last_AP50', 'std'),
        count=('prefix', 'size')
    ).reset_index()
    summary_sorted = summary.sort_values(by='prefix').reset_index(drop=True)
    
    numeric_cols = result.select_dtypes(include='number').columns
    result[numeric_cols] = result[numeric_cols].round(1)
    print(result.to_string())    
    
    numeric_cols = summary_sorted.select_dtypes(include='number').columns
    summary_sorted[numeric_cols] = summary_sorted[numeric_cols].round(1)
    print(summary_sorted.to_string())
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    summary_sorted.to_csv(os.path.join(base_dir, f"collated_metrics_{timestamp}.txt"), sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that uses output_dir")
    parser.add_argument('--output_dir', type=str, help='Path to the output directory', default="outputs/")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    print(f"Output directory is: {output_dir}")    
    main(args)
