import os
import re
import argparse

from collections import defaultdict
from glob import glob
import ast
import json
import pandas as pd



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
            seed = os.path.split(path)[0].split('_')[-1]
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
    df = pd.DataFrame(data)
    print(df.to_string())

    grouped = df.groupby(['prefix', 'seed'])
    max_ap50 = grouped['bbox/AP50'].max().reset_index().rename(columns={'bbox/AP50': 'max_AP50'})
    last_ap50 = grouped.apply(lambda g: g[g['iteration'] == g['iteration'].max()][['prefix', 'seed', 'bbox/AP50']])
    last_ap50 = last_ap50.reset_index(drop=True).rename(columns={'bbox/AP50': 'last_AP50'})
    result = pd.merge(max_ap50, last_ap50, on=['prefix', 'seed'])
    
    numeric_cols = result.select_dtypes(include='number').columns
    result[numeric_cols] = result[numeric_cols].round(1)

    # Display the result
    print(result)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that uses output_dir")
    parser.add_argument('--output_dir', type=str, help='Path to the output directory', default="../CoStudent/outputs/loose")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    print(f"Output directory is: {output_dir}")    
    main(args)
