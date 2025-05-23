import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from typing import Optional, List
import numpy as np

def fix_column_list_to_str(x):
    # 1. If x is np.ndarray or list, flatten and convert to string
    if isinstance(x, np.ndarray):
        # flatten numpy array and join elements with " | "
        flat = x.flatten()
        return " | ".join(str(i) for i in flat)
    if isinstance(x, list):
        # recursively flatten nested lists
        def flatten_list(l):
            for el in l:
                if isinstance(el, list):
                    yield from flatten_list(el)
                else:
                    yield el
        flat = list(flatten_list(x))
        return " | ".join(str(i) for i in flat)

    # 2. If x is scalar (number, string, None, etc.)
    if pd.isna(x):
        return ""
    if not isinstance(x, str):
        return str(x)
    return x

def contains_any(val, filters):
    # Return True if filters is None or empty
    if not filters:
        return True
    val_str = str(val).lower() if val else ""
    # Return True if any filter string is substring of val_str (case-insensitive)
    return any(f.lower() in val_str for f in filters)

def load_and_filter_benchhub(
    language: str = "ko",
    split: str = "train",
    target_types: Optional[List[str]] = None,
    task_types: Optional[List[str]] = None,
    subject_types: Optional[List[str]] = None,
    chunk_size: int = 5000,
) -> Dataset:
    """
    Safely load BenchHub dataset from Hugging Face repo in chunks,
    preprocess via pandas, filter by given types, then convert to Dataset.
    """

    repo_id = "BenchHub/BenchHub-En" if language.lower() == "en" else "BenchHub/BenchHub-Ko"

    # Get total number of samples in the split
    ds_full = load_dataset(repo_id, split=split)
    total_samples = len(ds_full)

    filtered_chunks = []

    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        # Load chunk of data from the dataset
        ds_chunk = load_dataset(repo_id, split=f"{split}[{start_idx}:{end_idx}]")

        # Convert to pandas DataFrame
        df = pd.DataFrame(ds_chunk)

        # Normalize columns which may contain mixed list/string types
        for col in ['options', 'subject_type', 'target_type']:
            if col in df.columns:
                df[col] = df[col].apply(fix_column_list_to_str)

        # Also clean 'answer_str' column if present
        if 'answer_str' in df.columns:
            df['answer_str'] = df['answer_str'].apply(fix_column_list_to_str)
        
        # Apply filtering conditions on target_type, task_type, and subject_type
        filtered_df = df[
            df['target_type'].apply(lambda x: contains_any(x, target_types)) &
            df['task_type'].apply(lambda x: contains_any(x, task_types)) &
            df['subject_type'].apply(lambda x: contains_any(x, subject_types))
        ].reset_index(drop=True)
        
        # Convert filtered pandas DataFrame back to Dataset object
        filtered_ds_chunk = Dataset.from_pandas(filtered_df)
        filtered_chunks.append(filtered_ds_chunk)
    
    # Concatenate all filtered chunks into one Dataset
    from datasets import concatenate_datasets
    filtered_dataset = concatenate_datasets(filtered_chunks)

    return filtered_dataset

if __name__ == "__main__":
    filtered_ds = load_and_filter_benchhub(
        language="ko",
        split="train",
        target_types=["general"],
        task_types=["reasoning"],
        subject_types=["math", "history"],
        chunk_size=5000,
    )

    print(f"Total filtered samples: {len(filtered_ds)}")
    print(filtered_ds[0])
