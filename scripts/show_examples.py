"""Print example data samples from CICIDS2017 and UNSW-NB15 in both raw and formatted forms."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_dataset, format_dataset

# ── CICIDS2017 ──
print("=" * 80)
print("CICIDS2017 EXAMPLES")
print("=" * 80)

df_cicids = load_dataset("cicids2017", "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/cicids2017/")
print(f"\nTotal samples: {len(df_cicids)}")
print(f"Columns: {list(df_cicids.columns[:20])}...")
print(f"\nAttack type distribution (top 10):")
print(df_cicids['attack_type'].value_counts().head(10))

# Show 1 normal + 1 attack raw
print("\n--- RAW Normal Sample ---")
normal = df_cicids[df_cicids['is_attack'] == 0].iloc[0]
for col in ['protocol', 'source_port', 'destination_port', 'Flow Duration',
            'Total Fwd Packets', 'Total Backward Packets', 'Flow Bytes/s',
            'SYN Flag Count', 'ACK Flag Count', 'attack_type']:
    if col in normal.index:
        print(f"  {col}: {normal[col]}")

print("\n--- RAW Attack Sample ---")
attack = df_cicids[df_cicids['is_attack'] == 1].iloc[0]
for col in ['protocol', 'source_port', 'destination_port', 'Flow Duration',
            'Total Fwd Packets', 'Total Backward Packets', 'Flow Bytes/s',
            'SYN Flag Count', 'ACK Flag Count', 'attack_type']:
    if col in attack.index:
        print(f"  {col}: {attack[col]}")

# Show formatted text - pick one normal and one attack
import pandas as pd
sample = pd.concat([
    df_cicids[df_cicids['is_attack'] == 0].head(1),
    df_cicids[df_cicids['is_attack'] == 1].head(1),
])
df_cicids_fmt = format_dataset(sample, "cicids2017", fmt="natural_language")
normal_fmt = df_cicids_fmt[df_cicids_fmt['is_attack'] == 0].iloc[0]
attack_fmt = df_cicids_fmt[df_cicids_fmt['is_attack'] == 1].iloc[0]

print(f"\n--- FORMATTED Normal (label: {normal_fmt['attack_type']}) ---")
print(normal_fmt['text'])
print(f"\n--- FORMATTED Attack (label: {attack_fmt['attack_type']}) ---")
print(attack_fmt['text'])

# ── UNSW-NB15 ──
print("\n" + "=" * 80)
print("UNSW-NB15 EXAMPLES")
print("=" * 80)

df_unsw = load_dataset("unsw_nb15", "/scratch/yh6210/datasets/LLM_Rep_NetSec_datasets/unsw_nb15/")
print(f"\nTotal samples: {len(df_unsw)}")
print(f"Columns: {list(df_unsw.columns[:20])}...")
print(f"\nAttack type distribution:")
print(df_unsw['attack_type'].value_counts())

# Show 1 normal + 1 attack raw
print("\n--- RAW Normal Sample ---")
normal = df_unsw[df_unsw['is_attack'] == 0].iloc[0]
for col in ['protocol', 'source_port', 'destination_port', 'dur', 'sbytes', 'dbytes',
            'sttl', 'dttl', 'service', 'state', 'spkts', 'dpkts', 'attack_type']:
    if col in normal.index:
        print(f"  {col}: {normal[col]}")

print("\n--- RAW Attack Sample ---")
attack = df_unsw[df_unsw['is_attack'] == 1].iloc[0]
for col in ['protocol', 'source_port', 'destination_port', 'dur', 'sbytes', 'dbytes',
            'sttl', 'dttl', 'service', 'state', 'spkts', 'dpkts', 'attack_type']:
    if col in attack.index:
        print(f"  {col}: {attack[col]}")

# Show formatted text
sample = pd.concat([
    df_unsw[df_unsw['is_attack'] == 0].head(1),
    df_unsw[df_unsw['is_attack'] == 1].head(1),
])
df_unsw_fmt = format_dataset(sample, "unsw_nb15", fmt="natural_language")
normal_fmt = df_unsw_fmt[df_unsw_fmt['is_attack'] == 0].iloc[0]
attack_fmt = df_unsw_fmt[df_unsw_fmt['is_attack'] == 1].iloc[0]

print(f"\n--- FORMATTED Normal (label: {normal_fmt['attack_type']}) ---")
print(normal_fmt['text'])
print(f"\n--- FORMATTED Attack (label: {attack_fmt['attack_type']}) ---")
print(attack_fmt['text'])
