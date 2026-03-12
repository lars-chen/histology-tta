import json, glob, pandas as pd

files = sorted(glob.glob('results/raw/tta_results_*.json'))

histology = ['gigapath', 'hoptimus', 'phikon', 'phikon2', 'uni', 'uni2', 'virchow', 'virchow2']
equivariant = ['d4wrn']

rows = []
pc_rows = []

for f in files:
    with open(f) as fh:
        data = json.load(fh)
    
    for entry in data:
        model = entry['model']
        model_type = 'histology' if model in histology else 'equivariant' if model in equivariant else 'general'
        
        # Detect train_subset from checkpoint name (e.g. "..._sub1000_seed42_best")
        ckpt = entry.get('checkpoint', '')
        train_subset = None
        if '_sub' in ckpt:
            import re
            m = re.search(r'_sub(\d+)_', ckpt)
            if m:
                train_subset = int(m.group(1))

        rows.append({
            'model': model, 'dataset': entry['dataset'],
            'backbone_mode': entry['backbone_mode'], 'train_augment': entry['train_augment'],
            'seed': entry['seed'], 'model_type': model_type,
            'strategy': entry['strategy'], 'aggregation': entry['aggregation'],
            'train_subset': train_subset,
            'n_test': entry.get('n_total'),
            'acc': entry['acc'], 'balanced_acc': entry['balanced_acc'],
            'n_correct': entry.get('n_correct'), 'n_wrong': entry.get('n_wrong'),
            'n_corrected': entry.get('n_corrected'), 'n_corrupted': entry.get('n_corrupted'),
            'total_unc': entry.get('total_unc'), 'aleatoric_unc': entry.get('aleatoric_unc'),
            'epistemic_unc': entry.get('epistemic_unc'), 'agreement_rate': entry.get('agreement_rate'),
            'ece': entry.get('ece'),
        })
        
        if 'per_class' in entry and entry['per_class']:
            for cls_data in entry['per_class']:
                pc_rows.append({
                    'model': model, 'dataset': entry['dataset'],
                    'backbone_mode': entry['backbone_mode'], 'train_augment': entry['train_augment'],
                    'seed': entry['seed'], 'model_type': model_type,
                    'strategy': entry['strategy'], 'aggregation': entry['aggregation'],
                    'class_name': cls_data.get('class'),
                    'precision': cls_data.get('precision'), 'recall': cls_data.get('recall'),
                    'f1': cls_data.get('f1'), 'support': cls_data.get('support'),
                })

pd.DataFrame(rows).to_csv('results/tta_results.csv', index=False)
pd.DataFrame(pc_rows).to_csv('results/tta_per_class.csv', index=False)
