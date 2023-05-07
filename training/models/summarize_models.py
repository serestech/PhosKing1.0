'''
Takes a bunch of model.info files and prints a summary table in TSV
'''
from os import listdir
from os.path import abspath, dirname, basename
from collections import defaultdict
from pprint import pprint

def parse_info_file(path: str) -> defaultdict:
    info = defaultdict(str)
    info['info_file'] = basename(path)
    
    with open(path, 'r') as info_file:
        contents = [line.strip() for line in info_file]
    
    current_section = None
    for line in contents:
        if line.startswith('%'):
            current_section = line[1:]
            continue
        
        if current_section not in ('NAMESPACE', 'PERFORMANCE'):
            continue
        
        param, value = line.split(':')
        info[param] = value.strip()
    
    return info

HERE = abspath(dirname(__file__))

info_files = [file for file in listdir(HERE) if file.endswith('.info')]

params_to_print = (
    'info_file',
    'mode',
    'test_auc',
    'model_args',
    'aa_window',
    'batch_size',
    'frac_phos',
    'loss_fn',
    'lr',
    'wd',
    'optimizer',
    'no_zero_grad',
    'n_epochs',
    )
print('\t'.join(params_to_print))
for info_file in info_files:
    info = parse_info_file(f'{HERE}/{info_file}')
    print('\t'.join([info[param] for param in params_to_print]))
