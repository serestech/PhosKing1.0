import sys, os
from urllib.request import urlopen
import requests as rq
from time import sleep
from json import dump as jsdump

if len(sys.argv) != 3:
    print(f'Usage: {os.path.basename(__file__)} seq_list output_file', file=sys.stderr)
    sys.exit(1)

seq_list_path, out_path = sys.argv[1:]

with open(seq_list_path, 'r') as seq_list_file:
    seq_list = set(line.strip() for line in seq_list_file)
    seq_list.remove('NULL')

def get_html(url: str) -> str:
    return urlopen(url).read().decode().replace('\r\n', '\n')

sequences = {}
seqs_for_uniparc = set()
for i, id in enumerate(seq_list):
    try:
        url = f'https://rest.uniprot.org/uniprotkb/{id}.fasta'
        fasta: str = get_html(url)
        if fasta.strip() == '':
            seqs_for_uniparc.add(id)
            continue
        lines = fasta.splitlines()
        assert lines[0].startswith('>')
        seq = ''.join(lines[1:])
        sequences[id] = seq
        print(f'{i + 1}/{len(seq_list)} ({((i + 1)/len(seq_list)) * 100:.2f}%)', end='\r')
    except Exception as err:
        print(err)
        print(f'{url=}', f'{fasta=}', f'{lines=}', f'{id=}', sep='\n')
        print('Continuing...')

print(f'Failed to retrieve {len(seqs_for_uniparc)} that will be searched in UniParc')

try:
    id_mapping_results = []
    if len(seqs_for_uniparc) > 0:
        print('Getting UniParc sequences from ID mapping API...')
        params = {'ids': ','.join(seqs_for_uniparc),
                'from': "UniProtKB_AC-ID",
                'to': 'UniParc'}
        
        req = rq.post('https://rest.uniprot.org/idmapping/run', params=params)
        req.raise_for_status()
        jobID = req.json()['jobId']
        print(f'JobID is {jobID}')
        
        job_status = None
        link = None
        while job_status != 'FINISHED':
            sleep(3)
            print('Checking ID mapping job status...')
            req = rq.get(f'https://rest.uniprot.org/idmapping/status/{jobID}')
            req.raise_for_status()
            try:
                job_status = req.json()['jobStatus']
            except KeyError as err:
                if 'results' in req.json().keys():
                    job_status = 'FINISHED'
                    id_mapping_results.extend(req.json()["results"])
                    link = req.headers['Link'] if 'Link' in req.headers.keys() else None
                else:
                    print(err)
                    print(f'{req.json()=}')
                    print('continuing to wait for results...')
            
            print(f'Job status is {job_status}')
        
        while link is not None:
            link = link[1:link.find('>')]
            print(f'Getting {link}')
            req = rq.get(link)
            link = req.headers['Link'] if 'Link' in req.headers.keys() else None
            id_mapping_results.extend(req.json()["results"])
    
    print(f'Got {len(id_mapping_results)} sequences from UniParc')
            
    for result_dict in id_mapping_results:
        uniprot_id = result_dict['from']
        sequence = result_dict['to']['sequence']['value']
        sequences[uniprot_id] = sequence
        
except Exception as err:
    from traceback import format_exc
    print(format_exc())
    print('Skipped UniParc search')

print(f'Writing output file {out_path}')
with open(out_path, 'w') as outfile:
    for id, seq in sequences.items():
        fastaseq = f'> {id}\n{seq}\n'
        outfile.write(fastaseq)
