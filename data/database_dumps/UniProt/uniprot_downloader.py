import requests as rq
import json
import os
import re
import datetime as dt
import time as t 


def search_uniprot(query: str = '', fields: list[str] = [], size: int = 5, includeIsoform: bool = False,
                   cursor: str = None, format_: str = 'json', params: dict = None, json_dump: str = None,
                   verbose = False) -> tuple[rq.models.Response, str]:
    '''
    Do a UniProt search through its API. Parameters are explained here: https://www.uniprot.org/help/api_queries

    Providing a custom params dict (https://requests.readthedocs.io/en/latest/api/?highlight=get#requests.request)
    overridess ALL other search parameters.
    '''
    if params is None:
        params = {
            'query': query,
            'format': format_,
            'fields': fields,
            'size': size,
            'includeIsoform': includeIsoform,
            'cursor': cursor,
        }

    uniprot_request = rq.Request(url=UNIPROT_API_URL, params=params, method='GET')
    uniprot_request = uniprot_request.prepare()

    log(f'Requesting data using URL {uniprot_request.url}', verbose)

    session = rq.Session()
    uniprot_request = session.send(uniprot_request)
    
    log(f'UniProt returned code {uniprot_request.status_code}', verbose)

    uniprot_request.raise_for_status()

    # Cursor is explained here: https://www.uniprot.org/help/pagination
    # Basically, it's a hash pointing to the next page of results. Passing it as
    # as parameter will retrieve the next set of results, following the current request.
    # This extracts it from the link returned in the headers.
    if 'link' in uniprot_request.headers.keys():
        header_link = uniprot_request.headers['link']
        cursor_match = re.search(string=header_link, 
                                 pattern=r'&cursor=.+&').group(0)
        next_cursor = cursor_match[8:-1]
    else:
        next_cursor = None

    if json_dump is not None:
        with open(json_dump, 'w') as out_json:
            log(f'Dumping response to json', verbose)
            json.dump(obj=uniprot_request.json(), fp=out_json)

    return uniprot_request, next_cursor


def log(msg: str, verbose: bool):
    if verbose:
        print(msg)


if __name__ == '__main__':
    UNIPROT_API_URL = 'https://rest.uniprot.org/uniprotkb/search'

    # Search query for UniProt
    UNIPROT_QUERY = 'reviewed:true AND keyword:KW-0597'  # Only manually annotated (> 500.000 proteins)

    # Fields to retrieve from UniProt
    RETRIEVE_FIELDS = ['accession', 'protein_name', 'organism_name', 'lineage',
                       'sequence', 'ft_mod_res']

    # Whether the search includes isoforms
    INCLUDE_ISOFORM = False

    # Size of the batch of proteins to retrieve on every search
    UNIPROT_FETCH_SIZE = 500   # UniProt sais 500 is the optimal size (https://www.uniprot.org/help/api_queries)

    # How many searches between logs
    LOG_EVERY = 25

    # Download folder for UniProt data
    TIMESTAMP = dt.datetime.now().strftime("%d_%m_%Y_%H_%M")
    DOWNLOAD_FOLDER = f'uniprot_dump_{TIMESTAMP}'

    os.makedirs(DOWNLOAD_FOLDER)

    start = t.time()

    finished = False
    fetched_results = 0
    next_cursor = None
    i = 0
    while not finished:
        uniprot_request, next_cursor = search_uniprot(query=UNIPROT_QUERY,
                                                      fields=RETRIEVE_FIELDS,
                                                      size=UNIPROT_FETCH_SIZE,
                                                      includeIsoform=INCLUDE_ISOFORM,
                                                      cursor=next_cursor,
                                                      json_dump=f'{DOWNLOAD_FOLDER}/{i}_{fetched_results}_{fetched_results + UNIPROT_FETCH_SIZE}.json',
                                                      verbose=True if (i + 1) % LOG_EVERY == 0 else False)

        fetched_results += len(uniprot_request.json()['results'])

        if (i + 1) % LOG_EVERY == 0:
            now = t.time()
            total_sequences = int(uniprot_request.headers["X-Total-Results"])
            print(f'Done {i + 1} searches in {now - start:.1f}s')
            print(f'Fetched {fetched_results}/{total_sequences} squences ({fetched_results / total_sequences * 100:.1f}% done)')

        if next_cursor is None:
            finished = True
            print('Finished!')

        i += 1
