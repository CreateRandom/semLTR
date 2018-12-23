import uuid
import collections

# namespace uuid as defined in some norm
namespace = uuid.UUID('6ba7b811-9dad-11d1-80b4-00c04fd430c8')

def compute_uuid_from_trec_id(trec_id,prefix='clueweb09'):
    # prefix : trec_id
    string_input = prefix + ':' + trec_id
    return str(uuid.uuid5(namespace,string_input))

# chunk a list of elements into a list of n_chunks sublists
def chunk_list(seq, n_chunks):
    return (seq[i::n_chunks] for i in range(n_chunks))

# see here: https://stackoverflow.com/a/9416020
# merges a list of dicts into a single dicts
def merge_dict_list(list_of_dicts):
    super_dict = {}
    for d in list_of_dicts:
        for k, v in d.items():
            super_dict[k] = v
    return super_dict