import json

FASHION_METAFILE="../data/fashion_prefab_metadata_all.json"
FURNITURE_METAFILE="../data/furniture_prefab_metadata_all.json"

CURRENT_SPECIAL_TOKENS_FILE="../data_object_special/simmc_special_tokens.json"
ALL_SPECIAL_TOKENS_FILE="../data_object_special/simmc_all_special_tokens.json"

def generate_all_special_tokens(fashion_file, furniture_file, current_st_file, all_st_file):
    with open(fashion_file, 'r') as f:
        fash_meta = json.load(f)

    with open(furniture_file, 'r') as f:
        fur_meta = json.load(f)
    
    with open(current_st_file, 'r') as f:
        curr_st = json.load(f)
    
    for obj in fash_meta:
        for attr_name in fash_meta[obj]:
            if attr_name not in curr_st['additional_special_tokens']:
                curr_st['additional_special_tokens'].append(attr_name)
            if attr_name == 'availableSizes':
                for elem in fash_meta[obj][attr_name]:
                    if elem not in curr_st['additional_special_tokens']:
                        curr_st['additional_special_tokens'].append(elem)
            else:
                if fash_meta[obj][attr_name] not in curr_st['additional_special_tokens']:
                    curr_st['additional_special_tokens'].append(fash_meta[obj][attr_name])
    
    for obj in fur_meta:
        for attr_name in fur_meta[obj]:
            if attr_name not in curr_st['additional_special_tokens']:
                curr_st['additional_special_tokens'].append(attr_name)
            if attr_name == 'availableSizes':
                for elem in fur_meta[obj][attr_name]:
                    if elem not in curr_st['additional_special_tokens']:
                        curr_st['additional_special_tokens'].append(elem)
            else:
                if fur_meta[obj][attr_name] not in curr_st['additional_special_tokens']:
                    curr_st['additional_special_tokens'].append(fur_meta[obj][attr_name])
    
    with open(all_st_file, 'w') as f:
        json.dump(curr_st, f)




if __name__ == "__main__":
    generate_all_special_tokens(FASHION_METAFILE, FURNITURE_METAFILE, CURRENT_SPECIAL_TOKENS_FILE, ALL_SPECIAL_TOKENS_FILE)