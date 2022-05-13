import json

DATA_ROOT = "../../data"
OUT_ROOT = "../processed"

def get_KB_dict():
    KB_fashion_path = f"{DATA_ROOT}/fashion_prefab_metadata_all.json"
    KB_furniture_path = f"{DATA_ROOT}/furniture_prefab_metadata_all.json"

    with open(KB_fashion_path, 'r') as fashion_file:
        KB_fash = json.load(fashion_file)

    with open(KB_furniture_path, 'r') as furniture_file:
        KB_fur = json.load(furniture_file)

    KB_dict = {}
    idx = 1

    for key, object_KB in KB_fash.items():
        KB_dict[idx] = {}
        KB_dict[idx]['path'] = key
        KB_dict[key] = idx

        object_string = f'''
                    The object's price is {object_KB['price']}.
                    Its size is {object_KB['size']}.
                    Its brand is {object_KB['brand']}.
                    It has a customer review of {object_KB['customerReview']} out of 5.
                    It is available in sizes {' and '.join(object_KB['availableSizes'])}.
                '''
        object_string = object_string.split('\n')
        object_string = ' '.join([line.strip() for line in object_string]).strip()
        KB_dict[idx]['string'] = object_string

        idx += 1

    for key, object_KB in KB_fur.items():
        KB_dict[idx] = {}
        KB_dict[idx]['path'] = key
        KB_dict[key] = idx

        object_string = f'''
                    The object's price is {object_KB['price']}.
                    Its brand is {object_KB['brand']}.
                    It is made with {object_KB['materials']}.
                    It has a customer review of {object_KB['customerRating']} out of 5.
                '''
        object_string = object_string.split('\n')
        object_string = ' '.join([line.strip() for line in object_string]).strip()
        KB_dict[idx]['string'] = object_string

        idx += 1
    
    with open(f'{OUT_ROOT}/KB_dict.json', 'w', encoding='utf-8') as out_file:
        json.dump(KB_dict, out_file)

if __name__ == '__main__':
    get_KB_dict()