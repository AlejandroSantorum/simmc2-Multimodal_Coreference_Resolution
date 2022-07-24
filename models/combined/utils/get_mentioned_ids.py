import json, argparse


def extract_mentioned_ids(example):
    mentioned_ids = set()

    start = example.find('<SOM>')
    while start != -1:
        end = example.find('<EOM>', start)
        som_part = example[start+len('<SOM>'):end]

        id_start = som_part.find('<')
        while id_start != -1:
            id_end = som_part.find('>', id_start)
            id = int(som_part[id_start+1:id_end])
            mentioned_ids.add(id)
            id_start = som_part.find('<', id_end)

        start = example.find('<SOM>', end)

    return list(mentioned_ids)


def main(args):
    file_path = args.test_file_path
    store_path = args.store_path

    with open(file_path, 'r') as f:
        data = f.readlines()
    
    mentioned_list = []
    for line in data:
        mentioned_ids = extract_mentioned_ids(line)
        mentioned_list.append(mentioned_ids)
    
    with open(store_path, 'w') as f:
        json.dump(mentioned_list, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # path of the file with the test examples
    parser.add_argument('--test_file_path', default='./simmc2_dials_dstc10_devtest_predict.txt')
    # path to store the resulting file
    parser.add_argument('--store_path', default='./mentioned_ids.json')

    args = parser.parse_args()
    main(args)