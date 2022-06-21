



def get_last_user_turns(filepath):
    last_user_turns_list = []

    with open(filepath, 'r') as f:
        for line in f.readlines():
            if len(line) > 1:
                idx_usr = line.rfind('User :')
                idx_soo = line.find(' <SOO>')
                last_user_turn = line[idx_usr+len('User : '):idx_soo]
                last_user_turns_list.append(last_user_turn)
   
    return last_user_turns_list
                


if __name__ == '__main__':
    path = "../data_object_special/simmc2_dials_dstc10_train_predict.txt"
    last_user_turns = get_last_user_turns(path)
    print(last_user_turns)
