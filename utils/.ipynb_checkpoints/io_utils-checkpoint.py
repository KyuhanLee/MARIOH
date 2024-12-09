def write_tuples_to_txt(tuples_list, file_path):
    with open(file_path, 'w') as f:
        for tup in tuples_list:
            line = ','.join(map(str, tup))
            f.write(line + '\n')
