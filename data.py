# coding=utf-8
"""
@file    data
@date    2023/5/19 15:47
@author  zlf
"""


def data_loader(path_str: str):
    word_list = []
    tag_list = []
    word_lists = []
    tag_lists = []
    with open(path_str) as reader:
        line_index = 0
        for line in reader:
            line_index += 1
            if line != "\n":
                lines = line.strip("\n").split("\t")
                if len(lines) != 2:
                    print(f"Error format in line[{line_index}]")
                word = lines[0]
                tag = lines[1]
                word_list.append(word)
                tag_list.append(tag)
            else:
                if len(word_list) > 0:
                    word_lists.append(word_list)
                    tag_lists.append(tag_list)
                    word_list = []
                    tag_list = []
    word2id = build_map(word_lists)
    tag2id = build_map(tag_lists)
    word2id["<pad>"] = len(word2id)
    word2id["<unk>"] = len(word2id)
    return word_lists, tag_lists, word2id, tag2id


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps
