import pickle
import os

def fmap_load(path):
    if not os.path.exists(path):
        return

    with open(path, 'rb') as f:
        return pickle.load(f)




if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    fmaps_info = fmap_load(path)
    # print(type(fmaps_info))
    # for k, v in fmaps_info.items():
    #     print(k, v['input_feat'])
