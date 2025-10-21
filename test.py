import numpy as np
import math
import random

#等距抽样
def Systematic_sampling(data:list[float],m,a) -> list[float]:
    part_len = math.ceil(1/a)
    # print(part_len)
    sample_n = math.ceil(m * a)
    # print(sample_n)
    result = [data[i*part_len + random.randint(0,part_len-1)] for i in range(sample_n)]


    return result

#分层抽样；假定已经分层
def Stratified_sampling(data,m,a,k,layer_size) -> list[float]:
    id = 0
    result = []
    for size in layer_size:
        layer = data[id:id+size]
        id += size
        n = max(1, int(len(layer) * a))
        if n == 1:
            result.append(layer[0])
        else:
            step = len(layer)//n
            picks = [layer[i] for i in range(0, len(layer), step)][:n]
            result.append(picks)
    return result

if __name__ == "__main__":
    data = [50,50.3,49.8,49.9,47.6,47,47.5,43,43.2,43.3, 43.1,43.3,43,43.5,0]
    m = 15
    a = 1 / 3
    k = 4
    layer_size = [4,3,7,1]
    systematic = Systematic_sampling(data,m,a)
    stratified = Stratified_sampling(data,m,a,k,layer_size)
    print("等距抽样结果:",systematic)
    print("分层抽样结果:",stratified)

