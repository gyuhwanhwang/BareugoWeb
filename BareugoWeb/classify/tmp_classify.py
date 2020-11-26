import random
def classifier(comments):
    index1 = sorted(random.sample([1,3,5,7], 3))
    index0 = sorted(random.sample([0,2,4,6,8], 1))
    data = {"0": index0, "1": index1}
    return data