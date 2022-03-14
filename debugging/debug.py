import pickle
 
with open('debugging/masks.txt', 'rb') as f:
    masks = pickle.load(f)

with open('debugging/importances.txt', 'rb') as f:
    importances = pickle.load(f)

global_weight_dict = {}
for convlayer in importances:
    # sort weights in ascending order
    W = convlayer.weight.flatten().sort() # maybe sort by absolute value?
    print(type(W))
    
    # add indecies and weights to dictionary
    index_to_weight = {}
    for weightIndex, weight in enumerate(W.values):
        index_to_weight.update({weightIndex:weight})
    
    # calculate LAMP score for each weight
    for weightIndex in range(len(index_to_weight)):
        numerator = index_to_weight[weightIndex]*index_to_weight[weightIndex]

        # iterate through all weights >= current weight to calculate denominator of LAMP equation
        denominator = 0
        for vIndex in range(weightIndex, len(index_to_weight)):
            denominator += index_to_weight[vIndex]*index_to_weight[vIndex]

        LAMP = numerator/denominator
        # store each weight's LAMP score in the dict
        # new format) index:(weight,LAMP)
        index_to_weight[weightIndex] = (index_to_weight[weightIndex], LAMP)

    # add weight dict for this layer to the mega dict
    global_weight_dict.update(index_to_weight) # this has the issue of repeated keys
    
    # # sort dict by LAMP score
    # LAMPList = [x[1] for x in (list(index_to_weight.values()))]
    # sorted_by_lamp = {}
    # for score in LAMPList:
    #     for key in index_to_weight.keys():
    #         if index_to_weight[key][1] == score:
    #             sorted_by_lamp.update({key:index_to_weight[key]})
    #             break

    # # both are ordered??
    # # this is because lamp scores relative importance within a layer so actually this algo doesn't make sense
    # for thing in sorted_by_lamp:
    #     print(sorted_by_lamp[thing][0])
    # for thing in sorted_by_lamp:
    #     print(sorted_by_lamp[thing][1])
    
    
    # ones = torch.nn.Parameter(torch.ones_like(convlayer.weight))
    # dict = {} # store indeces and LAMP scores to convert flat tensor back into 4d tensor
    # print('new layer')
    # #print(W)
    # for u, idx in enumerate(W.values):
    #     numerator = u * u
    #     denominator = 0
    #     for v in W.values: # need to figure out how to iterate starting at idx
    #         denominator += v * v

    #     u = numerator/denominator

    # #print(W)