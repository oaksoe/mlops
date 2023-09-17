import os
import random
import shutil
from ruamel.yaml import YAML

with open("params.yaml") as f:
    yaml = YAML(typ='safe')
    params = yaml.load(f)

seedValue = params['prepare']['seed']
split = params['prepare']['split']

classNames =  os.listdir(os.path.join(os.getcwd(), 'data', 'raw'))

for aclass in classNames:
    if not os.path.exists(os.path.join(os.getcwd(), 'data', 'prepared', 'train', aclass)):
        os.makedirs(os.path.join(os.getcwd(), 'data','prepared', 'train', aclass))
    if not os.path.exists(os.path.join(os.getcwd(), 'data','prepared', 'val', aclass)):
        os.makedirs(os.path.join(os.getcwd(), 'data','prepared', 'val', aclass))
    if not os.path.exists(os.path.join(os.getcwd(), 'data','prepared', 'test', aclass)):
        os.makedirs(os.path.join(os.getcwd(), 'data','prepared', 'test', aclass))
    directory = os.path.join(os.getcwd(), 'data', 'raw', aclass)
    fileNames = os.listdir(directory)
    random.Random(seedValue).shuffle(fileNames)
    nTrainFiles = int(split * len(fileNames))
    nValFiles = 1 #int((1 - split)/2 * len(fileNames))
    nTestFiles = len(fileNames) - nTrainFiles - nValFiles
    print("Copying files into train directory for class {}...".format(aclass))
    for i in range(nTrainFiles):
        shutil.copy(os.path.join(os.getcwd(), 'data', 'raw', aclass, fileNames[i]), \
                    os.path.join(os.getcwd(), 'data', 'prepared', 'train', aclass, fileNames[i]))
    print("Copying files into val directory for class {}...".format(aclass))
    for i in range(nTrainFiles, nTrainFiles + nValFiles):
        shutil.copy(os.path.join(os.getcwd(), 'data', 'raw', aclass, fileNames[i]), \
                    os.path.join(os.getcwd(), 'data', 'prepared', 'val', aclass, fileNames[i]))
    print("Copying files into test directory for class {}...".format(aclass))
    for i in range(nTrainFiles + nValFiles, nTrainFiles + nValFiles + nTestFiles):
        shutil.copy(os.path.join(os.getcwd(), 'data', 'raw', aclass, fileNames[i]), \
                    os.path.join(os.getcwd(), 'data', 'prepared', 'test', aclass, fileNames[i]))