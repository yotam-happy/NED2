import os

def getPath():
    _path = "/home/yotam/pythonWorkspace/deepProject"
    pc_name = 'yotam'
    if not os.path.isdir(_path):
        pc_name = 'euterpe'
        _path = "/home/noambox/DeepProject"
    if (not os.path.isdir(_path)):
        _path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"
        pc_name = 'noam'

    return _path, pc_name