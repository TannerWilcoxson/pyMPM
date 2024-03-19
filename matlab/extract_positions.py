import numpy as np
from gsd import hoomd

def hoomd_to_txt(filename,start=None,end=None,freq=None):
    # Open gsd file
    end = filename.split('.')[-1]
    if end == "gsd" or end == "txt":
        filename = filename[:-len(end)]

    in_file = filename + '.gsd'
    out_file = filename + '.txt'

    traj = hoomd.open(name=in_file, mode='rb')
    with open(out_file, 'w') as f:
        for frame in file[start:end:freq]:
            N = frame.particles.N
            box = frame.configuration.box
            x = frame.particles.position
            
            f.write('{:d}\n'.format(N))
            f.write('{:.5f} {:.5f} {:.5f}\n'.format(box[0], box[1], box[2]))
            for pos in x:
                f.write('{:.5f} {:.5f} {:.5f}\n'.format(pos[0], pos[1], pos[2]))
            f.write('\n')
        f.close()
