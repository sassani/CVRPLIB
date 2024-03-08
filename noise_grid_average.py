import numpy as np
from multiprocessing import Process, Pipe, Queue, SimpleQueue, Manager
from scipy import signal


def noise_grid_average(num_frames=1, x_size=10, y_size=10, kernel_size=3):
    """Create a grid of points and average the points in a kernel_size x kernel_size window.
    
    Parameters
    ----------
    num_frames: int
        number of frames to generate
    x_size: int
        x dimension of the grid
    y_size: int
        y dimension of the grid
    kernel_size: int
        size of the window to average
    
    Return
    -------
    grids: np.array
        A grid of points with the average of the points in a kernel_size x kernel_size window    
    """
    offset = kernel_size // 2
    grids = np.zeros([num_frames, x_size, y_size, 3])
    grids[0] = np.random.rand(x_size, y_size, 3)
    frame = 1
    while frame < num_frames:
        # grids[frame] = _moving_average(grids[frame-1],  kernel_size)
        grids[frame] = _convo_sp(grids[frame-1],  kernel_size)
        # grids[frame] = _convo_mp(grids[frame-1],  kernel_size)
        frame += 1
    return grids

def _moving_average(data, kernel_size):
    x_size, y_size, depth = data.shape
    offset = kernel_size // 2
    results = np.zeros([x_size, y_size, depth])
    for x in range(x_size):
        for y in range(y_size):
            xxs = np.max([0, x - offset])
            yys = np.max([0, y - offset])
            xxe = np.min([x_size, x + offset+1])
            yye = np.min([y_size, y + offset+1])        
            results[x, y] = data[xxs:xxe, yys:yye].mean(axis=(0,1))
    return results

def _convo_sp(data, kernel_size):
        x_size, y_size, depth = data.shape
        kernel = np.ones([kernel_size, kernel_size])/kernel_size**2
        results = np.zeros([x_size, y_size, depth])
        for d in range(depth):
            results[:,:,d] = signal.convolve2d(data[:,:,d], kernel, mode='same', boundary='wrap')
        return results
    
def _convo_mp(data, kernel_size):
        x_size, y_size, depth = data.shape
        kernel = np.ones([kernel_size, kernel_size])/kernel_size**2
        results = np.zeros([x_size, y_size, depth])
        jobs=[]
        with Manager() as manager: 
            shared = manager.dict()
            for d in range(depth):
                shared[d] = []
                p = Process(target=_convo_workers, args=(data, kernel, d, shared))
                p.start()
                jobs.append(p)
                
            for job in jobs:
                job.join()
                
            for d in range(depth):
                results[:,:,d] = shared[d]
        return results
        
def _convo_workers(data, kernel, channel, shared):
        result =  signal.convolve2d(data[:,:,channel], kernel, mode='same', boundary='wrap')
        shared[channel] = result
    
if __name__ == "__main__":
    import time
    start = time.time()
    noise_grid_average(2, 10, 10, 3)
    print(time.time()-start)
