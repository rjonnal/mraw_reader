# Some utility functions for reading data from the Photoron MRAW format
import numpy as np
import os,sys

class MRAW:

    def __init__(self,filename,frame_width,frame_height,
                 n_frames,dtype=np.uint16,fps=1000.0,t0=0.0):
        """Create a source of frames, either for flood or OCT images.
        t0 and fps are used in convenience methods for pulling frames 
        by time instead of index."""
        self.filename = filename
        self.sy = frame_height
        self.sx = frame_width
        self.n_frames = n_frames
        self.dtype = dtype
        self.bytes_per_pixel = np.array([self.dtype(1)]).itemsize
        self.fps = fps
        self.t0 = t0
        self.file_size = os.stat(filename).st_size
        self.expected_file_size = self.sy*self.sx*self.bytes_per_pixel*self.n_frames
        # sanity check:
        assert self.file_size==self.expected_file_size
        self.dt = 1.0/self.fps
        self.t = np.arange(self.n_frames)*self.dt+self.t0

    def get_frame(self,index):
        try:
            with open(self.filename,'rb') as fid:
                skipbytes = index*self.sy*self.sx*self.bytes_per_pixel
                if skipbytes>=self.file_size:
                    sys.exit('Error: file %s too small to retrieve frame %d.'%(self.filename,index))
                arr = np.fromfile(fid,dtype=self.dtype,count=self.sx*self.sy,offset=skipbytes).reshape((self.sy,self.sx))
        except Exception as e:
            sys.exit(e)
        return arr

    def get_frame_t(self,time):
        idx = np.argmin(np.abs(self.t-time))
        return self.get_frame(idx)


    def get_stats(self,use_cache=True):
        head,tail = os.path.split(self.filename)
        cache_directory = os.path.join(head,'.image_stats_%s'%tail)

        try:
            imax = np.loadtxt(os.path.join(cache_directory,'imax.txt'))
            imin = np.loadtxt(os.path.join(cache_directory,'imin.txt'))
            imean = np.loadtxt(os.path.join(cache_directory,'imean.txt'))
            istd = np.loadtxt(os.path.join(cache_directory,'istd.txt'))
        except Exception as e:
            
            try:
                os.mkdir(cache_directory)
            except Exception as e:
                pass

            imax = []
            imin = []
            imean = []
            istd = []

            for idx in range(self.n_frames):
                arr = self.get_frame(idx)
                imax.append(arr.max())
                imin.append(arr.min())
                imean.append(arr.mean())
                istd.append(arr.std())

            np.savetxt(os.path.join(cache_directory,'imax.txt'),imax)
            np.savetxt(os.path.join(cache_directory,'imin.txt'),imin)
            np.savetxt(os.path.join(cache_directory,'imean.txt'),imean)
            np.savetxt(os.path.join(cache_directory,'istd.txt'),istd)

        return imax,imin,imean,istd
