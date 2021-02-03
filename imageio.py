
import os
import json
import struct
import numpy as np
import openslide
import imageproc as iproc

def get_fullpath(f):
    return os.path.abspath(os.path.expanduser(f))
def create_directory(f):
    if not os.path.exists(os.path.dirname(f)):
        os.makedirs(os.path.dirname(f))

# TODO: should this class have a lvl attribute?
#       Would simplify lots of methods
#       Some methods would need an optional lvl argument (e.g. rescale_px)
class SVSLoader:
    def __init__(self, fname):
        self.fname = get_fullpath(fname)
        self.svs = openslide.OpenSlide(self.fname)
        self.lc = self.svs.level_count
        self.dim = self.svs.level_dimensions
        self.ds = self.svs.level_downsamples

        # TODO: this attribute might not always exist...
        self.mpp = float(self.svs.properties['aperio.MPP'])
        self.props = self.svs.properties

        self.thumbnail = self.load_thumbnail()
        self.slide_color = iproc.get_slide_color(self.thumbnail)

    # getters
    def get_dim(self, lvl=None):
        if lvl is None:
            lvl = self.lvl
        return self.dim[lvl]

    def get_ds(self, lvl=None):
        if lvl is None:
            lvl = self.lvl
        return self.ds[lvl]

    # gets the highest and lowest resolution levels
    # NOTE: highest might always be level 0 and lowest might be level -1
    def get_thumbnail_lvl(self):
        return np.argmax(self.ds)
    def get_original_lvl(self):
        return np.argmin(self.ds)

    # setters
    def set_lvl(self, lvl):
        self.lvl = lvl

    def set_frame_dims(self, size, step, flocs=None):
        self.fsize = np.rint(size).astype(np.int)
        self.fstep = np.rint(step).astype(np.int)

        # define the location of the frames
        if flocs is None:

            # calculate the numbers of frames in the image
            self.nf = (self.get_dim() // self.fstep) + 1
            self.fds = self.get_dim() / self.nf

            self.flocs = np.zeros((self.nf[0], self.nf[1], 2), dtype=int)
            for i in range(self.nf[0]):
                for j in range(self.nf[1]):
                    x, y = i * self.fstep[0], j * self.fstep[1]
                    self.flocs[i][j] = np.array((x, y))
        else:
            self.nf = flocs.shape[0] + flocs.shape[1]
            self.fds = self.get_dim() / self.nf
            self.flocs = flocs

    def set_tile_dims(self, size, step):
        self.tsize = self.micron_to_px(size)
        self.tstep = self.micron_to_px(step)
        self.fpad = self.tsize - 1

    def get_tile_count(self, fsize):
        nt = ((fsize + self.fpad - self.tsize) // self.tstep) + 1
        tds = fsize / nt
        return nt, tds

    def run_tiling(self, ffunc, fargs, tfunc, targs):

        # define an array to hold the result of each frame analysis
        arr = np.zeros((self.nf[0], self.nf[1]), dtype=object)

        # load the frames one by one into memory
        for i in range(self.nf[0]):
            for j in range(self.nf[1]):

                # shift the location to account for center-aligned tiles
                x = self.flocs[i][j][0] - self.tsize[0]//2
                y = self.flocs[i][j][1] - self.tsize[1]//2
                loc = np.array((x, y))

                # load the frame
                frame = self.load_region(loc, pad_color=self.slide_color)

                # apply the frame processing
                frame = ffunc(frame, *fargs)

                # generate the tiles from the frame
                tiles = self.load_tiles(frame)

                # apply the tile processing
                arr[i][j] = tfunc(tiles, *targs)
        return arr

    # loads an ROI of the image based on the relative pixel resolution
    # NOTE: this allows any coordinates to be given
    #        1) if loc is negative, we decrease size, and shift loc to zero
    #        2) if size is bigger than image, we decrease size
    #        3) a padded rectangle will be added to top/bottom/left/right using
    #             the given color
    def load_region(self, loc, size=None, lvl=None, pad_color=None):
        if size is None:
            size = self.fsize + self.fpad
        if lvl is None:
            lvl = self.lvl
        if pad_color is None:
            pad_color = (255, 255, 255)

        size = np.copy(size)
        h, w = self.get_dim(lvl)
        padx1, pady1, padx2, pady2 = 0, 0, 0, 0

        # make sure the location isn't negative
        if loc[0] < 0:
            padx1 = 0 - loc[0]
            loc[0] = 0
            size[0] -= padx1
        if loc[1] < 0:
            pady1 = 0 - loc[1]
            loc[1] = 0
            size[1] -= pady1

        # make sure the size isn't past the image boundary
        if loc[0] + size[0] >= h:
            padx2 = loc[0] + size[0] - h
            size[0] -= padx2
        if loc[1] + size[1] >= w:
            pady2 = loc[1] + size[1] - w
            size[1] -= pady2

        # load the region
        # NOTE: upscale the location before accessing the image
        loc = self.upscale_px(loc, lvl)
        img = np.array(self.svs.read_region(
            location=loc, level=lvl, size=size))[:, :, :3]

        # pad img with rows at the top/bottom with cols of img size
        pady1 = np.full_like(img, pad_color, shape=(pady1, img.shape[1], 3))
        pady2 = np.full_like(img, pad_color, shape=(pady2, img.shape[1], 3))
        img = np.concatenate((pady1, img, pady2), axis=0)

        # pad img with cols at the left/right with rows of the padded img size
        padx1 = np.full_like(img, pad_color, shape=(img.shape[0], padx1, 3))
        padx2 = np.full_like(img, pad_color, shape=(img.shape[0], padx2, 3))
        img = np.concatenate((padx1, img, padx2), axis=1)

        return img

    def load_tiles(self, frame):

        fsize = np.array((frame.shape[1], frame.shape[0])) - self.fpad
        nt, tds = self.get_tile_count(fsize)
        print(frame.shape, fsize, nt, self.tsize, self.tstep, self.fpad)
        # define the shape output
        shape = (
            nt[0],
            nt[1],
            self.tsize[0],
            self.tsize[1],
        )

        # define the stride lengths in each dimension
        N = frame.itemsize
        strides = (
            N * frame.shape[1] * self.tstep[0], # N bytes between tile rows
            N * self.tstep[1],                  # N bytes between tile cols
            N * frame.shape[1],                 # N bytes between element rows
            N * 1,                              # N bytes between element cols
        )
        # perform the tiling
        # NOTE: this is pretty complicated... but very fast
        #       23) in this article -> https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20
        return np.lib.stride_tricks.as_strided(
            frame, shape=shape, strides=strides)

    def load_thumbnail(self):
        lvl = self.get_thumbnail_lvl()
        loc = np.array((0,0))
        size = self.get_dim(lvl)
        return self.load_region(loc, size, lvl)

    # scales a pixel value to the original pixel resolution
    def upscale_px(self, px, lvl=None, order=1):
        if lvl is None:
            lvl = self.lvl
        return np.rint(px * self.get_ds(lvl)**order).astype(np.int)
    # scales a pixel value to the relative pixel resolution
    def downscale_px(self, px, lvl=None, order=1):
        if lvl is None:
            lvl = self.lvl
        return np.rint(px / self.get_ds(lvl)**order).astype(np.int)

    # upscales to original pixel resolution, converts to micron value
    def px_to_micron(self, px, lvl=None, order=1):
        if lvl is None:
            lvl = self.lvl
        px = self.upscale_px(px, lvl, order)
        return px * self.mpp**order
    # converts micron to pixel, downscales to relative pixel resolution
    def micron_to_px(self, micron, lvl=None, order=1):
        if lvl is None:
            lvl = self.lvl
        px = micron / self.mpp**order
        return self.downscale_px(px, lvl, order)

#
# end of class


def write_tissue_detections(ifname, dname, tissue_detections):

    # prepare the output binary file
    ofname = get_fullpath(ifname) \
        .replace('images', dname) \
        .replace('.svs', '.dat')
    create_directory(ofname)
    fp = open(ofname, 'wb')

    # write the header
    # NOTE: formatted as follows
    #       4 bytes: number of detections (4 byte integer)
    #     N*4 bytes: number of vertices in N detections (4 byte integers)
    N = len(tissue_detections)
    fp.write(struct.pack('i', N))
    for i in range(N):
        fp.write(struct.pack('i', tissue_detections[i].shape[0]))

    # write the data
    # NOTE: formatted as follows
    #   M*4*2 bytes: flattened list of vertices (2 values each, 4 byte floats)
    for td in tissue_detections:
        fp.write(td.astype(np.float32).tobytes())
    fp.close()

def read_tissue_detections(ifname, dname):

    # prepare the input binary file
    ifname = get_fullpath(ifname) \
        .replace('images', dname) \
        .replace('.svs', '.dat')
    fp = open(ifname, 'rb')

    # read the header
    N = struct.unpack('i', fp.read(4))[0]
    lengths = struct.unpack('i' * N, fp.read(4 * N))

    # read the data
    tissue_detections = []
    for length in lengths:
        tissue_detections.append(
            np.frombuffer(fp.read(length*4*2), dtype=np.float32))

    return tissue_detections

# TODO: this should be in GeoJSON format, use that library
# NOTE: currently only handles a single polygon annotation
def read_qupath_annotations(ifname):

    fix_json(ifname)

    # prepare the input file
    fp = open(ifname, 'r')

    data = json.loads(fp.read())

    # TODO: handle the multipolygon better than this, why is there sometimes 2 sets of coords
    annotations = []
    for annotation in data:
        geo = annotation['geometry']
        if geo['type'] == 'MultiPolygon':
            coords_list = geo['coordinates']
            for coords in coords_list:
                x = np.array([float(c[0]) for c in coords[0]])
                y = np.array([float(c[1]) for c in coords[0]])
                area = iproc.calc_area(x, y)
                if area > 5000:
                    annotations.append(np.array(coords[0]))
        elif geo['type'] == 'Polygon':
            coords = geo['coordinates']
            x = np.array([float(c[0]) for c in coords[0]])
            y = np.array([float(c[1]) for c in coords[0]])
            area = iproc.calc_area(x, y)
            if area > 5000:
                annotations.append(np.array(coords[0]).astype(float))

    return annotations

def fix_json(ifname):
    fp = open(ifname, 'rb')
    header = fp.read(7)
    if header == bytes([172, 237, 0, 5, 116, 4, 249]):
        data = fp.read()
        fp.close()
        fp = open(ifname, 'wb')
        fp.write(data)
        fp.close()




#
# end of file
