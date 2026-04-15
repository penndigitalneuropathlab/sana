
import os
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

from tqdm import tqdm
import numpy as np
import skimage
import skfmm
import cv2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull

from skimage.morphology import skeletonize
from skan import Skeleton, draw, sholl_analysis, summarize

import pdnl_sana as sana
import pdnl_sana.image


def detect_somas(
        pos: pdnl_sana.image.Frame, 
        minimum_soma_radius:int =1):
    dist = cv2.distanceTransform(pos.img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    ctrs = skimage.feature.peak_local_max(
        dist,
        min_distance=int(round(1.5*minimum_soma_radius)),
        threshold_abs=minimum_soma_radius,
    )[:,::-1]

    polys = pos.to_polygons()[0]
    keep_ctrs = []
    for poly in polys:
        best_dist = 0
        best_ctr = None
        for ctr in ctrs:
            x, y = ctr
            if pdnl_sana.geo.ray_tracing(*ctr, poly):
                if dist[y,x] > best_dist:
                    best_dist = dist[y,x]
                    best_ctr = ctr
        if not best_ctr is None:
            keep_ctrs.append(best_ctr)
    
    ctrs = np.array(keep_ctrs)

    return ctrs

def segment_somas(pos, ctrs, n_directions=2, stride=1, sigma=3, fm_threshold=10, npools=1, instance_segment=True, debug=False):

    if len(ctrs) == 0:
        return []
    
    # create the rotated anisotropic gaussian filters
    step = sana.geo.Point(stride, stride, False, pos.level)
    thetas = np.linspace(0, np.pi, n_directions, endpoint=False)
    filters = [sana.filter.AnisotropicGaussianFilter(th=th, sg_x=1, sg_y=sigma) for th in thetas]

    # apply the filters to the positive pixels
    def apply_filter(filt, frame, step):
        return filt.apply(frame, step)
    if npools == 1:
        directions = [apply_filter(filt, pos, step) for filt in filters]
    else:
        directions = ThreadPool(npools).map(partial(apply_filter, frame=pos, step=step), filters)

    # calculate directional ratio using the min and max directional values
    mi = np.min(directions, axis=0)
    mx = np.max(directions, axis=0)
    directional_ratio = cv2.resize(np.divide(mi**3, mx, where=mx!=0), dsize=(pos.img.shape[1], pos.img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # initial points for fast march algorithm are the soma centers
    phi = np.full_like(pos.img[:,:,0], -1, dtype=int)
    phi[ctrs[:,1], ctrs[:,0]] = 1

    # speed is derived from the directional ratio
    speed = directional_ratio
    speed[pos.img[:,:,0] == 0] = 0

    # run the fast march
    time = skfmm.travel_time(phi, speed)
    time[time.mask] = np.inf

    # threshold the time to create the soma mask
    mask = sana.image.frame_like(pos, (time < fm_threshold).astype(np.uint8))

    if debug:
        fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
        axs = axs.ravel()
        axs[0].imshow(pos.img, cmap='gray')
        axs[0].plot(*ctrs.T, '*', color='red')
        axs[1].imshow(speed, cmap='gray')
        axs[2].imshow(time, cmap='gray')
        axs[3].imshow(mask.img, cmap='gray')

    return mask

def get_skeleton_vertices(skeleton, skel_id):
    
    skeleton = Skeleton(skeleton)
    df = summarize(skeleton)
    
    path_ids = list(df[df['skeleton-id'] == skel_id].index)
    if len(path_ids) == 0:
        print(skel_id)
        print(df[df['skeleton-id'] == skel_id])
    v = []
    for path_id in path_ids:
        v.append(skeleton.path_coordinates(path_id))
    if len(v) == 0:
        return []
        
    v = np.concatenate(v, axis=0)[:,:-1][:,::-1]
    
    return v

def match_polygons_to_skeleton(skeleton, polygons, debug=False):
    
    df = summarize(Skeleton(skeleton))

    if debug:
        fig, ax = plt.subplots(1,1)
        ax.imshow(skeleton, cmap='gray')
        [ax.plot(*polygon.T, color='red') for polygon in polygons]
        ax.invert_yaxis()
    
    # get all skeleton ids
    skel_ids = list(df['skeleton-id'].drop_duplicates())
    
    if debug:
        print('Number of Skeletons:', len(skel_ids))
        print('Number of Polygons to match:', len(polygons))

    ids = [None]*len(polygons)
    for skel_id in tqdm(skel_ids):
        
        # get the vertices of the current skeleton
        v = get_skeleton_vertices(skeleton, skel_id)
        if len(v) == 0:
            continue
        p = sana.geo.Polygon(v[:,0], v[:,1], is_micron=False, level=0)

        for i, polygon in enumerate(polygons):
            
            if not ids[i] is None:
                continue
                
            if p.is_partially_inside(polygon):
                ids[i] = skel_id
                
                if debug:
                    ax.plot(*p.T, color='blue')
                break
        else:
            if debug:
                ax.plot(*p.T, color='green')
      
    # get the unaccounted for skeletons
    other_ids = list(df['skeleton-id'].drop_duplicates())
    [other_ids.remove(skel_id) for skel_id in ids if not skel_id is None]    
    
    return ids, other_ids

def merge_skeletons(skeleton, ids, other_ids, distance_threshold=10, debug=False):
    new_skeleton = skeleton.copy()

    if debug:
        plotted = False
        
    # no somas to connect to, delete all processes
    if len(ids) == 0:
        return np.zeros_like(skeleton)

    vertices = {}
    for skel_id in ids:
        vertices[skel_id] = get_skeleton_vertices(skeleton, skel_id)

    # loop through the unattached skeletons
    for other_id in tqdm(other_ids):

        # get the vertices of this skeleton
        v_process = get_skeleton_vertices(skeleton, other_id)

        # stores the potential connections to each skeleton
        potential_connections = []

        # loop through main skeletons
        for skel_id in ids:

            # get the vertices the skeleton
            v = vertices[skel_id]

            if len(v_process) == 0 or len(v) == 0:
                continue

            # loop through vertices in the skeleton
            connections = []
            for v0 in v:

                # get euclid distance to all vertices in skeleton to merge into
                d = np.sqrt(np.sum(np.square(v_process - v0[None,:]), axis=1))
                
                idx = np.argmin(d)
                if d[idx] < distance_threshold:
                    line = np.array([v0, v_process[idx]])
                    dist = d[idx]
                    connections.append((line, dist))

            if len(connections) == 0:
                continue
                
            # store the best potential connection to this soma
            best_v0_idx = np.argmin([c[1] for c in connections])
            best_line, best_dist = connections[best_v0_idx]            
            potential_connections.append((skel_id, best_line, best_dist))

            if debug:
                if not plotted:
                    fig, ax = plt.subplots(1,1)
                    ax.imshow(skeleton, cmap='gray')
                    plotted = True
                ax.plot(*best_line.T)
                
        # no potential connections, delete this process
        if len(potential_connections) == 0:
            x = v_process[:,0]
            y = v_process[:,1]
            new_skeleton[y, x, 0] = 0
            continue
        
        # get the best connection to a soma
        closest_soma_idx = np.argmin(c[2] for c in potential_connections)
        l = potential_connections[closest_soma_idx][1]
    
        # draw the connection on the new skeleton
        x0 = np.min(l[:,0])
        x1 = np.max(l[:,0])
        if x0 != x1:
            m = (l[1,1]-l[0,1])/(l[1,0]-l[0,0])
            b = l[1,1]-m*l[1,0]
            for x in np.linspace(x0, x1, 100):
                y = m*x + b
                new_skeleton[int(round(y)),int(round(x))] = 1
        else:
            # special case: vertical line
            y0 = np.min(l[:,1])
            y1 = np.max(l[:,1])
            new_skeleton[y0:y1, x0] = 1

    new_skeleton = skeletonize(new_skeleton)
        
    if debug:
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(skeleton)
        axs[1].imshow(new_skeleton)

    return new_skeleton

def reconstruct_instances_from_skeleton(pos, skeleton, soma_polygons, debug=False):
    
    df = summarize(Skeleton(skeleton))
    
    new_pos = pos.copy()
    new_pos.img[:,:,:] = 0
    
    microglia_masks, soma_masks, microglia_skeletons, locs, sizes = [], [], [], [], []

    pad = 100
    
    soma_ids, _ = match_polygons_to_skeleton(skeleton, soma_polygons)

    microglia_instances = []    
    for microglia_id, (soma_id, soma_polygon) in enumerate(zip(soma_ids, soma_polygons)):

        soma_mask = sana.image.create_mask_like(pos, [soma_polygon])
        
        # no skeleton
        if soma_id is None:
            loc, size = soma_polygon.bounding_box()
            loc -= pad//2
            size += pad
            
            best_tile = soma_mask.copy()
            best_tile.crop(loc, size)
            soma_tile = best_tile.copy()
            skeleton_tile = best_tile.copy()
            skeleton_tile.img[:,:,:] = 0
            
            locs.append(loc)
            sizes.append(size)
            microglia_masks.append(best_tile)
            soma_masks.append(soma_tile)
            microglia_skeletons.append(skeleton_tile)
                
            continue
        
        # get the bounding box of the vertices
        v = get_skeleton_vertices(skeleton, soma_id)
        p = sana.geo.Polygon(v[:,0], v[:,1], False, 0)
        loc, size = p.bounding_box()
        loc -= pad//2
        size += pad

        # crop the frame and the skeleton by the bounding box
        skeleton_tile = sana.image.frame_like(pos, np.zeros_like(skeleton))
        skeleton_tile.img[p[:,1], p[:,0], 0] = 1
        skeleton_tile.crop(loc, size)
        pos_tile = pos.copy()
        pos_tile.crop(loc, size)
        soma_tile = soma_mask.copy()
        soma_tile.crop(loc, size)
        
        # start with the skeleton, OR'd with soma
        new_tile = sana.image.frame_like(pos_tile, skeleton_tile.img | soma_tile.img) 

        # DICE score with the original classified DAB
        score = np.mean(new_tile == pos_tile.img)

        # skeleton is blank, just return the soma
        if np.sum(skeleton_tile.img) == 0:
            microglia = MicrogliaInstance(soma_tile, skeleton_tile, best_tile, loc, size)
            microglia_instances.append(microglia)
            continue
        
        # dilate the mask a bit, then test DICE
        best_tile = new_tile
        best_score = score
        for r in range(1, 10):

            # perform dilation, OR'd with soma
            new_tile = sana.image.frame_like(pos_tile, skeleton_tile.img.astype(np.uint8))
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            new_tile.img = cv2.dilate(new_tile.img, kern)[:,:,None] | soma_tile.img

            # new DICE score
            score = np.mean(new_tile.img == pos_tile.img)

            if score > best_score:
                best_tile = new_tile
                best_score = score

        microglia = MicrogliaInstance(soma_tile, skeleton_tile, best_tile, loc, size)
        microglia_instances.append(microglia)

    return microglia_instances

class MicrogliaInstance:
    def __init__(self, soma, skeleton, mask, loc, size):
        self.soma = soma
        self.skeleton = skeleton
        self.mask = mask
        self.loc = loc
        self.size = size

    def to_features(self):
        v = np.full((18,), np.nan, dtype=float)

        self.to_polygon_features(v)
        self.to_soma_features(v)
        self.to_convexhull_features(v)
        if np.sum(self.skeleton) != 0:
            self.to_skeleton_features(v)
            self.to_sholl_features(v)
        else:
            v[10:18] = 0

        return v

    def to_polygon_features(self, v):
        polys = pdnl_sana.image.Frame(self.mask).to_polygons()[0]
        if len(polys) != 0:
            poly = polys[0]
            v[0] = poly.get_area()
            v[1] = poly.get_perimeter()
            v[6] = poly.get_circularity()
        else:
            v[0] = 0
            v[1] = 0
            v[6] = 0

    def to_soma_features(self, v):
        poly = pdnl_sana.image.Frame(self.soma).to_polygons()[0][0]
        v[7] = poly.get_area()
        v[8] = poly.get_perimeter()
        v[9] = poly.get_circularity()

    def to_convexhull_features(self, v):
        polys = pdnl_sana.image.Frame(self.mask).to_polygons()[0]
        if len(polys) != 0:
            poly = polys[0]
            ch = ConvexHull(poly)
            ch_xy = poly[ch.vertices]
            microglia_ch = pdnl_sana.geo.Polygon(*ch_xy.T)
            v[2] = microglia_ch.get_area()
            v[3] = microglia_ch.get_perimeter()
            v[4] = v[0] / v[2] # solidity
            v[5] = v[1] / v[3] # convexity

    def to_skeleton_features(self, v):
        df = summarize(Skeleton(self.skeleton))

        v[10] = np.sum(self.skeleton)
        v[11] = np.sum(df['branch-type'] == 2) # branchpoints
        v[12] = np.sum(df['branch-type'] == 1) # endpoints

    def to_sholl_features(self, v):
        soma = pdnl_sana.image.Frame(self.soma).to_polygons()[0][0]
        loc, size = soma.bounding_box()
        soma_center = np.rint(loc + size // 2).astype(int)
        pos_idxs = np.array(np.where(self.skeleton == 1)[:-1]).T[:,::-1]
        dist = np.sum(np.square(pos_idxs - soma_center), axis=1)
        center = pos_idxs[np.argmin(dist)][::-1]

        skeleton = Skeleton(self.skeleton[:,:,0])
        max_radius = np.max([center[0], center[1], self.skeleton.shape[0]-center[0], self.skeleton.shape[1]-center[1]])
        max_radius += 10
        radii = np.arange(4, max_radius, 4)
        center, radii, counts = sholl_analysis(skeleton, center=center, shells=radii)

        # branching index
        bi = 0
        for i in range(1, len(radii)):
            bi += np.max([(counts[i]-counts[i-1])*radii[i], 0])
        v[14] = bi
        v[15] = radii[np.argmax(counts)] # critical radius
        v[16] = np.max(counts)

        inv_soma = self.soma.copy()
        inv_soma = 1-inv_soma
        process_mask = pdnl_sana.image.Frame(self.skeleton.copy().astype(np.uint8))
        process_mask.mask(pdnl_sana.image.Frame(inv_soma.astype(np.uint8)))

        v[13] = len(process_mask.to_polygons()[0])
        if v[13] == 0:
            v[17] = 0
        else:
            v[17] = v[16] / v[13]
    
    def save(self, f):
        np.savez(f, soma=self.soma.img, skeleton=self.skeleton.img, mask=self.mask.img, loc=self.loc, size=self.size)

    def overlay_skeleton(self, frame, color=[255,0,0]):
        tile = frame.get_tile(self.loc, self.size)
        tile[self.skeleton[:,:,0] != 0] = color
        frame.set_tile(self.loc, self.size, tile)

    def overlay_microglia(self, frame, color=[255,0,0]):
        tile = frame.get_tile(self.loc, self.size)
        dil_mask = pdnl_sana.image.Frame(self.mask)
        dil_mask.apply_morphology_filter(pdnl_sana.filter.MorphologyFilter('dilation', 'ellipse', 3))
        polys = dil_mask.to_polygons()[0]
        outline = pdnl_sana.image.create_mask(np.array(tile.shape[:2][::-1]), polys, outlines_only=True, thickness=2)
        tile[outline.img[:,:,0] != 0] = color
        frame.set_tile(self.loc, self.size, tile)

    def overlay_prediction(self, frame, proba, colors):
        tile = frame.get_tile(self.loc, self.size)
        mask = pdnl_sana.image.Frame(self.mask)
        polys = mask.to_polygons()[0]
        chs = []
        for poly in polys:
            ch = ConvexHull(poly)
            chs.append(pdnl_sana.geo.Polygon(*poly[ch.vertices].T))
        outline = pdnl_sana.image.create_mask(np.array(tile.shape[:2][::-1]), chs, outlines_only=True, thickness=2)

        tile[outline.img[:,:,0] != 0] = colors[np.argmax(proba)]
        frame.set_tile(self.loc, self.size, tile)

def segment_microglia(pos, somas, debug):
    
    skeleton = skeletonize(pos.img)

    soma_ids, process_ids = match_polygons_to_skeleton(skeleton, somas, debug=debug)

    merged_skeleton = merge_skeletons(skeleton, soma_ids, process_ids, distance_threshold=30, debug=True)

    microglia_instances = reconstruct_instances_from_skeleton(pos, skeleton, somas, debug=debug)

    return microglia_instances

def segment_wsi_chunk_wrapper(args):
    return segment_wsi_chunk(*args)
def segment_wsi_chunk(temp_dir, i, j, hem_threshold, dab_threshold):
    if not os.path.exists(os.path.join(temp_dir, f"hem_{i}_{j}.png")):
        return

    logger = sana.logging.Logger('normal', os.path.join(temp_dir, f'parameters_{i}_{j}.pkl'))
    frame_loc = logger.data['loc']

    hem = sana.image.Frame(os.path.join(temp_dir, f"hem_{i}_{j}.png"))
    dab = sana.image.Frame(os.path.join(temp_dir, f"dab_{i}_{j}.png"))
    mask = sana.image.Frame(os.path.join(temp_dir, f"mask_{i}_{j}.png"))
    hem_int = hem.copy()

    # apply the global thresholds
    hem.threshold(hem_threshold)
    dab.threshold(dab_threshold)

    # clean up the stains
    dab.apply_morphology_filter(sana.filter.MorphologyFilter('closing', 'ellipse', 2))
    hem.apply_morphology_filter(sana.filter.MorphologyFilter('closing', 'ellipse', 2))
    hem.apply_morphology_filter(sana.filter.MorphologyFilter('opening', 'ellipse', 2))

    # remove positive DAB and pixels outside the mask
    hem.mask(dab, invert=True)
    hem.mask(mask)

    # find all the somas throughout the counterstain
    # TODO: parameter should be in microns!
    soma_ctrs = sana.process.detect_somas(hem, minimum_soma_radius=3)
    
    # segment the somas using polygons
    soma_polygons, _ = hem.instance_segment(soma_ctrs)[0]
    soma_polygons = [p for p in soma_polygons if not type(p) is list and len(p) != 0]
    
    # move the polygons into the slide coordinate system
    [p.translate(-frame_loc) for p in soma_polygons]

    # re-calculate the centers of the polygons using the bounding box
    soma_bbs = [p.bounding_box() for p in soma_polygons]
    soma_ctrs = np.array([loc + size//2 for (loc, size) in soma_bbs])

    # feature 1: calculate the area of each polygon
    soma_areas = np.array([p.get_area() for p in soma_polygons])

    # feature 2: calculate the mean HEM intensity within the polygon
    soma_ints = []
    for poly in soma_polygons:

        # move to frame coordinate system
        poly.translate(frame_loc)
        
        # extract tile based on the bounding box of the polygon
        loc, size = poly.bounding_box()
        tile = sana.image.Frame(hem_int.get_tile(loc, size))

        # create a mask of pixels within the polygon
        poly.translate(loc)
        tile_mask = sana.image.create_mask_like(tile, [poly])
        poly.translate(-loc)

        # calculate average intensity
        soma_ints.append(np.mean(tile.img[tile_mask.img != 0]))

        # move back to slide coordinate system
        poly.translate(-frame_loc)

    # combine into an (N,4) array
    soma_feats = np.concatenate([
        soma_ctrs, 
        soma_areas[:,None], 
        np.array(soma_ints)[:,None]], 
        axis=1)

    # cache the cell features
    np.save(os.path.join(temp_dir, f"feats_{i}_{j}.npy"), soma_feats)


def train_wm_segmenter(heatmap, wm_coords=None, gm_coords=None, priors=None):

    heatmap = heatmap.copy()
    h, w, d = heatmap.img.shape    
    feats = heatmap.img.reshape(h*w, d)
    non_zero = ~np.any(feats == 0, axis=1)

    ss = StandardScaler()
    ss.fit(feats[non_zero])
    feats = ss.transform(feats)

    # fig, axs = plt.subplots(1,d)
    # for i in range(d):
    #     axs[i].hist(feats[non_zero,i], bins=100, density=True,
    #             color='gray', alpha=0.7, edgecolor='k')

    model = Pipeline([
        ('ss', ss),
        ('gmm', GaussianMixture(n_components=2, covariance_type='full')),
    ])

    if not wm_coords is None:
        wm_coords = np.ravel_multi_index(wm_coords.T, (h,w))
        gm_coords = np.ravel_multi_index(gm_coords.T, (h,w))
        means_init = np.array([
            np.mean(feats[wm_coords], axis=0),
            np.mean(feats[gm_coords], axis=0),
        ])
        print(means_init)
        model.set_params(gmm__means_init=means_init)

        
        # for i in range(d):
        #     axs[i].axvline(means_init[0,i], color='red')
        #     axs[i].axvline(means_init[1,i], color='blue')

        model.get_params()['gmm'].fit(feats[non_zero])
        model.wm_label = 0
        
    elif not priors is None:
        model.get_params()['gmm'].fit(feats[non_zero])

        mu = model.get_params()['gmm'].means_

        # store votes for each label
        votes = np.zeros(mu.shape[0], dtype=int)
    
        # check each feature prior
        for i in range(priors.shape[0]):
    
            # figure out which label has the most extreme value for this feature
            if priors[i] == 1:
                label = np.argmax(mu[:,i])
            else:
                label = np.argmin(mu[:,i])
    
            # vote for this label
            votes[label] += 1
        model.wm_label = np.argmax(votes)
        model.label_agrees_with_priors = np.max(votes) == len(priors)
    else:
        print('need information on which label is wm')
        return None

    return model

def deploy_wm_segmenter(model, heatmap, open_r=21, close_r=2):
    h, w, d = heatmap.img.shape
    pred = model.predict(heatmap.img.reshape(h*w, d)).reshape(h, w, 1)
    
    wm_mask = sana.image.frame_like(heatmap, (pred == model.wm_label).astype(np.uint8))
    wm_mask.apply_morphology_filter(sana.filter.MorphologyFilter('closing', 'ellipse', close_r))
    wm_mask.apply_morphology_filter(sana.filter.MorphologyFilter('opening', 'ellipse', open_r))

    return wm_mask