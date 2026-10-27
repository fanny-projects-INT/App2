HISTOLOGY_RESAMPLING_VERSION = 3
"""Resample original histology using the saved sample-to-atlas deformation fields."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates


@lru_cache(maxsize=2)
def _deformation(folder, signatures):
    fields = [tifffile.memmap(Path(folder) / f"deformation_field_{i}.tiff") for i in range(3)]
    shape = np.array(fields[0].shape)
    grid = np.stack(np.meshgrid(*[np.linspace(0, s - 1, 12) for s in shape], indexing="ij")).reshape(3, -1)
    mapped = np.stack([map_coordinates(f, grid, order=1, prefilter=False) for f in fields])
    affine = np.linalg.lstsq(np.column_stack([grid.T, np.ones(grid.shape[1])]), mapped.T, rcond=None)[0]
    return fields, shape, affine, np.linalg.inv(affine[:3].T)


def inverse_coordinates(folder, index, height=80, width=114):
    folder = Path(folder)
    signatures = tuple((p.stat().st_mtime_ns, p.stat().st_size)
                       for p in (folder / f"deformation_field_{i}.tiff" for i in range(3)))
    fields, shape, affine, inverse_affine = _deformation(str(folder), signatures)
    with tifffile.TiffFile(folder / "downsampled_standard.tiff") as standard:
        atlas_shape = standard.series[0].shape
    if not 0 <= index < atlas_shape[0]:
        raise IndexError("Coronal slice out of bounds")
    yy, xx = np.meshgrid(np.linspace(0, atlas_shape[1] - 1, height),
                         np.linspace(0, atlas_shape[2] - 1, width), indexing="ij")
    target = np.stack([np.full_like(yy, index), yy, xx]).reshape(3, -1) * .025
    coordinates = inverse_affine @ (target - affine[3, :, None])
    for _ in range(24):
        mapped = np.stack([map_coordinates(f, coordinates, order=1, mode="nearest", prefilter=False) for f in fields])
        delta = inverse_affine @ (target - mapped)
        coordinates += np.clip(delta, -15, 15) * .8
        if np.max(np.abs(delta)) < 1e-4:
            break
    mapped = np.stack([map_coordinates(f, coordinates, order=1, mode="nearest", prefilter=False) for f in fields])
    error = np.linalg.norm(mapped - target, axis=0)
    valid = (error < .025) & np.all((coordinates >= 0) & (coordinates <= shape[:, None] - 1), axis=0)
    return coordinates.reshape(3, height, width), valid.reshape(height, width), error


def original_registered_slice(folder, index, channel, height=800, width=1140, mapping=None):
    folder = Path(folder)
    config = json.loads((folder / "brainreg.json").read_text())
    orientation = config.get("data_orientation", config.get("orientation"))
    if orientation not in ("asl", "asr"):
        raise ValueError(f"Unsupported original orientation: {orientation}")
    coarse, valid, _ = mapping if mapping is not None else inverse_coordinates(folder, index)
    yy, xx = np.meshgrid(np.linspace(0, coarse.shape[1] - 1, height),
                         np.linspace(0, coarse.shape[2] - 1, width), indexing="ij")
    coordinates = np.stack([map_coordinates(c, [yy, xx], order=1, prefilter=False) for c in coarse])
    valid = map_coordinates(valid.astype("uint8"), [yy, xx], order=0) > 0
    with tifffile.TiffFile(folder / "deformation_field_0.tiff") as deformation:
        sample_shape = np.array(deformation.series[0].shape)
    original = folder.parent / "fiji" / f"brain_C{channel}.tif"
    with tifffile.TiffFile(original) as source:
        shape = np.array(source.series[0].shape)
        coordinates *= ((shape - 1) / (sample_shape - 1))[:, None, None]
        # Allen atlas is ASR; ASL originals have the opposite lateral direction.
        if orientation == "asl":
            coordinates[2] = shape[2] - 1 - coordinates[2]
        coordinates = np.clip(coordinates, 0, shape[:, None, None] - 1)
        z, y, x = coordinates.reshape(3, -1)
        z0 = np.floor(z).astype(int)
        z1 = np.minimum(z0 + 1, shape[0] - 1)
        weight = z - z0
        result = np.zeros(z.size, dtype=np.float32)
        mapped = tifffile.memmap(original) if len(source.pages) != shape[0] else None
        active = valid.ravel()
        for plane in np.unique(np.concatenate([z0[active], z1[active]])):
            frame = (source.pages[int(plane)].asarray() if mapped is None else mapped[int(plane)]).astype(np.float32)
            for positions, weights in ((z0, 1 - weight), (z1, weight)):
                selected = (positions == plane) & active & (weights > 0)
                if selected.any():
                    result[selected] += map_coordinates(frame, [y[selected], x[selected]], order=1, prefilter=False) * weights[selected]
    return result.reshape(height, width)


def brain_crop_bounds(frame):
    """Find a stable field of view around the main tissue, expressed in fractions."""
    from scipy.ndimage import gaussian_filter, label, binary_fill_holes
    image = gaussian_filter(np.asarray(frame, dtype=np.float32), 1.4)
    low, high = np.percentile(image, (20, 95))
    if high - low < 3:
        return [0.0, 0.0, 1.0, 1.0]
    mask = binary_fill_holes(image > low + .18 * (high - low))
    labels, count = label(mask)
    if not count:
        return [0.0, 0.0, 1.0, 1.0]
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = sizes.max()
    if largest < .01 * frame.size:
        return [0.0, 0.0, 1.0, 1.0]
    mask = np.isin(labels, np.flatnonzero(sizes >= max(largest * .12, 30)))
    y, x = np.nonzero(mask)
    height, width = frame.shape
    left, right = x.min() / width, (x.max() + 1) / width
    top, bottom = y.min() / height, (y.max() + 1) / height
    # A fixed aspect ratio keeps anatomy undistorted; padding prevents tight cuts.
    scale = min(1.1, max(.6, right - left + .12, bottom - top + .12))
    cx, cy = (left + right) / 2, (top + bottom) / 2
    return [float(cx - scale / 2), float(cy - scale / 2), float(scale), float(scale)]


@lru_cache(maxsize=4)
def _sample_volume(path, signature):
    return tifffile.memmap(path)


def registered_preview_channels(folder, index, available, height=160, width=228):
    """Preview and detail share the same inverse map and atlas coordinates."""
    folder = Path(folder)
    mapping = inverse_coordinates(folder, index)
    coarse, valid, _ = mapping
    yy, xx = np.meshgrid(np.linspace(0, coarse.shape[1] - 1, height),
                         np.linspace(0, coarse.shape[2] - 1, width), indexing="ij")
    coordinates = np.stack([map_coordinates(c, [yy, xx], order=1, prefilter=False) for c in coarse])
    mask = map_coordinates(valid.astype("uint8"), [yy, xx], order=0) > 0
    frames = []
    for channel, exists in enumerate(available):
        if not exists:
            frames.append(None)
            continue
        path = folder / f"downsampled_brain_C{channel}.tiff"
        stat = path.stat()
        sample = _sample_volume(str(path), (stat.st_mtime_ns, stat.st_size))
        frame = map_coordinates(sample, coordinates, order=1, mode="constant", cval=0, prefilter=False)
        frame[~mask] = 0
        frames.append(frame)
    return frames


def native_histology_info(folder):
    folder = Path(folder)
    paths = [folder.parent / 'fiji' / f'brain_C{i}.tif' for i in range(2)]
    available = tuple(path.exists() for path in paths)
    path = next(path for path in paths if path.exists())
    with tifffile.TiffFile(path) as source:
        shape = source.series[0].shape
    return shape, available


def native_histology_slice(folder, index, channel, height=1140, width=1140):
    path = Path(folder).parent / 'fiji' / f'brain_C{channel}.tif'
    with tifffile.TiffFile(path) as source:
        shape = source.series[0].shape
        if not 0 <= index < shape[0]:
            raise IndexError(index)
        frame = source.pages[index].asarray() if len(source.pages) == shape[0] else tifffile.memmap(path)[index]
        yy, xx = np.meshgrid(np.linspace(0, shape[1]-1, height), np.linspace(0, shape[2]-1, width), indexing='ij')
        return map_coordinates(frame.astype(np.float32), [yy, xx], order=1, prefilter=False)


def native_preview_channels(folder, index, available, count, height=384, width=384):
    folder = Path(folder)
    config = json.loads((folder / 'brainreg.json').read_text())
    orientation = config.get('orientation', 'asl')
    frames = []
    for channel, exists in enumerate(available):
        if not exists:
            frames.append(None)
            continue
        path = folder / f'downsampled_brain_C{channel}.tiff'
        if not path.exists():
            frames.append(native_histology_slice(folder, index, channel, height, width))
            continue
        stat = path.stat()
        volume = _sample_volume(str(path), (stat.st_mtime_ns, stat.st_size))
        plane = volume[round(index * (volume.shape[0]-1) / max(count-1, 1))]
        if orientation == 'asl':
            plane = plane[:, ::-1]
        yy, xx = np.meshgrid(np.linspace(0, plane.shape[0]-1, height), np.linspace(0, plane.shape[1]-1, width), indexing='ij')
        frames.append(map_coordinates(plane.astype(np.float32), [yy, xx], order=1, prefilter=False))
    return frames
