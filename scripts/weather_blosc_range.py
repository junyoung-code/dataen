"""Fetch selected Blosc blocks; the official Blosc2 library decodes the values.

Format: https://www.blosc.org/c-blosc2/format/chunk_format.html
No compressed data stream or filter is decoded here. Unusual layouts use a full
chunk read. This is intended for ZIP_STORED Zarr chunks, never whole ZIP files.
"""
import struct

import numpy as np


def selected_values(read, length, indices, dtype, read_many=None):
    import blosc2
    dtype = np.dtype(dtype)
    indices = np.asarray(indices, dtype=np.int64)
    head = read(0, min(8192, length))
    if len(head) < 16:
        raise ValueError("Truncated Blosc header")
    nbytes, blocksize, cbytes = struct.unpack_from("<iii", head, 4)
    if nbytes <= 0 or cbytes != length or nbytes % dtype.itemsize:
        raise ValueError("Invalid Blosc lengths")
    if (indices < 0).any() or (indices >= nbytes // dtype.itemsize).any():
        raise ValueError("Selection outside chunk")
    def full():
        return np.frombuffer(blosc2.decompress2(read(0, length), nthreads=1), dtype=dtype)[indices]
    extended = head[2] & 5 == 5
    header_size = 32 if extended else 16
    encoded_itemsize = head[3]
    # Special/memcpy/lazy/variable-block/delta chunks need different addressing.
    # Falling back remains bounded by one compressed chunk, not the remote ZIP.
    if (not encoded_itemsize or nbytes % encoded_itemsize or blocksize % encoded_itemsize
            or head[2] & 2 or not extended or len(head) < 32 or head[31] & 0xFE
            or head[30] or 3 in head[16:22] or blocksize <= 0 or blocksize % dtype.itemsize):
        return full()
    blocks = (nbytes + blocksize - 1) // blocksize
    table_end = header_size + 4 * blocks
    if table_end > length:
        raise ValueError("Invalid Blosc block table")
    table = head[:table_end] if table_end <= len(head) else read(0, table_end)
    starts = np.frombuffer(table, dtype="<i4", count=blocks, offset=header_size).astype("int64")
    if (starts < table_end).any() or (starts >= length).any() or len(set(starts.tolist())) != blocks:
        return full()
    prefix_end = int(starts.min())  # Includes optional compression dictionary.
    prefix = head[:prefix_end] if prefix_end <= len(head) else read(0, prefix_end)
    physical = sorted(starts.tolist() + [length])
    ends = dict(zip(physical[:-1], physical[1:]))
    needed = np.unique(indices * dtype.itemsize // blocksize)
    ranges = sorted((int(starts[b]), ends[int(starts[b])]) for b in needed)
    # Merge adjacent blocks and small gaps to avoid one HTTP request per stream.
    merged = []
    for start, end in ranges:
        if merged and start - merged[-1][1] <= 16384:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    sparse = bytearray(length)
    sparse[:prefix_end] = prefix
    lengths = [(start, end - start) for start, end in merged]
    payloads = read_many(lengths) if read_many else [read(start, count) for start, count in lengths]
    for (start, end), data in zip(merged, payloads, strict=True):
        if len(data) != end - start:
            raise ValueError("Truncated compressed block")
        sparse[start:end] = data
    container = blosc2.SChunk(chunksize=nbytes, cparams={"typesize": encoded_itemsize}, dparams={"nthreads": 1})
    container.insert_chunk(0, bytes(sparse))
    result = np.empty(indices.shape, dtype=dtype)
    for block in needed:
        start = int(block * blocksize // dtype.itemsize)
        stop = min(start + blocksize // dtype.itemsize, nbytes // dtype.itemsize)
        decoded = np.frombuffer(container.get_slice(start=start * dtype.itemsize // encoded_itemsize,
                                stop=stop * dtype.itemsize // encoded_itemsize), dtype=dtype)
        mask = indices * dtype.itemsize // blocksize == block
        result[mask] = decoded[indices[mask] - start]
    return result
