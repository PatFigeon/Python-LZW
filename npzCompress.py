import struct
import numpy as np
import io

def compressNpzDirect(npzFilePath: str, outputLzwPath: str):
    # load NPZ into memory
    data = np.load(npzFilePath)
    buffer = io.BytesIO()
    buffer.write(struct.pack('I', len(data.files)))

    for key in data.files:
        array = data[key]
        shape = array.shape
        dtypeStr = str(array.dtype)

        buffer.write(struct.pack('I', len(key)))
        buffer.write(key.encode('utf-8'))
        buffer.write(struct.pack('I', len(shape)))
        buffer.write(struct.pack('I' * len(shape, *shape)))
        buffer.write(struct.pack('I', len(dtypeStr)))
        buffer.write(dtypeStr.encode('utf-8'))

    # raw array bytes
    for key in data.files:
        buffer.write(data[key].tobytes())

    rawBytes = buffer.getvalue()

    # send raw bytes to LZW
    compressedPayload = lzwCompress(rawBytes)

    with open(outputLzwPath, 'wb') as f:
        f.write(compressedPayload)
