import struct
import numpy as np
import io

"""
Takes in .npz, converts it directly to bytes and performs LZW compression directly from memory 
"""
# conver .npz to bytes
def npzToBytes(npzFilePath: str) -> bytes:
    # load .npz into memory
    data = np.load(npzFilePath)
    buffer = io.BytesIO()

    # write metadata: number of keys (as a 4 byte unsigned int)
    buffer.write(struct.pack('I', len(data.files)))

    for key in data.files:
        array = data[key]
        shape = array.shape
        dtypeStr = str(array.dtype)

        # write key name length and key string
        keyBytes = key.encode('utf-8')
        buffer.write(struct.pack('I', len(keyBytes)))
        buffer.write(keyBytes)

        # write array dimensions
        buffer.write(struct.pack('I', len(shape)))
        buffer.write(struct.pack('I' * len(shape), *shape))

        # write data type length and string
        dtypeBytes = dtypeStr.encode('utf-8')
        buffer.write(struct.pack('I', len(dtypeBytes)))
        buffer.write(dtypeBytes)

    # write raw array bytes
    for key in data.files:
        buffer.write(data[key].tobytes())

    return buffer.getvalue()


# test routine
npzPath = "/Users/maciej/VS Code/OrbitCompressionAlgorithm/OriginalImplementation/LZW_image_compression/S2B_60HXD_20170910_0_L2A.npz"
memoryBytes = npzToBytes(npzPath)

with open ("/Users/maciej/VS Code/OrbitCompressionAlgorithm/OriginalImplementation/LZW_image_compression/outputTest.bin", "rb") as f:
    fileBytes = f.read()

assert memoryBytes == fileBytes, "You biffed it"
print ("Congratuwelldone, those 3 hours of reading were worth it")