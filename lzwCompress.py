import struct
import numpy as np
import io


#Takes in .npz, converts it directly to bytes and performs LZW compression directly from memory 

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

initDictSize = 8
maxDictSize = 256

# LZW compression
def lzwCompressBytes(rawData: bytes) -> bytes:

    # initialise dictionary
    dictionary = {bytes([i]): i for i in range(initDictSize)}
    dictSize = initDictSize

    compressedCodes = []
    sequence = b""

    # compression loop
    for byteInt in rawData:
        currentByte = bytes([byteInt])
        newSequence = sequence + currentByte

        if newSequence in dictionary:
            sequence = newSequence
        else:
            # emit code for existing sequence
            if sequence in dictionary:
                compressedCodes.append(dictionary[sequence])

            # if space allows, add new pattern to dictionary
            if dictSize < maxDictSize:
                dictionary[newSequence] = dictSize
                dictSize +=1

            # reset to current byte
            sequence = currentByte

    # emit code for remaining sequence
    if sequence in dictionary:
        compressedCodes.append(dictionary[sequence])

    # produce header for decompression: 4-byte int, little-endian
    outputBytes = bytearray(struct.pack('<i', dictSize))

    # payload: 4-byte int per code
    for code in compressedCodes:
        outputBytes.extend(struct.pack('<i', code))

    return bytes(outputBytes)

def compressNpzToLzw(inputNpzPath: str, outputLzwPath: str) -> None:
    # the interface between .npz -> bytes and bytes to compressed file
    rawPayloadBytes = npzToBytes(inputNpzPath)
    compressedBytes = lzwCompressBytes(rawPayloadBytes)

    with open(outputLzwPath, 'wb') as f_out:
        f_out.write(compressedBytes)

    print(f"Compressed '{inputNpzPath}' -> '{outputLzwPath}' successfully")

# testing
input = "/Users/maciej/VS Code/OrbitCompressionAlgorithm/pythonLZWRewrite/S2B_60HXD_20170910_0_L2A.npz"
output = "/Users/maciej/VS Code/OrbitCompressionAlgorithm/pythonLZWRewrite/outputLzw.lzw"
compressNpzToLzw(input, output)