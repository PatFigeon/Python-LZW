import struct
import numpy as np
import io
import argparse
import os


#Takes in .npz, converts it directly to bytes and performs LZW compression directly from memory

# check if provided input file is real
def validateInputFile(pathStr: str) -> str:
    if not os.path.exists(pathStr):
        raise argparse.ArgumentTypeError(f"The file '{pathStr}' does not exist.")
    if not os.path.isfile(pathStr):
        raise argparse.ArgumentTypeError(f"'{pathStr} is not an operable file.")
    return pathStr

# check if provided output directory is real
def validateOutputFile(pathStr: str) -> str:
    outputDir = os.path.dirname(pathStr) or "."
    if not os.path.exists(outputDir):
        raise argparse.ArgumentTypeError(f"Output directory '{pathStr}' does not exist.")
    return pathStr


# convert .npz to bytes
def npzToBytes(npzFilePath: str) -> bytes:
    # load .npz into memory
    data = np.load(npzFilePath)
    buffer = io.BytesIO()

    # write metadata: number of keys (as a 4 byte unsigned int)
    buffer.write(struct.pack('<I', len(data.files)))

    for key in data.files:
        array = data[key]
        shape = array.shape
        dtypeStr = str(array.dtype)

        # write key name length and key string
        keyBytes = key.encode('utf-8')
        buffer.write(struct.pack('<I', len(keyBytes)))
        buffer.write(keyBytes)

        # write array dimensions
        buffer.write(struct.pack('<I', len(shape)))
        buffer.write(struct.pack('<' + 'I' * len(shape), *shape))

        # write data type length and string
        dtypeBytes = dtypeStr.encode('utf-8')
        buffer.write(struct.pack('<I', len(dtypeBytes)))
        buffer.write(dtypeBytes)

    # write raw array bytes
    for key in data.files:
        buffer.write(data[key].tobytes())

    return buffer.getvalue()

initDictSize = 256
maxDictSize = 2 ** 20
initCodeBits = 9
maxCodeBits = 20

class BitWriter:
    def __init__(self):
        self.buffer = bytearray()
        self.bitAccumulator = 0
        self.bitCount = 0

    def writeBits(self, value: int, numBits: int) -> None:
        self.bitAccumulator |= (value << self.bitCount)
        self.bitCount += numBits
        while self.bitCount >= 8:
            self.buffer.append(self.bitAccumulator & 0xFF)
            self.bitAccumulator >>= 8
            self.bitCount -= 8

    def getBytes(self) -> bytes:
        if self.bitCount > 0:
            self.buffer.append(self.bitAccumulator & 0xFF)
            self.bitAccumulator = 0
            self.bitCount = 0
        return bytes(self.buffer)
    

# LZW compression
def lzwCompressBytes(rawData: bytes) -> bytes:

    # initialise dictionary
    dictionary = {bytes([i]): i for i in range(initDictSize)}
    dictSize = initDictSize
    codeBits = initCodeBits

    writer = BitWriter()
    numCodes = 0
    sequence = b""

    def emit(code: int) -> None:
        nonlocal numCodes
        writer.writeBits(code, codeBits)
        numCodes +=1

    # compression loop
    for byteInt in rawData:
        currentByte = bytes([byteInt])
        newSequence = sequence + currentByte

        if newSequence in dictionary:
            sequence = newSequence
        else:
            # emit code for existing sequence
            emit(dictionary[sequence])

            # if space allows, add new pattern to dictionary
            if dictSize < maxDictSize:
                dictionary[newSequence] = dictSize
                dictSize +=1
                if dictSize > (1 << codeBits) and codeBits < maxCodeBits:
                    codeBits += 1

            # reset to current byte
            sequence = currentByte

    # emit code for remaining sequence
    if sequence in dictionary:
        emit(dictionary[sequence])

    payload = writer.getBytes()

    # produce header for decompression: 4-byte int, little-endian
    outputBytes = bytearray(struct.pack('<B', maxCodeBits))
    outputBytes.extend(struct.pack('<I', numCodes))
    outputBytes.extend(payload)

    return bytes(outputBytes)

def compressNpzToLzw(inputNpzPath: str, outputLzwPath: str) -> None:
    # the interface between .npz -> bytes and bytes to compressed file
    rawPayloadBytes = npzToBytes(inputNpzPath)
    originalFileSize = len(rawPayloadBytes)

    compressedBytes = lzwCompressBytes(rawPayloadBytes)
    compressedFileSize = len(compressedBytes)

    with open(outputLzwPath, 'wb') as f_out:
        f_out.write(compressedBytes)

    
    # compare file sizes
    if originalFileSize > 0:
        compressionRatio = (1 - (compressedFileSize/originalFileSize)) * 100
        spaceSaved = originalFileSize - compressedFileSize
    else:
        compressionRatio = 0.0
        spaceSaved = 0

    print(f"Compressed '{inputNpzPath}' -> '{outputLzwPath}' successfully")
    print(f"The original file size: {originalFileSize:,} bytes")
    print(f"The compressed file size: {compressedFileSize:,} bytes")
    print(f"This saved: {spaceSaved:,} bytes (reduction of {compressionRatio:.2f}%)")

# fetch input and output from CLI input
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress .npz into .lzw")

    # get file paths
    parser.add_argument("inputNpz", type=validateInputFile, help="Path to the .npz input file")
    parser.add_argument("outputLzw", type=validateOutputFile, help="Path for the generated .lzw output file")

    args = parser.parse_args()

    # run with args
    compressNpzToLzw(args.inputNpz, args.outputLzw)