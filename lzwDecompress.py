import struct
import numpy as np
import io
import argparse
import os

# takes in an LZW compressed file and decompresses it back into .npz

initDictSize = 256
maxDictSize = 2 ** 20
initCodeBits = 9

# verify provided file paths are real
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

class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.byteIndex = 0
        self.bitAccumulator = 0
        self.bitCount = 0

    def readBits(self, numBits: int) -> int:
        while self.bitCount < numBits:
            if self.byteIndex >= len(self.data):
                raise ValueError("Unexpected end of LZW bitstream")
            self.bitAccumulator |= (self.data[self.byteIndex] << self.bitCount)
            self.bitCount += 8
            self.byteIndex += 1
        value = self.bitAccumulator & ((1 << numBits) - 1)
        self.bitAccumulator >>= numBits
        self.bitCount -= numBits
        return value

# LZW decompression
def lzwDecompressBytes(compressedBytes: bytes) -> bytes:
    # verify header is readable and valid
    if len(compressedBytes) < 4:
        raise ValueError("Invalid LZW binary format: File header is missing, incomplete or corrupted")

    maxCodeBits = struct.unpack('<B', compressedBytes[0:1])[0]
    numCodes = struct.unpack('<I', compressedBytes[1:5])[0]

    if maxCodeBits <= 0 or maxCodeBits > 31:
        raise ValueError (f"Invalid max code width in header: {maxCodeBits}")

    if numCodes == 0:
        return b""

    reader = BitReader(compressedBytes[5:])

    # initialise basline dictionary
    dictionary = {i: bytes([i]) for i in range(initDictSize)}
    dictSize=initDictSize
    codeBits = initCodeBits

    decompressedBytes = bytearray()

    previousCode = reader.readBits(codeBits)
    if previousCode not in dictionary:
        raise ValueError(f"Corrupted LZW stream: first code {previousCode} not found in dictionary")
    sequence = dictionary[previousCode]
    decompressedBytes.extend(sequence)

    for _ in range(numCodes - 1):
        reservedSlot = None
        if dictSize < maxDictSize:
            reservedSlot = dictSize
            dictSize += 1
            if dictSize > (1 << codeBits) and codeBits < maxCodeBits:
                codeBits += 1

        currentCode = reader.readBits(codeBits)

        # process sequence codes
        if currentCode in dictionary:
            sequence = dictionary[currentCode]
        elif reservedSlot is not None and currentCode == reservedSlot: # handle boundary condition: pattern occurs before entering dictionary table
            previousSequence = dictionary[previousCode]
            sequence = previousSequence + bytes([previousSequence[0]])
        else:
            raise ValueError(f"Corrupted LZW stream: code {currentCode} not found in dictionary (dictSize={dictSize})")

        decompressedBytes.extend(sequence)

        if reservedSlot is not None:
            previousSequence = dictionary[previousCode]
            dictionary[reservedSlot] = previousSequence + bytes([sequence[0]])

        previousCode = currentCode

    return bytes(decompressedBytes)

# convert decompressed raw bytes back into .npz
def bytesToNpz(decompressedBytes: bytes, outputNpzPath: str) -> None:
    buffer = io.BytesIO(decompressedBytes)
    dataDict ={}

    # read total key count
    numKeys = struct.unpack('<I', buffer.read(4))[0]
    metadata = []

    # read metadata header
    for _ in range(numKeys):
        # key string
        keyLength = struct.unpack('<I', buffer.read(4))[0]
        key = buffer.read(keyLength).decode('utf-8')

        # array dimensions
        shapeLength = struct.unpack('<I', buffer.read(4))[0]
        shape = struct.unpack('<' + 'I' * shapeLength, buffer.read(4 * shapeLength))

        # data type string
        dtypeLength = struct.unpack('<I', buffer.read(4))[0]
        dtype = buffer.read(dtypeLength).decode('utf-8')

        metadata.append((key, shape, dtype))

        # rebuild arrays
    for key, shape, dtype in metadata:
        numElements = int(np.prod(shape)) if shape else 1
        byteSize = numElements * np.dtype(dtype).itemsize

        rawArrayBytes = buffer.read(byteSize)
        array = np.frombuffer(rawArrayBytes, dtype=dtype).reshape(shape)
        dataDict[key] = array
        
    # save to numpy container
    np.savez_compressed(outputNpzPath, **dataDict)

# entry point for decompression
def decompressLzwToNpz(inputLzwPath: str, outputNpzPath: str) -> None:
    with open(inputLzwPath, 'rb') as f_in:
        compressedBytes = f_in.read()

    compressedSize = len(compressedBytes)

    # decompress to memory
    decompressedBytes = lzwDecompressBytes(compressedBytes)
    decompressedSize = len(decompressedBytes)

    # rebuild .npz
    bytesToNpz(decompressedBytes, outputNpzPath)

    print(f"Decompressed '{inputLzwPath}' -> '{outputNpzPath}' successfully")
    print(f"The compressed file size: {compressedSize:,} bytes")
    print(f"The decompressed file size: {decompressedSize:,} bytes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompress .lzw to .npz")

    parser.add_argument("inputLzw", type=validateInputFile, help="Path to the input .lzw file")
    parser.add_argument("outputNpz", type=validateOutputFile, help="Target path to save the decompressed .npz file")

    args = parser.parse_args()

    decompressLzwToNpz(args.inputLzw, args.outputNpz)