import math

while True:
    in_channels = int(input("Input channels: "))
    padding = int(input("Padding: "))
    kernel_size = int(input("Kernel size ( > 0): "))
    stride = int(input("Stride ( > 0): "))
    dilation = int(input("Dilation ( > 0): "))
    
    out_channels = math.floor(((in_channels + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1)
    print(f"Output shape: {out_channels}")