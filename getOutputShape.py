import math

def get_one_layer_output(in_channels, padding = 0, kernel_size = 3, stride=1, dilation=1):
    
    return math.floor(((in_channels + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1)

def get_output_shape(input_shape, layers):
    
    for i in range(layers):
        input_shape = get_one_layer_output(input_shape, kernel_size=3,stride=2)
    return input_shape

print(get_output_shape(4000, 4))
        
    
    