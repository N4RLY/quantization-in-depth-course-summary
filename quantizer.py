# The code is from the course "Quantization in Depth" by deeplearning.ai
# https://learn.deeplearning.ai/courses/quantization-in-depth/

import torch
import torch.nn as nn
import torch.nn.functional as F

###### Linear Quantization

def w8_a16_forward(weight, input, scales, bias=None):
    
    casted_weights = weight.to(input.dtype)
    output = F.linear(input, casted_weights) * scales
    
    if bias is not None:
        output = output + bias
      
    return output

class W8A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()
    
        self.register_buffer(
            "int8_weights",
            torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        )
        
        self.register_buffer("scales", torch.randn((out_features), dtype=dtype))
        
        if bias:
            self.register_buffer("bias", torch.randn((1, out_features), dtype=dtype))
        else:
            self.bias = None

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)

        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights
                        /scales.unsqueeze(1)).to(torch.int8)

        self.int8_weights = int8_weights
        self.scales = scales
    
    def deqantize(self):
        return self.int8_weights * self.scales.unsqueeze(1)
    
    def calculate_error(self, weights):
        return (weights - self.deqantize()).abs().mean()

    def forward(self, input):
        return w8_a16_forward(self.int8_weights, input, self.scales, self.bias)

def replace_linear_with_target(module, target_class, module_name_to_exclude):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
          any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias

            new_module = target_class(child.in_features, child.out_features, old_bias is not None, child.weight.dtype)
            setattr(module, name, new_module)
            if old_bias is not None:
              getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target(child, target_class, module_name_to_exclude)

def replace_linear_with_target_and_quantize(module, target_class, module_name_to_exclude):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
        any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias
            old_weight = child.weight

            new_module = target_class(child.in_features, child.out_features, old_bias is not None, child.weight.dtype)
            setattr(module, name, new_module)

            getattr(module, name).quantize(old_weight)
            
            if old_bias is not None:
              getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize(child, target_class, module_name_to_exclude)

### Example usage 
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model_id = "..."
# model = AutoModelForCausalLM.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)

### We should exclude the lm_head layer from quantization
# replace_linear_with_target_and_quantize(model, W8A16LinearLayer, ["lm_head"])

### Now the model is quantized and ready to be used

### To save weights:
# quantized_state_dict = model.state_dict()
# torch.save(quantized_state_dict, "quantized_state_dict.pth")

### To load model skeleton:
# config = AutoConfig.from_pretrained(model_id)
# with torch.device("meta"):
#   model = OPTForCausalLM(config)
# replace_linear_with_target(model, W8A16LinearLayer, ["lm_head"])

### To load weights:
# state_dict = torch.load("quantized_state_dict.pth")
# model.load_state_dict(state_dict, strict=True, assign=True)

###### Weights packing and unpacking

# Packing weights to int8
def pack_weights(uint8tensor, bits):
    if uint8tensor.shape[0] * bits % 8 != 0:
        raise ValueError(f"The input shape needs to be a mutiple of {8 / bits} - got {uint8tensor.shape[0]}")

    num_values = uint8tensor.shape[0] * bits // 8
    num_steps = 8 // bits
    unpacked_idx = 0
    packed_tensor = torch.zeros((num_values), dtype=torch.uint8)

    for i in range(num_values):
        for j in range(num_steps):
            packed_tensor[i] |= uint8tensor[unpacked_idx] << (bits * j)
            unpacked_idx += 1
    return packed_tensor

# Unpacking weights from int8
def unpack_weights(uint8tensor, bits):
    num_values = uint8tensor.shape[0] * 8 // bits

    num_steps = 8 // bits
    unpacked_tensor = torch.zeros((num_values), dtype=torch.uint8)
    unpacked_idx = 0
    mask = 2 ** bits - 1

    for i in range(uint8tensor.shape[0]):
        for j in range(num_steps):
            unpacked_tensor[unpacked_idx] |= uint8tensor[i] >> (bits * j)
            unpacked_idx += 1

    unpacked_tensor &= mask
    return unpacked_tensor