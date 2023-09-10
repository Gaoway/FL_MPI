two quantize methods

8bit quantize
if param.dtype == torch.float32:
    param_i = (param*127*2).to(torch.int16)
    param_f = param_i.to(torch.float16)/(127.0*2)
    params_dict[name] = param_f

8bit quantize
if param.dtype == torch.float32:
    min_val = param.min()
    max_val = param.max()
    scale_8 = 255. / (max_val - min_val)
    zero_point_8 = -min_val * scale_8
    param_8 = (param * scale_8 + zero_point_8).clamp(0, 255).byte()
                    
    param_dq8 = (param_8.float() - zero_point_8) / scale_8
    params_dict[name]   = param_dq8   

8bit quantize
if param.dtype == torch.float32:
    min_val = param.min()
    max_val = param.max()
    scale_8 = 255. / (max_val - min_val)
    zero_point_8 = -min_val * scale_8
    param_8 = (param * scale_8 + zero_point_8+0.5).clamp(0, 255).byte()
                    
    param_dq8 = (param_8.float() - zero_point_8) / scale_8
    params_dict[name]   = param_dq8   