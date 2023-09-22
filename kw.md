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

主要是先把wide-and-deep的模型实现了，再基于此实现各自负责的模型压缩技术

大家先调研一下联邦学习中自适应压缩（即根据一些条件来在训练过程中动态调整压缩的程度）的论文， 每位同学还是着重调研一下自己所负责的压缩技术
同时也请尽可能地多留意双向压缩（客户端的上传和下载都进行压缩）的论文以及糅合了多种压缩技术的混合技术（例如，先量化再稀疏化）的论文
调研之后，我们再一起讨论，然后设计并实现出我们自己的算法，与上周的几个baseline进行对比

    每次迭代都需要向服务器发送模型的更新。这可能会消耗大量的通信带宽, 不同的客户端和不同的网络条件可能需要不同级别的压缩，这导致了自适应压缩策略的需求。
    自适应压缩旨在根据当前的网络状况、客户端状态和模型的更新特性动态地调整压缩率。通过自适应地选择压缩算法或参数，可以确保既不牺牲模型的训练质量，又可以有效地减少通信开销。

    错误补偿：为了减少由于压缩引入的误差，一些方法在客户端上存储压缩误差，并在下次迭代中对其进行补偿。
    梯度压缩：如梯度截断、梯度量化、稀疏更新等方法，可以减少需要传输的梯度的大小或数量。
    编码方法：使用如Huffman编码或其他有效的编码方式，根据梯度或更新的分布对其进行编码。

自适应决策：
根据网络带宽、延迟、客户端计算能力和模型更新的重要性，动态地选择最佳的压缩策略或参数。
这可能涉及到连续地监控这些参数，并使用启发式方法或优化算法来做决策。    