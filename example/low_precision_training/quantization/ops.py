import paddle
import paddle.nn.functional as F
import cudnn_convolution_custom

class forward_quantizer(paddle.autograd.PyLayer):
    # qat
    @staticmethod
    def forward(ctx, tensor, q):
        return q(tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class quantized_linear_for_training(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input, weight, bias,
                hw, wfq, wbq,
                hi, ifq, ibq,
                hg, gxq, gwq):
        
        # weight quantization
        # print(weight.shape)
        # assert 0==1

        # print("linear1weight", weight.stop_gradient)
        # print("linear1input", input.stop_gradient)
        # if bias is not None:
        #     print("linear1bias", bias.stop_gradient)

        q_weight_forward = wfq(weight)

        if hw:
            q_weight_backward = wbq(weight)
        else:
            q_weight_backward = q_weight_forward

        # input quantization
        q_input_forward = ifq(input)
        if hi:
            q_input_backward = ibq(input)
        else:
            q_input_backward = q_input_forward

        # print("linear_q_weight_backward", q_weight_backward.stop_gradient)
        # print("linear_q_input_backward", q_input_backward.stop_gradient)
        # if bias is not None:
        #     print("linear_bias", bias.stop_gradient)

        # save tensors for backward
        ctx.save_for_backward(q_weight_backward, bias, q_input_backward)
        if bias is None:
            b_grad = None
        else:
            b_grad = bias.stop_gradient
        ctx.saved = hg, gxq, gwq, weight.stop_gradient, input.stop_gradient, b_grad

        # forward linear operation
        # print("q_input_forward: ", q_input_forward.shape)
        # print("q_weight_forward: ", q_weight_forward.shape)  # [512, 1000]反的
        # output = paddle.matmul(q_input_forward, q_weight_forward, transpose_y=True)   ####
        output = paddle.matmul(q_input_forward, q_weight_forward)
        # print("output: ", output.shape)
        # assert 0==1
        if bias is not None:
            output += paddle.unsqueeze(bias, axis=0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hg, gxq, gwq, w_grad, i_grad, b_grad = ctx.saved
        weight, bias, input = ctx.saved_tensor()
        # print("grad_output \n", grad_output)
        # print("weight \n", weight)
        # print("bias \n", bias)
        # print("input \n", input)
        # assert 0==1

        # print("linear2weight", weight.stop_gradient)
        # print("linear2input", input.stop_gradient)
        # if bias is not None:
        #     print("linear2bias", bias.stop_gradient)

        del ctx.saved

        # dual path quantization for grad_output
        q_grad_output_for_x = gxq(grad_output)
        if hg:
            q_grad_output_for_w = gwq(grad_output)
        else:
            q_grad_output_for_w = q_grad_output_for_x

        # grad calculation
        grad_input = grad_weight = grad_bias = None

        # print("linear2q_grad_output_for_w", q_grad_output_for_w.stop_gradient)
        if not i_grad:
            # print("q_grad_output_for_x: ", q_grad_output_for_x.shape)   # [64, 1000]
            # print("weight: ", weight.shape)  # [512, 1000]
            # grad_input = paddle.matmul(q_grad_output_for_x, weight)
            grad_input = paddle.matmul(q_grad_output_for_x, weight, transpose_y=True)
            # print("grad_input: ", grad_input.shape)
        if not w_grad:
            # print("q_grad_output_for_w: ", q_grad_output_for_w.shape) # [64, 1000]
            # print("input: ", input.shape) #  [64, 512]
            grad_weight = paddle.matmul(q_grad_output_for_w, input, transpose_x=True)
        if bias is not None and not b_grad:
            grad_bias = paddle.sum(grad_output, axis=0)
            return grad_input, grad_weight, grad_bias

        return grad_input, grad_weight


class quantized_conv2d_for_training(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups,
                hw, wfq, wbq,
                hi, ifq, ibq,
                hg, gxq, gwq):
        
        # print("1weight", weight.stop_gradient)
        # print("1input", input.stop_gradient)
        # if bias is not None:
        #     print("1bias", bias.stop_gradient)

        # weight quantization
        q_weight_forward = wfq(weight)

        if hw:
            q_weight_backward = wbq(weight)
        else:
            q_weight_backward = q_weight_forward

        # input quantization
        q_input_forward = ifq(input)
        if hi:
            q_input_backward = ibq(input)
        else:
            q_input_backward = q_input_forward

        # print("q_weight_backward: ", q_weight_backward.stop_gradient)
        # print("q_weight_forward: ", q_weight_forward.stop_gradient)
        # print("q_input_backward", q_input_backward.stop_gradient)
        # print("q_input_forward", q_input_forward.stop_gradient)
        # if bias is not None:
        #     print("bias", bias.stop_gradient)

        # print("q_weight_backward", q_weight_backward.stop_gradient)
        # print("q_input_backward", q_input_backward.stop_gradient)
        # if bias is not None:
        #     print("bias", bias.stop_gradient)


        # save tensors and parameters for backward
        # q_weight_backward.stop_gradient = weight.stop_gradient
        # q_input_backward.stop_gradient = input.stop_gradient
        ctx.save_for_backward(q_weight_backward, bias, q_input_backward)
        if bias is None:
            b_grad = None
        else:
            b_grad = bias.stop_gradient
        ctx.saved =  stride, padding, dilation, groups, hg, gxq, gwq, weight.stop_gradient, input.stop_gradient, b_grad
        # ctx.saved = stride, padding, dilation, groups, hg, gxq, gwq

        # output = cudnn_convolution.convolution(q_input_forward, q_weight_forward, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        output = paddle.nn.functional.conv2d(q_input_forward, q_weight_forward, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        stride, padding, dilation, groups, hg, gxq, gwq, w_grad, i_grad, b_grad = ctx.saved
        # q_weight_backward, bias, q_input_backward = ctx.saved_tensor()
        weight, bias, input = ctx.saved_tensor()

        # print("2weight", weight.stop_gradient)
        # print("2input", input.stop_gradient)
        # if bias is not None:
        #     print("2bias", bias.stop_gradient)

        del ctx.saved

        # dual path quantization for grad_output
        q_grad_output_for_x = gxq(grad_output)
        if hg:
            q_grad_output_for_w = gwq(grad_output)
        else:
            q_grad_output_for_w = q_grad_output_for_x

        # grad back-propagation
        grad_input = grad_weight = grad_bias = None
        if not i_grad:
            grad_input = cudnn_convolution_custom.convolution_bwd(input, weight, q_grad_output_for_x, stride, [padding, padding], dilation, 1)[0]

        if not w_grad:
            grad_weight = cudnn_convolution_custom.convolution_bwd(input, weight, q_grad_output_for_w, stride, [padding, padding], dilation, 1)[1]

        if bias is not None and not b_grad:
            grad_bias = paddle.sum(grad_output, axis=[0, 2, 3])
            return grad_input, grad_weight, grad_bias

        # print("grad_input", grad_input)
        # print("weight_input", grad_input)

        return grad_input, grad_weight
