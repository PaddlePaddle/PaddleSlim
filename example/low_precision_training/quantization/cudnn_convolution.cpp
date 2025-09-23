#include <paddle/extension.h>
#include <vector>

namespace paddle{
    namespace experimental{
        extern PADDLE_API void conv2d_grad(const paddle::Tensor& input, const paddle::Tensor& filter,
                                 const paddle::Tensor& out_grad, const std::vector<int>& strides,
                                 const std::vector<int>& paddings, const std::string& padding_algorithm,
                                 const std::vector<int>& dilations, int groups,
                                 const std::string& data_format, paddle::Tensor* input_grad,
                                 paddle::Tensor* filter_grad);
    }
}

std::vector<paddle::Tensor> convolution_bwd(
    const paddle::Tensor& input,
    const paddle::Tensor& weight,
    const paddle::Tensor& grad_output,
    std::vector<int> stride,
    std::vector<int> padding,
    std::vector<int> dilation,
    int groups) {

    paddle::Tensor input_grad, weight_grad;

    paddle::experimental::conv2d_grad(input, weight, grad_output, stride, padding, "EXPLICIT", dilation, groups, "NCHW", &input_grad, &weight_grad);

    // return {input_grad, weight_grad};
    return {input_grad, weight_grad};
}



PYBIND11_MODULE(cudnn_convolution_custom, m) {
  // 将加法函数绑定至 python 端
  m.def("convolution_bwd", &convolution_bwd, "convolution_bwd");
}


// PD_BUILD_OP(convolution_bwd)
//     .Inputs({"input", "weight", "grad_output", paddle::Vec("stride"), paddle::Vec("padding"), paddle::Vec("dilation"), "groups"})
//     .Outputs({"input_grad", "weight_grad"})
//     .SetKernelFn(PD_KERNEL(Convolution_bwd));


// namespace paddle{
//     namespace experimental{
//         extern PADDLE_API void conv2d_grad(const paddle::Tensor& input, const paddle::Tensor& filter,
//                                  const paddle::Tensor& out_grad, const std::vector<int>& strides,
//                                  const std::vector<int>& paddings, const std::string& padding_algorithm,
//                                  const std::vector<int>& dilations, int groups,
//                                  const std::string& data_format, paddle::Tensor* input_grad,
//                                  paddle::Tensor* filter_grad);
//     }
// }

// std::vector<paddle::Tensor> convolution(
//     const paddle::Tensor& input,
//     const paddle::Tensor& weight,
//     std::vector<int> stride,
//     std::vector<int> padding,
//     std::vector<int> dilation,
//     int groups) {

//     paddle::Tensor output = paddle::experimental::conv2d(input, weight, stride, padding, "EXPLICIT", dilation, groups);

//     return {output};
// }



// PD_BUILD_OP(cudnn_convolution)
//     .Inputs({"input", "weight", paddle::Vec("stride"), paddle::Vec("padding"), paddle::Vec("dilation"), "groups"})
//     .Outputs({"output"})
//     .SetKernelFn(PD_KERNEL(convolution));


// paddle::Tensor convolution_backward_weight(
//     const paddle::Tensor& input,
//     const paddle::Tensor& weight,
//     const paddle::Tensor& grad_output,
//     std::vector<int> stride,
//     std::vector<int> padding,
//     std::vector<int> dilation,
//     int groups) {

//     paddle::Tensor input_grad, weight_grad;

//     conv2d_grad(input, weight, grad_output, stride, padding, "EXPLICIT", dilation, groups, "NCHW", &input_grad, &weight_grad);

//     return weight_grad;
// }

// paddle::Tensor convolution_backward_input(
//     const paddle::Tensor& input,
//     const paddle::Tensor& weight,
//     const paddle::Tensor& grad_output,
//     std::vector<int> stride,
//     std::vector<int> padding,
//     std::vector<int> dilation,
//     int groups) {

//     paddle::Tensor input_grad, weight_grad;

//     conv2d_grad(input, weight, grad_output, stride, padding, "EXPLICIT", dilation, groups, "NCHW", &input_grad, &weight_grad);

//     return input_grad;
// }

// PD_BUILD_OP(cudnn_convolution)
//     .Inputs({"input", "weight", paddle::Vec("stride"), paddle::Vec("padding"), paddle::Vec("dilation"), "groups"})
//     .Outputs({"output"})
//     .SetKernelFn(PD_KERNEL(convolution));

// PD_BUILD_GRAD_OP(cudnn_convolution)
//     .Inputs({"input", "weight", paddle::Grad("output"), paddle::Vec("padding"), paddle::Vec("dilation"), "groups"})
//     .Outputs({paddle::Grad("weight")})
//     .SetKernelFn(PD_KERNEL(convolution_backward_weight));

// PD_BUILD_GRAD_OP(cudnn_convolution)
//     .Inputs({"input", "weight", paddle::Grad("output"), paddle::Vec("padding"), paddle::Vec("dilation"), "groups"})
//     .Outputs({paddle::Grad("input")})
//     .SetKernelFn(PD_KERNEL(convolution_backward_input));

