// from https://discuss.pytorch.org/t/how-to-convert-an-opencv-image-into-libtorch-tensor/90818/2

#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

string get_image_type(const cv::Mat& img, bool more_info=true) 
{
    string r;
    int type = img.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');
   
    if (more_info)
        cout << "depth: " << img.depth() << " channels: " << img.channels() << endl;

    return r;
}

void show_image(cv::Mat& img, string title)
{
    string image_type = get_image_type(img);
    cv::namedWindow(title + " type:" + image_type, cv::WINDOW_NORMAL); // Create a window for display.
    cv::imshow(title, img);
    cv::waitKey(0);
}

auto transpose(at::Tensor tensor, c10::IntArrayRef dims = { 0, 3, 1, 2 })
{
    cout << "############### transpose ############" << endl;
    cout << "shape before : " << tensor.sizes() << endl;
    tensor = tensor.permute(dims);
    cout << "shape after : " << tensor.sizes() << endl;
    cout << "######################################" << endl;
    return tensor;
}

auto ToTensor(cv::Mat img, bool show_output = false, bool unsqueeze=false, int unsqueeze_dim = 0)
{
    cout << "image shape: " << img.size() << endl;
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kByte);

    if (unsqueeze)
    {
        tensor_image.unsqueeze_(unsqueeze_dim);
        cout << "tensors new shape: " << tensor_image.sizes() << endl;
    }
    
    if (show_output)
    {
        cout << tensor_image.slice(2, 0, 1) << endl;
    }
    cout << "tenor shape: " << tensor_image.sizes() << endl;
    return tensor_image;
}

auto ToInput(at::Tensor tensor_image)
{
    // Create a vector of inputs.
    return vector<torch::jit::IValue>{tensor_image};
}

auto ToCvImage(at::Tensor tensor)
{
    int width = tensor.sizes()[0];
    int height = tensor.sizes()[1];
    try
    {
        cv::Mat output_mat(cv::Size{ height, width }, CV_8UC3, tensor.data_ptr<uchar>());
        
        // show_image(output_mat, "converted image from tensor");
        return output_mat.clone();
    }
    catch (const c10::Error& e)
    {
        cout << "an error has occured : " << e.msg() << endl;
    }
    return cv::Mat(height, width, CV_8UC3);
}

int main(int argc, char** argv) {
    if (argc != 3)
    {
        cout << "usage: test-opencv <Input Image Path> <Model Path>";
        return -1;
    }
    Mat image = imread(argv[1], IMREAD_COLOR);
    auto tensor = ToTensor(image);

    auto cv_img = ToCvImage(tensor);
    // convert the tensor into float and scale it 
    tensor = tensor.toType(c10::kFloat).div(255);
    // swap axis 
    tensor = transpose(tensor, {(2),(0),(1)});
    tensor.unsqueeze_(0);

    auto input_to_net = ToInput(tensor);

    torch::jit::script::Module module;

    try 
    {
        string model_path = argv[2];

        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(model_path);
    
        // Execute the model and turn its output into a tensor.
        at::Tensor output = module.forward(input_to_net).toTensor();
        //sizes() gives shape. 
        cout << output.sizes() << endl;
        cout << "Output: " << output[0] << endl;
        cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    
    }
    catch (const c10::Error& e) 
    {
        cerr << "error loading the model\n" <<e.msg();
        return -1;
    }

    return 0;
}