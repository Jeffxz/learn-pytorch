#include <iostream>
#include <torch/torch.h>

using namespace torch;
using namespace std;

int main(int argc, char** argv)
{
    Tensor a = torch::ones(3);
    cout << a << endl;
    a[2] = 2.0;
    cout << a << endl;

    Tensor points = torch::tensor({4, 1, 5, 3, 2, 1});
    cout << points << endl;

    points = torch::tensor({{4, 1}, {5, 3}, {2, 1}});
    cout << points << endl;
    cout << points[0] << endl;
    cout << points.storage() << endl;

    Tensor second_points = points[1];
    cout << "storage_offset: " << second_points.storage_offset() << endl;
    cout << "stride: " << second_points.stride(0) << endl;
    cout << points.t() << endl;

    cout << "CUDA available: " << torch::cuda::is_available() << endl;

    cout << "requires grad: " << points.requires_grad() << endl;
    cout << "is inference: " << points.is_inference() << endl;

    return 0;
}