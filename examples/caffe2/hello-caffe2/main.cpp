#include <iostream>
#include <ATen/ATen.h>
#include <c10/util/irange.h>
#include <caffe2/core/tensor.h>
#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>

using namespace std;

int main(int argc, char** argv) {
  caffe2::Workspace workspace;
  caffe2::NetDef net;

  auto at_tensor_a = at::ones({5, 5}, at::dtype(at::kFloat));
  auto at_tensor_b = at::ones({5, 5}, at::dtype(at::kFloat));
  auto at_tensor_c = at::ones({5, 5}, at::dtype(at::kFloat));

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto* c2_tensor_a = BlobSetTensor(workspace.CreateBlob("a"), caffe2::Tensor(at_tensor_a));
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  auto* c2_tensor_b = BlobSetTensor(workspace.CreateBlob("b"), caffe2::Tensor(at_tensor_b));

  // Test Alias
  {
    caffe2::Tensor c2_tensor_from_aten(at_tensor_c);
    BlobSetTensor(workspace.CreateBlob("c"), c2_tensor_from_aten.Alias());
  }

  {
    auto op = net.add_op();
    op->set_type("Sum");
    op->add_input("a");
    op->add_input("b");
    op->add_input("c");
    op->add_output("d");
  }

  workspace.RunNetOnce(net);

  auto result = XBlobGetMutableTensor(workspace.CreateBlob("d"), {5, 5}, at::kCPU);

  auto it = result.data<float>();
  at::Tensor at_result(result);
  cout << at_result << endl;
}
