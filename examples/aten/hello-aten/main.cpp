/*
 * aten sample code for basic Tensor initialization and operations
 */

#include <iostream>
#include <ATen/ATen.h>

using namespace std;
using namespace at;

int main(int argc, char** argv)
{
    Tensor a = tensor(2);
    cout << a << endl << endl;
    cout << a.strides() << endl << endl;
    cout << a + 1 << endl << endl;

    Tensor b = ones(5);
    cout << b << endl << endl;
    cout << b.strides() << endl << endl;
    cout << b + 2 << endl << endl;

    Tensor c = ones({2, 2}, kInt);
    Tensor d = randn({2, 2});
    cout << c << endl;
    cout << "strides: " << c.strides() << "  sizes: " << c.sizes() << endl;
    cout << c + d.to(kInt) << endl << endl;

    Tensor e = tensor({10, -1, 0, 1, -10});
    cout << e << endl << endl;

    return 0;
}