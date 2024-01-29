#include <torch/torch.h>
#include <QApplication>
#include <QDebug>
#include <QImage>

using namespace torch;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 2) {
        qDebug() << "Usage: load-image <image>";
        return -1;
    }
    QImage image = QImage(argv[1]);
    int width = image.width();
    int height = image.height();
    qDebug() << image << width << height;

    Tensor imageTensor = torch::from_blob(image.bits(), {image.height(), image.width(), 3}, at::kByte);
    cout << "tensor shape: " << imageTensor.sizes() << endl;
    imageTensor = imageTensor.permute({2, 0, 1});
    cout << "tensor shape: " << imageTensor.sizes() << endl;

    return 0;
}