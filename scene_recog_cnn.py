"""
Main train and test function. You could call the subroutines from the `Subroutines` sections. Please kindly follow the 
train and test guideline.

`train` function should contain operations including loading training images, computing features, constructing models, training 
the models, computing accuracy, saving the trained model, etc
`test` function should contain operations including loading test images, loading pre-trained model, doing the test, 
computing accuracy, etc.
"""
import torch
from torch.utils.data import DataLoader

from dataset import MyDataset
from learn import worker
from model import build_model


DEVICE = ('mps' if torch.backends.mps.is_available() else 'cpu')


def train(train_data_dir, model_dir, 
          batch_size=32, epochs=50, lr=0.001, patience=10, dropout_rate=0.1, num_classes=15):
    """Main training model.

    Arguments:
        train_data_dir (str):   The directory of training data
        model_dir (str):        The directory of the saved model.
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        train_accuracy (float): The training accuracy.
    """
    return worker(data_path=train_data_dir, model_path=model_dir, 
                  batch_size=batch_size, epochs=epochs, lr=lr, patience=patience, dropout_rate=dropout_rate, num_classes=num_classes)


def test(test_data_dir, model_dir, num_classes=15):
    """Main testing model.

    Arguments:
        test_data_dir (str):    The `test_data_dir` is blind to you. But this directory will have the same folder structure as the `train_data_dir`.
                                You could reuse the snippets of loading data in `train` function
        model_dir (str):        The directory of the saved model. You should load your pretrained model for testing
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        test_accuracy (float): The testing accuracy.
    """
    dataset = MyDataset(path=test_data_dir, is_train=False)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = build_model(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    total_samples, total_correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            total_correct += (predicted + 1 == target).sum().item()
    test_acc = total_correct / total_samples
    return test_acc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train','test'])
    parser.add_argument('--train_data_dir', default='./data/train/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./data/test/', help='the directory of testing data')
    parser.add_argument('--model_dir', default='model.pkl', help='the pre-trained model')
    opt = parser.parse_args()

    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir)
        print(testing_accuracy)
