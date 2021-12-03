import torch

from utils import order_tensor

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def test_model(net, data_generator, loss_fn, transform=None):
    """Function to easily test model on specified dataset"""
    batch_loss, batch_steps = 0.0, 0
    correct_pred, total_pred = 0, 0

    net = net.to(device)
    net.eval()
    for batch_id, (data, label) in enumerate(data_generator):
        data, label = data.to(device), label.long().to(device)

        if data_generator.dataset.transform is None:
            data = order_tensor(data)
        if transform is not None:
            data = transform(data.cuda())

        output = net(data)
        batch_loss += loss_fn(output, label).item()
        batch_steps += 1

        # indices where probability is maximum
        _, pred_label = torch.max(output, 1)
        correct_pred += (pred_label == label).sum().item()
        total_pred += label.shape[0]

    # average loss/acc across ALL batches
    # i.e. ACROSS specified dataset
    avg_loss = batch_loss / batch_steps
    avg_acc = correct_pred / total_pred

    return avg_loss, avg_acc
