import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.utils.utils import current_time, write_json
from src.utils.storage import Storage

def report(model, dataset, config, N=3):
    print("REPORTING")
    plt.clf()
    time = current_time()
    path = config["REPORT_DIR"]
    path = f"{path}/{time}"
    os.mkdir(path)

    ld = DataLoader(dataset, batch_size=1)
    device = model.current_device()
    model.evaluate(ld, device)
    metrics = model.observer_metrics()
    write_json(metrics, path+"/eval")
    idxs = np.random.randint(len(dataset), size=N)
    model = model.cpu()
    model.eval()

    figure, ax = plt.subplots(nrows=N, ncols=3, figsize=(10,10))
    figure.subplots_adjust(wspace=0, hspace=0)
    for row,idx in enumerate(idxs):
      img, mask = dataset[idx]
      img_real, _ = dataset.get(idx)
      logits = model(img.unsqueeze(0)).squeeze()
      y_pred = logits.squeeze(0).argmax(axis=0)
      ax[row, 0].imshow(img_real)
      ax[row, 1].imshow(mask)
      ax[row, 2].imshow(y_pred)
      for k in range(3):
        ax[row, k].set_xticklabels([])
        ax[row, k].set_yticklabels([])
    img_path = path + "/samples.png"
    figure.savefig(img_path)
    plt.clf()

    metrics = Storage.get().get_metrics()
    epochs      = [m["epoch"] for m in metrics]
    train_loss  = [m["train loss"] for m in metrics]
    val_loss    = [m["valid loss"] for m in metrics]
    accuracy    = [m["accuracy"] for m in metrics]
    miou        = [m["mIoU"] for m in metrics]

    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label = "valid loss")
    plt.legend(loc="best")
    plt.savefig(path + "/train_val.png")
    plt.clf()

    plt.plot(epochs, accuracy, label="accuracy")
    plt.plot(epochs, miou, label = "mIoU")
    plt.legend(loc="best")
    plt.savefig(path + "/metrics.png")
    plt.clf()

    write_json(config["HIPER_PARAMS"], path+"/config")
    torch.save(model.state_dict(), path + "/model")

