from mil_framework.dataloader import make_data_iter, Batch
from mil_framework.utils import return_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import time


def validate_on_data(model, data, batch_size, use_cuda, features_dim):
    valid_iter = make_data_iter(
        dataset=data,
        batch_size=batch_size,
        shuffle=False,
        train=False)

    preds = []
    true = []

    model.eval()
    with torch.no_grad():
        for valid_batch in iter(valid_iter):
            batch = Batch(
                is_train=False,
                torch_batch=valid_batch,
                use_cuda=use_cuda,
                features_dim=features_dim,
            )
            _, output = model.get_metrics_for_batch(batch=batch)
            preds.append(output.detach().cpu())
            true.append(batch.target.detach().cpu())

    bas, auc, thres = return_metrics(true, preds)
    return bas, auc, thres


class TrainManager:
    def __init__(self, model, config):

        self.model_dir = config.model_dir
        self.name_model = config.name_model
        self.epochs = config.epochs
        self.features_dim = config.features_dim
        self.use_cuda = config.use_cuda
        self.batch_size = config.batch_size
        self.eval_batch_size = config.batch_size
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.8, verbose=True)
        self.new_best = 0.0
        self.is_best = (lambda score: score > self.new_best)

    def _save_checkpoint(self) -> None:
        model_path = "{}/{}.ckpt".format(self.model_dir, self.name_model)
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
        }
        torch.save(state, model_path)

    def train_and_validate(self, train_data, valid_data):
        train_iter = make_data_iter(
            train_data,
            batch_size=self.batch_size,
            train=True,
            shuffle=True)

        for epoch_no in range(self.epochs):

            self.model.train()
            start = time.time()
            i = 0
            preds = []
            true = []

            for batch in iter(train_iter):
                i += 1
                batch = Batch(
                    is_train=True,
                    torch_batch=batch,
                    use_cuda=self.use_cuda,
                    features_dim=self.features_dim,
                )

                lympho_loss, output = self._train_batch(
                    batch
                )
                preds.append(output.detach().cpu())
                true.append(batch.target.detach().cpu())

            train_bac, train_auc, thresh = return_metrics(true, preds)
            ensemble = validate_on_data(
                model=self.model,
                data=valid_data,
                batch_size=self.eval_batch_size,
                use_cuda=self.use_cuda,
                features_dim=self.features_dim)

            balanc_acc = ensemble[0]

            bst = (balanc_acc + train_bac) / 2
            self.scheduler.step(bst)
            if self.is_best(bst):
                self.new_best = bst
                print("Yes! New best validation result!")
                self._save_checkpoint()
            self.model.train()
            print(f"Epoch {epoch_no}/{self.epochs} , Time : {round(time.time() - start, 3)}")
            print(
                f"Total Loss : {round(lympho_loss, 4)}, Train_BACC: {round(train_bac, 3)}, Train_AUC : {round(train_auc, 3)}, Threshold : {round(thresh, 3)}")
            print(
                f"Val_BACC : {round(ensemble[0], 3)}, Val AUC : {round(ensemble[1], 3)}, Threshold : {round(ensemble[2], 3)}")
            print()
            print()

    def _train_batch(self, batch):
        lympho_loss, final_output = self.model.get_metrics_for_batch(
            batch=batch)
        lympho_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return lympho_loss.item(), final_output


def infer(model, test_data, cfg):
    test_iter = make_data_iter(
        test_data,
        batch_size=16)
    model.eval()

    preds = []
    ids = []
    with torch.no_grad():
        for batch in iter(test_iter):
            batch = Batch(
              is_train=False,
              torch_batch=batch,
              use_cuda=True,
              features_dim=cfg.features_dim)
            _, output = model.get_metrics_for_batch(batch)
            preds.append(output.detach().cpu())
            ids += batch.id

    preds = torch.cat(preds, dim=0).numpy()
    return preds, ids
