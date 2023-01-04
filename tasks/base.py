from torch.utils.data import DataLoader

class Task:
    def __init__(self) -> None:
        pass

    def get_train_loader(self) -> DataLoader:
        raise NotImplementedError

    def get_train_data_len() -> int:
        raise NotImplementedError

    def loss_and_step(model, batch_data) -> float:
        raise NotImplementedError

    def eval_model_with_log(model, writer, lr_now) -> None:
        raise NotImplementedError    
        
    