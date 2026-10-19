from configs.mnist_baseline import MODEL_CONFIG, TRAIN_CONFIG
from src.model import ImageTokenTransformer, ModelConfig
from src.train import train_model


def main():
    from data import get_dataloaders

    train_loader, val_loader = get_dataloaders()

    model_config = ModelConfig(**MODEL_CONFIG)
    model = ImageTokenTransformer(model_config)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TRAIN_CONFIG,
    )


if __name__ == "__main__":
    main()
