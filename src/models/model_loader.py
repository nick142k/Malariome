"""Build or load models by name. Transfer-learning models are added in a later phase."""


def build_model(name, cfg):
    overrides = cfg.get("model", {}).get(name, {})          # optional config.yaml section: model: {custom_cnn: {...}}
    if name == "custom_cnn":
        from src.models.custom_cnn import build_custom_cnn
        return build_custom_cnn(image_size=cfg["data"]["image_size"], **overrides)
    raise NotImplementedError(f"Model '{name}' is not implemented yet (available: custom_cnn).")


def load_model(path):
    import keras
    return keras.saving.load_model(str(path))
