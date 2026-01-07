from .TransformerModel import TransformerModel

def load_model(config, vocab_size):
    return TransformerModel(config, vocab_size).to(config.device)