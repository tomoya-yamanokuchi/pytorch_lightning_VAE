from .variational_autoencoder.LitVariationalAutoencoder import LitVariationalAutoencoder

class ModelFactory:
    def create(self, name: str, **kwargs):
        if    name == "vae"      : return LitVariationalAutoencoder(**kwargs)
        else                     : raise NotImplementedError()
