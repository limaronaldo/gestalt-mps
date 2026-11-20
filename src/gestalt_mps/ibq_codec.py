"""Interface reserved by the implementation plan."""


class IBQCodec:
    @classmethod
    def from_checkpoint(cls, checkpoint, device):
        raise NotImplementedError

    def encode_image(self, path, *, lat_h, lat_w):
        raise NotImplementedError

    def decode_tokens(self, codes, *, lat_h, lat_w):
        raise NotImplementedError
