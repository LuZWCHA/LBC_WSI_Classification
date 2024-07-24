import os
from typing import Any
from detectron2.checkpoint import DetectionCheckpointer
import torch

# not used
class EMADetectionCheckpointer(DetectionCheckpointer):
    
    def __init__(self, model, swa_model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(model, save_dir, save_to_disk=save_to_disk, **checkpointables)
        self.swa_model = swa_model
        
        
    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        data["swa_model"] = self.swa_model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            # pyre-fixme[6]: For 2nd param expected `Union[PathLike[typing.Any],
            #  IO[bytes], str, BinaryIO]` but got `Union[IO[bytes], IO[str]]`.
            torch.save(data, f)
        self.tag_last_checkpoint(basename)
        
    def load(self, path, *args, **kwargs):
        return super().load(path, *args, **kwargs)
        
    