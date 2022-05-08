from typing import Union

import torch


#
#
#  -------- unpad -----------
#
def unpad(padded: Union[list, torch.Tensor], length: Union[list, torch.Tensor]) -> Union[list, torch.Tensor]:
    """Convert the given packaged sequence into a list of vectors."""
    output = []
    for v, n in zip(padded, length):
        output.append(v[:n])
    return output

