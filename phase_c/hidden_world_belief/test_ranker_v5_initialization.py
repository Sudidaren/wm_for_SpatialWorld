import torch

from phase_c.hidden_world_belief.train_object_ranker_v5 import (
    BilinearTypeRanker,
    initialize_from_checkpoint,
)


def test_v5_initialization_preserves_shared_token_rows():
    old_vocab = {
        "target": {"<unk>": 0, "Egg": 1},
        "room": {"<unk>": 0, "kitchen": 1},
        "receptacle": {"<unk>": 0, "Fridge": 1},
    }
    old = BilinearTypeRanker(2, 2, 2)
    payload = {"architecture": "bilinear_type_v5", "vocabs": old_vocab,
               "model": old.state_dict()}
    new_vocab = {
        "target": {"<unk>": 0, "Apple": 1, "Egg": 2},
        "room": {"<unk>": 0, "bedroom": 1, "kitchen": 2},
        "receptacle": {"<unk>": 0, "Cabinet": 1, "Fridge": 2},
    }
    new = BilinearTypeRanker(3, 3, 3)
    initialize_from_checkpoint(new, new_vocab, payload)
    assert torch.equal(new.target.weight[2], old.target.weight[1])
    assert torch.equal(new.room.weight[2], old.room.weight[1])
    assert torch.equal(new.receptacle.weight[2], old.receptacle.weight[1])
    assert torch.equal(new.bias[2], old.bias[1])
