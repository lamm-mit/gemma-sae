import torch

from gemma4_sae.sae import BatchTopKSAE


def test_batch_topk_enforces_average_l0() -> None:
    torch.manual_seed(0)
    sae = BatchTopKSAE(d_model=6, n_features=12, target_l0=3)
    with torch.no_grad():
        sae.encoder.bias.fill_(10.0)
    x = torch.randn(5, 6)
    output = sae.train()(x, use_threshold=False)
    assert int((output.features > 0).sum()) == 5 * 3
    assert len(output.selected_indices) == 5 * 3
    assert output.reconstruction.shape == x.shape


def test_thresholded_inference_uses_recorded_threshold() -> None:
    sae = BatchTopKSAE(d_model=4, n_features=8, target_l0=2, threshold_ema_decay=0.5)
    sae.update_inference_threshold_(torch.tensor(1.25))
    assert sae.inference_threshold.item() == 1.25
    assert bool(sae.threshold_initialized)

    with torch.no_grad():
        sae.encoder.weight.zero_()
        sae.encoder.bias.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0, 0.0, 0.0, 0.0, 0.0]))
    features, _, _ = sae.eval().encode(torch.zeros(1, 4), use_threshold=True)
    assert features.count_nonzero().item() == 2


def test_decoder_columns_remain_unit_norm() -> None:
    sae = BatchTopKSAE(d_model=5, n_features=9, target_l0=2)
    with torch.no_grad():
        sae.decoder.weight.mul_(3.7)
        sae.normalize_decoder_()
    norms = sae.decoder.weight.norm(dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_dead_feature_resampling_changes_requested_columns() -> None:
    torch.manual_seed(1)
    sae = BatchTopKSAE(d_model=4, n_features=8, target_l0=2)
    before = sae.decoder.weight[:, [1, 5]].clone()
    residual = torch.randn(6, 4)
    count = sae.resample_dead_features_(torch.tensor([1, 5]), residual)
    assert count == 2
    assert not torch.equal(before, sae.decoder.weight[:, [1, 5]])
    norms = sae.decoder.weight[:, [1, 5]].norm(dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms))

