import unittest

import torch
import torch.nn.functional as F

from models import DPSR, QDPSR, FSRCNN
from utils.metrics import MixedLoss
from utils.trainer import train_epoch


class BilinearSkipTest(unittest.TestCase):
    def test_switch_and_legacy_reconstruction(self):
        torch.manual_seed(7)
        for cls in (DPSR, QDPSR):
            for scale in (2, 3, 4):
                for in_dim in (1, 3):
                    model = cls(scale, in_dim, 8, num_blocks=1, subnet_channels=4)
                    if cls is QDPSR and in_dim == 1:
                        model.set_quantization_enabled(False)
                    # Nonzero residuals ensure this also exercises the learned path.
                    tail = model.tail if cls is DPSR else model.tail.conv
                    torch.nn.init.normal_(tail.weight, std=0.03)
                    lr = torch.randint(0, 256, (1, in_dim, 7, 9)).float()
                    x = lr / 255.0
                    base = F.interpolate(lr, scale_factor=scale, mode='bilinear',
                                         align_corners=False).round().clamp(0, 255)
                    for channels in model.supported_channels:
                        residual = model(x, channels, use_bilinear=False)
                        sr = model(x, channels)
                        torch.testing.assert_close(sr * 255, residual * 255 + base,
                                                   rtol=1e-6, atol=5e-5)
                        model.use_bilinear = False
                        torch.testing.assert_close(model(x, channels), residual)
                        torch.testing.assert_close(model(x, channels, True), sr)
                        self.assertFalse(model.use_bilinear)
                        model.use_bilinear = True
                    restored = cls(scale, in_dim, 8, num_blocks=1, subnet_channels=4,
                                   use_bilinear=False)
                    restored.load_state_dict(model.state_dict(), strict=True)
                    self.assertEqual(set(restored.state_dict()), set(model.state_dict()))

    def test_loss_and_parameter_gradients(self):
        torch.manual_seed(9)
        model = DPSR(2, 3, 8, num_blocks=1, subnet_channels=4)
        torch.nn.init.normal_(model.tail.weight, std=0.03)
        lr = torch.randint(0, 256, (1, 3, 7, 9)).float()
        hr = torch.rand(1, 3, 14, 18) * 255
        base = F.interpolate(lr, scale_factor=2, mode='bilinear',
                             align_corners=False).round().clamp(0, 255)
        loss_fn = MixedLoss()
        old_loss = loss_fn(model(lr / 255, use_bilinear=False), (hr - base) / 255)
        old_loss.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        new_loss = loss_fn(model(lr / 255), hr / 255)
        new_loss.backward()
        torch.testing.assert_close(new_loss, old_loss)
        for parameter, expected in zip(model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, expected, rtol=1e-4, atol=1e-7)

    def test_patch_routing_respects_skip(self):
        from final_test import DPSRFrame
        from threshold_sweep import CachedPatches

        lr = torch.full((1, 3, 8, 8), 64.0)
        for enabled in (False, True):
            model = DPSR(2, 3, 8, num_blocks=1, subnet_channels=4,
                         use_bilinear=enabled).eval()
            frame = DPSRFrame(model, 2, 8, 0, -1, -2, 'cpu', 4)
            with torch.no_grad():
                actual, stats = frame.infer(lr)
                expected = torch.full_like(actual, 64.0 if enabled else 0.0)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(stats['branch_usage']['full'], 1)
                cached = CachedPatches(frame, lr, 1)
                for branch in (1, 2):
                    torch.testing.assert_close(cached.outputs[branch].float(), expected)

    def test_unified_training(self):
        for model in (DPSR(2, 3, 8, num_blocks=1, subnet_channels=4),
                      DPSR(2, 3, 8, num_blocks=1, subnet_channels=4, use_bilinear=False),
                      FSRCNN(2, num_channels=3)):
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            loss = train_epoch(model, [(torch.rand(1, 3, 7, 9) * 255,
                                       torch.rand(1, 3, 14, 18) * 255)],
                               MixedLoss(), optimizer, 'cpu', 0)
            self.assertTrue(torch.isfinite(torch.tensor(loss)))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
