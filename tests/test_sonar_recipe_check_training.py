"""Meaningful recipe deltas: raw parity, resume, LR endpoints and bbox geometry."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sonar_recipe_check import train as runner
from tools.sonar_recipe_check import augmentation as aug
import numpy as np
import torch


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.BatchNorm2d(2),
                                       torch.nn.Dropout(.2), torch.nn.Flatten(), torch.nn.Linear(32, 1))

    def forward(self, images):
        return self.net(images)


def loss(model, criterion, images, targets):
    value = (model(images)-.3).square().mean()
    return value, dict(loss=value.detach().item())


def batch():
    images = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)/32
    targets = [dict(boxes=torch.tensor([[1., 1., 3., 3.]]), labels=torch.tensor([i]),
                    path=f'train/images/{i}.png') for i in range(2)]
    return images, targets


def tensor_state_equal(test, first, second):
    test.assertEqual(first.keys(), second.keys())
    for key in first:
        test.assertTrue(torch.equal(first[key], second[key]), key)


class RecipeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_R1_preserves_raw_optimizer_path_data_and_rng(self):
        runner.p.seed_all(42)
        base = TinyModel()
        candidate = deepcopy(base)
        base_opt = torch.optim.AdamW(base.parameters(), lr=.001, weight_decay=.0001)
        opt = torch.optim.AdamW(candidate.parameters(), lr=.001, weight_decay=.0001)
        initial_rng = torch.get_rng_state().clone()
        ema = runner.ModelEMA(candidate)
        self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
        scheduler = runner.RecipeSchedule(opt, 'none')
        with patch('tools.sonar_head_factorial.model.batch_loss', side_effect=loss):
            runner.p.seed_all(42)
            baseline = runner.old.train_epoch(base, None, base_opt, [batch(), batch()], 'cpu')
            baseline_rng = torch.get_rng_state().clone()
            runner.p.seed_all(42)
            changed = runner.train_epoch(candidate, None, opt, [batch(), batch()], 'cpu', ema, scheduler)
        tensor_state_equal(self, base.state_dict(), candidate.state_dict())
        self.assertTrue(torch.equal(baseline_rng, torch.get_rng_state()))
        for key in ('batch_sha256', 'augmented_batches_sha256', 'image_order_sha256', 'losses', 'lr'):
            self.assertEqual(baseline[key], changed[key], key)
        for baseline_state, candidate_state in zip(base_opt.state.values(), opt.state.values()):
            tensor_state_equal(self, baseline_state, candidate_state)
        self.assertEqual(ema.updates, 2)
        self.assertNotEqual(runner.p.state_hash(candidate), runner.p.state_hash(ema.ema))

    def test_EMA_formula_and_integer_buffers_match_author(self):
        model = TinyModel()
        ema = runner.ModelEMA(model)
        original = deepcopy(model.state_dict())
        with torch.no_grad():
            for value in model.state_dict().values():
                value.add_(2)
        ema.update(model)
        decay = .9999*(1-np.exp(-1/2000))
        for key, actual in ema.ema.state_dict().items():
            expected = original[key].clone()
            if expected.dtype.is_floating_point:
                expected *= decay
                expected += (1-decay)*model.state_dict()[key]
            self.assertTrue(torch.equal(actual, expected), key)
        self.assertTrue(all(not item.requires_grad for item in ema.ema.parameters()))

    def test_checkpoint_resume_matches_uninterrupted_optimizer_EMA_schedule_and_rng(self):
        runner.p.seed_all(43)
        model = TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        ema = runner.ModelEMA(model)
        schedule = runner.RecipeSchedule(opt, 'linear3_cosine')
        with patch('tools.sonar_head_factorial.model.batch_loss', side_effect=loss):
            runner.train_epoch(model, None, opt, [batch(), batch()], 'cpu', ema, schedule)
            saved = deepcopy(dict(raw=model.state_dict(), optimizer=opt.state_dict(),
                                  ema=ema.state_dict(), schedule=schedule.state_dict(), rng=torch.get_rng_state()))
            runner.train_epoch(model, None, opt, [batch(), batch()], 'cpu', ema, schedule)
            final_rng = torch.get_rng_state().clone()
            resumed = TinyModel()
            resumed.load_state_dict(saved['raw'])
            resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=.001, weight_decay=.0001)
            resumed_opt.load_state_dict(saved['optimizer'])
            resumed_ema = runner.ModelEMA(resumed)
            resumed_ema.load_state_dict(saved['ema'])
            resumed_schedule = runner.RecipeSchedule(resumed_opt, 'linear3_cosine')
            resumed_schedule.load_state_dict(saved['schedule'])
            torch.set_rng_state(saved['rng'])
            runner.train_epoch(resumed, None, resumed_opt, [batch(), batch()], 'cpu', resumed_ema, resumed_schedule)
        tensor_state_equal(self, model.state_dict(), resumed.state_dict())
        tensor_state_equal(self, ema.ema.state_dict(), resumed_ema.ema.state_dict())
        self.assertEqual(ema.updates, resumed_ema.updates)
        self.assertEqual(schedule.state_dict(), resumed_schedule.state_dict())
        self.assertTrue(torch.equal(final_rng, torch.get_rng_state()))

    def test_schedule_boundaries_and_no_budget_extension(self):
        opt = torch.optim.AdamW(TinyModel().parameters(), lr=.001)
        schedule = runner.RecipeSchedule(opt, 'linear3_cosine')
        rates = [schedule.lr_at(step) for step in range(80*83)]
        self.assertEqual(rates[0], 0.)
        self.assertEqual(rates[248], .001)
        self.assertGreater(rates[248], rates[249])
        self.assertAlmostEqual(rates[-1], 1e-5, places=15)
        self.assertTrue(all(a < b for a, b in zip(rates[:248], rates[1:249])))
        self.assertTrue(all(a > b for a, b in zip(rates[248:-1], rates[249:])))
        with self.assertRaises(ValueError):
            schedule.lr_at(80*83)
        with self.assertRaises(ValueError):
            schedule.load_state_dict(dict(spec={**schedule.spec, 'warmup_epochs':4}, next_step=2))

    def test_mosaic_placement_preserves_class_box_and_pixel_correspondence(self):
        samples = [(torch.full((1, 8, 8), (i+1)/10),
                    dict(boxes=torch.tensor([[2., 2., 6., 6.]]), labels=torch.tensor([i%2])))
                   for i in range(4)]
        image, labels = aug.compose_mosaic(samples, 8, (8, 8))
        self.assertEqual(image.shape, (16, 16))
        np.testing.assert_array_equal(labels, np.array([[0,2,2,6,6],[1,10,2,14,6],
                                                       [0,2,10,6,14],[1,10,10,14,14]], np.float32))
        for i, (y, x) in enumerate([(4,4),(4,12),(12,4),(12,12)]):
            self.assertAlmostEqual(float(image[y,x]), (i+1)*25.5, places=4)

    def test_pinned_author_perspective_filters_and_clips_boxes(self):
        # Compile only the actual pinned geometry functions. No substitute
        # geometry implementation or optional author import stack is involved.
        import ast
        import cv2
        path = ROOT / 'reference/yolov9/utils/augmentations.py'
        parsed = ast.parse(path.read_text(encoding='utf-8'))
        selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                    and node.name in ('random_perspective', 'box_candidates')]
        namespace = dict(np=np, cv2=cv2, random=random, math=runner.math)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
        image = np.zeros((16,16), np.float32)
        boxes = np.array([[0,2,2,10,10],[1,0,0,1,1],[1,12,12,16,16]], np.float32)
        transformed, kept = namespace['random_perspective'](image, boxes, degrees=0., translate=0.,
                                    scale=0., shear=0., perspective=0., border=(-4,-4))
        self.assertEqual(transformed.shape, (8,8))
        np.testing.assert_array_equal(kept, np.array([[0,0,0,6,6]], np.float32))

    def test_wrapped_loader_preserves_sampler_generator_for_resume(self):
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
        class Base(Dataset):
            split, input_channels, image_size = 'train', 1, 8
            samples = [Path(f'{i}.png') for i in range(6)]
            def __len__(self): return 6
            def __getitem__(self, index):
                return torch.ones(1,8,8), dict(boxes=torch.tensor([[1.,1.,7.,7.]]),
                    labels=torch.tensor([index%2]), path=str(self.samples[index]))
        sampler = WeightedRandomSampler(torch.ones(6), 6, generator=torch.Generator().manual_seed(42))
        loader = DataLoader(Base(), batch_size=2, sampler=sampler, generator=torch.Generator().manual_seed(42))
        with patch.object(aug, 'author_random_perspective', return_value=lambda *args, **kwargs: None):
            wrapped = aug.wrap_loader(loader, ROOT/'reference/yolov9')
        self.assertIs(wrapped.sampler, loader.sampler)
        self.assertIs(wrapped.generator, loader.generator)
        state = runner.p.rng_state(wrapped)
        expected = list(iter(wrapped.sampler))
        runner.p.restore_rng(state, wrapped)
        self.assertEqual(expected, list(iter(wrapped.sampler)))


if __name__ == '__main__':
    unittest.main()
