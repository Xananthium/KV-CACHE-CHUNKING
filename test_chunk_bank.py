import unittest

import torch

from chunk_bank import (ChunkBank, ChunkInUseError, ChunkMetadata, ChunkUse,
                        ModelFingerprint, rotate)
from adapter_contract import CacheABI, Placement, prepare_reference_view


def meta(layer, positions):
    positions = tuple(positions)
    return ChunkMetadata(layer, tuple(range(len(positions))), positions, "test-prefix")


class ChunkBankTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.bank = ChunkBank(ModelFingerprint("test", "revision", "tokenizer"))

    def test_owning_ingestion_four_forks_and_safe_eviction(self):
        source_k, source_v = torch.randn(1, 2, 200, 8), torch.randn(1, 2, 200, 8)
        k, v = source_k[:, :, 5:7], source_v[:, :, 5:7]
        handle = self.bank.put(k, v, meta(0, [5, 6]))
        self.assertEqual(handle, self.bank.put(k, v, meta(0, [5, 6])))
        expected = 2 * 1 * 2 * 2 * 8 * 4
        self.assertEqual(self.bank.stats()["unique_physical_bytes"], expected)
        identity = self.bank.storage_identity(handle)
        root = self.bank.view([ChunkUse(handle)])
        children = [root.fork() for _ in range(4)]
        stats = self.bank.stats()
        self.assertEqual(stats["handle_references"], 5)
        self.assertEqual(stats["unique_physical_bytes"], expected)
        self.assertEqual(stats["logical_reference_bytes"], expected * 5)
        self.assertEqual(stats["avoided_duplicate_bytes"], expected * 4)
        self.assertEqual(self.bank.storage_identity(handle), identity)
        q = torch.randn(1, 4, 1, 8)
        before = root.attend(q, layer=0, query_positions=[6])
        source_k.zero_()
        source_v.zero_()
        torch.testing.assert_close(before, root.attend(q, layer=0, query_positions=[6]))
        with self.assertRaises(ChunkInUseError):
            self.bank.evict(handle)
        root.close()
        children[0].remove(handle)
        self.assertEqual(self.bank.stats()["handle_references"], 3)
        with self.assertRaises(ChunkInUseError):
            self.bank.evict(handle)
        for child in children:
            child.close()
        self.bank.evict(handle)
        self.assertEqual(self.bank.stats()["unique_physical_bytes"], 0)

    def test_segmented_gqa_causal_holes_and_rope_shift_match_dense(self):
        inv = torch.tensor([1.0, 0.1, 0.01, 0.001])
        query = torch.randn(2, 6, 4, 8)
        qpos = torch.tensor([-1, 2, 7, 9])
        uses, dense_k, dense_v, dense_positions = [], [], [], []
        for positions, shift in [([0, 1, 2], 0), ([5, 6], 2)]:
            k, v = torch.randn(2, 2, len(positions), 8), torch.randn(2, 2, len(positions), 8)
            handle = self.bank.put(k, v, meta(0, positions))
            uses.append(ChunkUse(handle, shift))
            dense_k.append(rotate(k, shift, inv))
            dense_v.append(v)
            dense_positions.extend(p + shift for p in positions)
        with self.bank.view(uses) as view:
            output = view.attend(query, layer=0, query_positions=qpos, inv_freq=inv)
            # Dense concatenation exists only in the independent test oracle.
            key = torch.cat(dense_k, 2).repeat_interleave(3, dim=1)
            value = torch.cat(dense_v, 2).repeat_interleave(3, dim=1)
            scores = query @ key.transpose(-1, -2) / (8 ** 0.5)
            mask = torch.tensor(dense_positions)[None, :] <= qpos[:, None]
            weights = scores.masked_fill(~mask, -torch.inf).softmax(-1).nan_to_num()
            expected = weights @ value
            torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
            self.assertTrue(torch.equal(output[:, :, 0], torch.zeros_like(output[:, :, 0])))
            with self.assertRaises(ValueError):
                view.attend(query, layer=0, query_positions=qpos)

    def test_independent_tails_seal_and_fork_after_tail(self):
        k, v = torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8)
        handle = self.bank.put(k, v, meta(0, [0, 1]))
        root = self.bank.view([ChunkUse(handle)])
        sibling = root.fork()
        root.append_tail(k[:, :, :1], v[:, :, :1], meta(0, [2]))
        self.assertGreater(root.tail_bytes, 0)
        self.assertEqual(sibling.tail_bytes, 0)
        child = root.fork()
        self.assertEqual(child.tail_bytes, root.tail_bytes)
        self.assertNotEqual(child._tails[0][0].key.data_ptr(), root._tails[0][0].key.data_ptr())
        root.append_tail(k[:, :, :1], v[:, :, :1], meta(0, [3]))
        self.assertEqual(root.tail_bytes, 2 * child.tail_bytes)
        child.seal_tail()
        self.assertEqual(child.tail_bytes, 0)
        self.assertEqual(len(child.uses), 2)
        self.assertEqual(len(sibling.uses), 1)
        sealed_fork = child.fork()
        self.assertEqual(sealed_fork.uses, child.uses)
        for view in [root, sibling, child, sealed_fork]:
            view.close()
        self.assertEqual(self.bank.stats()["handle_references"], 0)
        with self.assertRaises(RuntimeError):
            root.fork()

    def test_fingerprint_and_invalid_position_rejected(self):
        key = torch.zeros(1, 2, 1, 8)
        with self.assertRaises(ValueError):
            self.bank.put(key.half(), key.half(), meta(0, [0]))
        handle = self.bank.put(key, key, meta(0, [0]))
        with self.assertRaises(ValueError):
            self.bank.view([ChunkUse(handle, -1)])
        self.assertEqual(self.bank.stats()["handle_references"], 0)

    def test_adapter_rejects_contextual_change_as_exact_hit_and_unsupported_abi(self):
        key = torch.zeros(1, 2, 2, 8)
        handle = self.bank.put(key, key, meta(0, [5, 6]))
        abi = CacheABI(self.bank.fingerprint)
        changed = [Placement(handle, 2, "different-prefix")]
        with self.assertRaises(ValueError):
            prepare_reference_view(self.bank, abi, changed, mode="exact_prefix")
        with prepare_reference_view(self.bank, abi, changed, mode="contextual_chunk") as view:
            self.assertEqual(view.uses[0].position_delta, -3)
        with self.assertRaises(NotImplementedError):
            prepare_reference_view(self.bank, CacheABI(self.bank.fingerprint, quantization="custom_turboquant"),
                                   changed, mode="contextual_chunk")


if __name__ == "__main__":
    unittest.main()
