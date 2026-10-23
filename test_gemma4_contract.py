"""Metadata rejection and algebra checks; no model-quality or GPU claims."""
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import unittest

from gemma4_contract import (
    LayerChunk, ProducerIdentity, derive_layers, rope_inverse_frequencies,
    rotate_vector, validate_bundle,
)


FIXTURE = Path(__file__).parent / "results/gemma4_compatibility/model_attention_config.json"


def hadamard(vector):
    values = list(vector)
    width = 1
    while width < len(values):
        for start in range(0, len(values), 2 * width):
            for j in range(start, start + width):
                a, b = values[j], values[j + width]
                values[j], values[j + width] = a + b, a - b
        width *= 2
    return tuple(v / math.sqrt(len(values)) for v in values)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class Gemma4Contract(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(FIXTURE.read_text())
        self.layers = derive_layers(self.config)
        self.identity = ProducerIdentity("1" * 64, "2" * 64, "3" * 64,
                                         "4" * 64, "synthetic-engine", 2, 0)
        # Synthetic allocator metadata, informed by real model dimensions.
        # No field here claims to describe live GPU allocations.
        self.chunks = [LayerChunk(
            replace(self.identity, rank=rank), layer, "5" * 64, "6" * 64,
            4096, 32, layer.layer, 16, (10, 11), (3, 3),
            16 * layer.kv_heads * layer.bytes_per_head_token,
            True, True, True, True, True,
        ) for layer in self.layers for rank in range(2)]

    def validate(self, chunks=None, **kwargs):
        options = dict(target_start=0, query_position=32,
                       target_prefix_sha256="7" * 64, mode="contextual_chunk")
        options.update(kwargs)
        return validate_bundle(self.chunks if chunks is None else chunks,
                               self.layers, self.identity, **options)

    def test_real_config_geometry_and_softcap_distinction(self):
        self.assertEqual(len(self.layers), 60)
        self.assertEqual(sum(x.sliding_window == 1024 for x in self.layers), 50)
        self.assertEqual([x.layer for x in self.layers if x.codec == "native_bf16"], [0, 1, 58, 59])
        local, global_layer = self.layers[2], self.layers[5]
        self.assertEqual((local.query_heads, local.kv_heads, local.head_dim), (16, 8, 256))
        self.assertEqual((global_layer.query_heads, global_layer.kv_heads, global_layer.head_dim), (16, 2, 512))
        self.assertEqual((local.bytes_per_head_token, global_layer.bytes_per_head_token), (262, 518))
        self.assertTrue(all(x.scale == 1.0 and x.attention_softcap is None for x in self.layers))
        self.assertEqual(self.config["text_config"]["final_logit_softcapping"], 30.0)

    def test_valid_metadata_never_claims_runtime_support(self):
        plan = self.validate()
        self.assertTrue(plan.metadata_valid)
        self.assertFalse(plan.runtime_execution_supported)
        self.assertEqual(plan.position_delta, -4096)
        self.assertTrue(any("BEFORE Hadamard" in h for h in plan.required_extensions))
        self.assertTrue(any("MTP" in h for h in plan.required_extensions))

    def test_contextual_state_cannot_be_claimed_as_an_exact_prefix(self):
        with self.assertRaisesRegex(ValueError, "contextual"):
            self.validate(mode="exact_prefix")
        plan = self.validate(mode="exact_prefix", target_start=4096,
                             query_position=4128, target_prefix_sha256="6" * 64)
        self.assertEqual(plan.position_delta, 0)
        self.assertFalse(plan.runtime_execution_supported)

    def test_same_tensor_shape_does_not_make_packing_or_rotary_abi_compatible(self):
        index = 4  # Compressed local layer 2, rank 0.
        for field, value in (("packing", "tq_aos"),
                             ("key_norm_semantics", "raw_norm_fp16"),
                             ("scale", 1 / 16), ("rope_theta", 1e6)):
            with self.subTest(field=field):
                changed = list(self.chunks)
                changed[index] = replace(changed[index], abi=replace(changed[index].abi, **{field: value}))
                with self.assertRaisesRegex(ValueError, "ABI"):
                    self.validate(changed)

    def test_model_configuration_weights_and_engine_generation_must_match(self):
        for field, value in (("weights_sha256", "8" * 64),
                             ("model_config_sha256", "8" * 64),
                             ("tokenizer_sha256", "8" * 64),
                             ("implementation_sha256", "8" * 64),
                             ("engine_instance", "restarted-engine")):
            with self.subTest(field=field):
                changed = list(self.chunks)
                changed[0] = replace(changed[0], identity=replace(changed[0].identity, **{field: value}))
                with self.assertRaisesRegex(ValueError, "Different"):
                    self.validate(changed)

    def test_missing_evicted_uncommitted_or_unpinned_shards_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "every layer"):
            self.validate(self.chunks[:-1])
        for flag in ("resident", "producer_write_complete", "accepted_tokens_only",
                     "immutable", "owner_lease_held"):
            for value in (False, "false"):
                with self.subTest(flag=flag, value=value):
                    changed = list(self.chunks)
                    changed[0] = replace(changed[0], **{flag: value})
                    with self.assertRaisesRegex(ValueError, "cannot be borrowed"):
                        self.validate(changed)

    def test_partial_blocks_bad_strides_and_multimodal_spans_fail_closed(self):
        with self.assertRaises(NotImplementedError):
            self.validate(target_start=1, query_position=33)
        with self.assertRaises(NotImplementedError):
            self.validate(modality="image")
        changed = list(self.chunks)
        changed[0] = replace(changed[0], block_stride_bytes=1)
        with self.assertRaisesRegex(ValueError, "stride"):
            self.validate(changed)

    def test_logical_positions_must_fit_independently_of_physical_length(self):
        with self.assertRaisesRegex(ValueError, "positions"):
            self.validate(query_position=200000)
        with self.assertRaisesRegex(ValueError, "positions"):
            self.validate(query_position=32.0)
        with self.assertRaisesRegex(ValueError, "causal"):
            self.validate(query_position=31)
        # Keeping a hole still needs logical-position masks, even with delta=0.
        plan = self.validate(target_start=4096, query_position=10000)
        self.assertEqual(plan.position_delta, 0)
        self.assertTrue(any("sliding masks" in h for h in plan.required_extensions))

    def test_proportional_rope_places_identity_pairs_in_both_halves(self):
        layer = self.layers[5]
        frequencies = rope_inverse_frequencies(layer)
        self.assertEqual(len(frequencies), 256)
        self.assertEqual(sum(f != 0 for f in frequencies), 64)
        self.assertAlmostEqual(frequencies[1], 1e6 ** (-2 / 512))
        rotated = rotate_vector([1.0] * 512, 13, layer)
        self.assertEqual(rotated[64:256], (1.0,) * 192)
        self.assertEqual(rotated[320:512], (1.0,) * 192)

    def test_query_shift_identity_requires_rope_before_hadamard(self):
        rng = random.Random(7319)
        for layer in (self.layers[2], self.layers[5]):
            # Mathematical oracle on a fixed synthetic quantized-basis vector.
            # This does not run TQ kernels or measure Gemma answer quality.
            codes = [rng.randrange(-8, 8) / 8 for _ in range(layer.head_dim)]
            reconstructed_key = hadamard(codes)
            query = [rng.gauss(0, 1) for _ in range(layer.head_dim)]
            delta = -137
            dense = dot(query, rotate_vector(reconstructed_key, delta, layer))
            correct = dot(hadamard(rotate_vector(query, -delta, layer)), codes)
            wrong_order = dot(rotate_vector(hadamard(query), -delta, layer), codes)
            self.assertAlmostEqual(dense, correct, places=10)
            self.assertGreater(abs(dense - wrong_order), 1e-3)


if __name__ == "__main__":
    unittest.main()
