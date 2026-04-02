"""
tests/test_all.py — Unit tests for every major component.

Run: pytest tests/test_all.py -v
"""

import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FullConfig, EMGFeatureConfig, AudioFeatureConfig, TransducerConfig


# ─── Config Tests ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_emg_feature_config(self):
        cfg = EMGFeatureConfig()
        assert cfg.total_features == 112
        assert cfg.frame_length_samples == 27
        assert cfg.frame_shift_samples == 10
        assert cfg.feature_rate == 100

    def test_audio_feature_config(self):
        cfg = AudioFeatureConfig()
        assert cfg.frame_length_samples == 432  # 27ms * 16000
        assert cfg.frame_shift_samples == 160   # 10ms * 16000

    def test_full_config(self):
        cfg = FullConfig()
        assert cfg.transducer.lstm_input_dim == 144  # 112 + 32
        assert cfg.wavenet.n_layers == 16


# ─── EMG Feature Tests ────────────────────────────────────────────────────────

class TestEMGFeatures:
    def test_preprocess_shape(self):
        from features.emg_features import preprocess_emg
        cfg = EMGFeatureConfig()
        raw = np.random.randn(5000, 8).astype(np.float32)
        filtered = preprocess_emg(raw, cfg)
        assert filtered.shape == (5000, 8)

    def test_extract_features_shape(self):
        from features.emg_features import extract_emg_features
        cfg = EMGFeatureConfig()
        emg = np.random.randn(5000, 8).astype(np.float32)
        features = extract_emg_features(emg, cfg)
        # n_frames = (5000 - 27) // 10 + 1 = 498
        expected_frames = (5000 - cfg.frame_length_samples) // cfg.frame_shift_samples + 1
        assert features.shape == (expected_frames, 112)

    def test_extract_features_112_dim(self):
        from features.emg_features import extract_emg_features
        emg = np.random.randn(3000, 8).astype(np.float32)
        features = extract_emg_features(emg)
        assert features.shape[1] == 112  # 14 features × 8 channels

    def test_normalizer(self):
        from features.emg_features import FeatureNormalizer
        data = np.random.randn(1000, 112) * 5 + 3
        norm = FeatureNormalizer()
        normalized = norm.fit_transform(data)
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.1

    def test_normalizer_save_load(self):
        from features.emg_features import FeatureNormalizer
        data = np.random.randn(500, 112)
        norm1 = FeatureNormalizer().fit(data)
        state = norm1.state_dict()

        norm2 = FeatureNormalizer()
        norm2.load_state_dict(state)

        test = np.random.randn(10, 112)
        np.testing.assert_array_equal(
            norm1.transform(test), norm2.transform(test)
        )


# ─── Audio Feature Tests ──────────────────────────────────────────────────────

class TestAudioFeatures:
    def test_mfcc_shape(self):
        from features.audio_features import extract_mfcc
        audio = np.random.randn(16000).astype(np.float32)  # 1 second
        mfcc = extract_mfcc(audio)
        assert mfcc.shape[1] == 26
        # ~100 frames for 1 second at 10ms stride
        assert 95 <= mfcc.shape[0] <= 105

    def test_mu_law_roundtrip(self):
        from features.audio_features import mu_law_encode, mu_law_decode
        audio = np.random.randn(1000).astype(np.float32) * 0.5
        audio = np.clip(audio, -1, 1)
        encoded = mu_law_encode(audio)
        decoded = mu_law_decode(encoded)
        # Should be close but not exact (quantization)
        assert np.max(np.abs(decoded - audio)) < 0.1

    def test_mu_law_range(self):
        from features.audio_features import mu_law_encode
        audio = np.array([-1.0, 0.0, 1.0])
        encoded = mu_law_encode(audio, mu=255)
        assert np.all(encoded >= 0)
        assert np.all(encoded <= 255)


# ─── DTW Tests ────────────────────────────────────────────────────────────────

class TestDTW:
    def test_identity_alignment(self):
        from alignment.dtw import dtw_alignment
        seq = np.random.randn(50, 10)
        align = dtw_alignment(seq, seq)
        # Aligning a sequence to itself → identity mapping
        np.testing.assert_array_equal(align, np.arange(50))

    def test_alignment_shape(self):
        from alignment.dtw import dtw_alignment
        s1 = np.random.randn(30, 10)
        s2 = np.random.randn(50, 10)
        align = dtw_alignment(s1, s2)
        assert len(align) == 30
        assert np.all(align >= 0)
        assert np.all(align < 50)

    def test_alignment_monotonic(self):
        from alignment.dtw import dtw_alignment
        s1 = np.random.randn(40, 5)
        s2 = np.random.randn(60, 5)
        align = dtw_alignment(s1, s2)
        # Alignment should be monotonically non-decreasing
        for i in range(1, len(align)):
            assert align[i] >= align[i - 1]

    def test_warp_features(self):
        from alignment.dtw import warp_features
        features = np.random.randn(100, 26)
        alignment = np.array([0, 0, 1, 2, 3, 3, 4])
        warped = warp_features(features, alignment)
        assert warped.shape == (7, 26)
        np.testing.assert_array_equal(warped[0], features[0])
        np.testing.assert_array_equal(warped[2], features[1])


# ─── CCA Tests ────────────────────────────────────────────────────────────────

class TestCCA:
    def test_cca_fit_project(self):
        from alignment.cca import CCAAligner
        n, d = 500, 112
        X = np.random.randn(n, d)
        Y = np.random.randn(n, d)
        cca = CCAAligner(n_components=15)
        cca.fit(X, Y)
        proj = cca.project_silent(X[:10])
        assert proj.shape == (10, 15)

    def test_cca_cost_matrix(self):
        from alignment.cca import CCAAligner
        n, d = 200, 112
        X = np.random.randn(n, d)
        Y = np.random.randn(n, d)
        cca = CCAAligner(n_components=15)
        cca.fit(X, Y)

        s1 = np.random.randn(30, d)
        s2 = np.random.randn(40, d)
        cost = cca.cca_cost_matrix(s1, s2)
        assert cost.shape == (30, 40)
        assert np.all(cost >= 0)


# ─── Transducer Model Tests ──────────────────────────────────────────────────

class TestTransducer:
    def test_forward_shape(self):
        from models.transducer import EMGTransducer
        cfg = TransducerConfig()
        model = EMGTransducer(cfg, n_sessions=5)

        B, T = 4, 50
        emg = torch.randn(B, T, 112)
        sessions = torch.randint(0, 5, (B,))
        out = model(emg, sessions)
        assert out.shape == (B, T, 26)

    def test_with_lengths(self):
        from models.transducer import EMGTransducer
        cfg = TransducerConfig()
        model = EMGTransducer(cfg, n_sessions=2)

        B, T = 3, 40
        emg = torch.randn(B, T, 112)
        sessions = torch.zeros(B, dtype=torch.long)
        lengths = torch.tensor([40, 30, 20])
        out = model(emg, sessions, lengths)
        assert out.shape == (B, T, 26)

    def test_predict_no_grad(self):
        from models.transducer import EMGTransducer
        cfg = TransducerConfig()
        model = EMGTransducer(cfg, n_sessions=1)

        emg = torch.randn(1, 20, 112)
        sessions = torch.zeros(1, dtype=torch.long)
        out = model.predict(emg, sessions)
        assert out.shape == (1, 20, 26)
        assert not out.requires_grad

    def test_parameter_count(self):
        from models.transducer import EMGTransducer
        cfg = TransducerConfig()
        model = EMGTransducer(cfg, n_sessions=10)
        n_params = sum(p.numel() for p in model.parameters())
        # Should be in the tens of millions range
        assert n_params > 1_000_000


# ─── WaveNet Tests ────────────────────────────────────────────────────────────

class TestWaveNet:
    def test_conditioning_shape(self):
        from models.wavenet import ConditioningNetwork, WaveNetConfig
        cfg = WaveNetConfig()
        cond = ConditioningNetwork(input_dim=26, cfg=cfg)

        B, T = 2, 50  # 50 frames × 160 → 8000 samples
        mfcc = torch.randn(B, T, 26)
        out = cond(mfcc)
        assert out.shape[0] == B
        assert out.shape[1] == 128  # n_cond_channels

    def test_wavenet_forward(self):
        from models.wavenet import WaveNet, WaveNetConfig
        cfg = WaveNetConfig()
        # Use smaller dims for testing
        cfg.n_residual_channels = 8
        cfg.n_skip_channels = 16
        cfg.n_layers = 4
        cfg.max_dilation = 8

        model = WaveNet(cfg)
        B = 1
        T_audio = 1000
        T_feat = T_audio // 160 + 1

        audio = torch.randn(B, T_audio)
        mfcc = torch.randn(B, T_feat, 26)
        logits = model(audio, mfcc)
        assert logits.shape[0] == B
        assert logits.shape[1] == cfg.n_out_channels


# ─── Dataset Tests ────────────────────────────────────────────────────────────

class TestDataset:
    def test_emg_speech_dataset(self):
        from data.dataset import EMGSpeechDataset
        n = 10
        emg = [np.random.randn(np.random.randint(50, 100), 112) for _ in range(n)]
        audio = [np.random.randn(len(e), 26) for e in emg]
        sessions = [0] * n

        ds = EMGSpeechDataset(emg, audio, sessions)
        assert len(ds) == n

        item = ds[0]
        assert 'emg' in item
        assert 'audio' in item
        assert item['emg'].shape[1] == 112
        assert item['audio'].shape[1] == 26

    def test_collate_fn(self):
        from data.dataset import EMGSpeechDataset, collate_fn
        emg = [np.random.randn(50, 112), np.random.randn(30, 112)]
        audio = [np.random.randn(50, 26), np.random.randn(30, 26)]
        ds = EMGSpeechDataset(emg, audio, [0, 0])

        batch = collate_fn([ds[0], ds[1]])
        assert batch['emg'].shape == (2, 50, 112)   # padded to max
        assert batch['audio'].shape == (2, 50, 26)
        assert batch['lengths'].tolist() == [50, 30]


# ─── WER Tests ────────────────────────────────────────────────────────────────

class TestWER:
    def test_perfect_match(self):
        from utils.evaluation import word_error_rate
        wer, _ = word_error_rate("hello world", "hello world")
        assert wer == 0.0

    def test_complete_mismatch(self):
        from utils.evaluation import word_error_rate
        wer, details = word_error_rate("hello world", "foo bar baz")
        assert wer > 0

    def test_empty_hypothesis(self):
        from utils.evaluation import word_error_rate
        wer, details = word_error_rate("hello world", "")
        assert wer == 1.0  # all deletions
        assert details['deletions'] == 2

    def test_empty_reference(self):
        from utils.evaluation import word_error_rate
        wer, _ = word_error_rate("", "hello")
        assert wer == 1.0

    def test_batch_wer(self):
        from utils.evaluation import batch_wer
        refs = ["hello world", "the cat sat"]
        hyps = ["hello world", "the dog sat"]
        results = batch_wer(refs, hyps)
        assert results['n_utterances'] == 2
        assert results['aggregate_wer'] < 0.5


# ─── Integration Test ─────────────────────────────────────────────────────────

class TestIntegration:
    def test_emg_to_mfcc_pipeline(self):
        """Test the full feature extraction → model → prediction pipeline."""
        from features.emg_features import preprocess_emg, extract_emg_features
        from features.audio_features import extract_mfcc
        from models.transducer import EMGTransducer

        cfg = FullConfig()

        # Simulate raw EMG (1 second at 1000 Hz)
        raw_emg = np.random.randn(1000, 8).astype(np.float32)
        filtered = preprocess_emg(raw_emg, cfg.emg)
        emg_feats = extract_emg_features(filtered, cfg.emg)
        assert emg_feats.shape[1] == 112

        # Run through model
        model = EMGTransducer(cfg.transducer, n_sessions=1)
        emg_t = torch.tensor(emg_feats, dtype=torch.float32).unsqueeze(0)
        sess_t = torch.zeros(1, dtype=torch.long)

        with torch.no_grad():
            predicted_mfcc = model(emg_t, sess_t)

        assert predicted_mfcc.shape[0] == 1
        assert predicted_mfcc.shape[2] == 26  # MFCC dimension

    def test_alignment_pipeline(self):
        """Test DTW → CCA → target generation."""
        from alignment.target_transfer import TargetTransfer

        transfer = TargetTransfer(cca_n_components=5)

        # Simulate features
        n_silent, n_vocal, d = 80, 100, 112
        silent = [np.random.randn(n_silent, d)]
        vocal = [np.random.randn(n_vocal, d)]
        audio = [np.random.randn(n_vocal, 26)]

        # Initial alignment
        alignments = transfer.initial_alignment(silent, vocal)
        assert len(alignments[0]) == n_silent

        # CCA refinement
        alignments = transfer.fit_cca_and_realign(
            silent, vocal, alignments
        )
        assert len(alignments[0]) == n_silent

        # Generate targets
        targets = transfer.generate_targets(audio, alignments)
        assert targets[0].shape == (n_silent, 26)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
