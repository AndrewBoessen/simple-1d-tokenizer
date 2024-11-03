import warnings
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from .inception import get_inception_model


def get_covariance(sigma, total, num_examples):
    return torch.zeros_like(sigma) if num_examples == 0 else (sigma - torch.outer(total, total) / num_examples) / (num_examples - 1)


class VQGANEvaluator:
    def __init__(self, device, enable_rfid=True, enable_inception_score=True,
                 enable_codebook_usage_measure=False, enable_codebook_entropy_measure=False,
                 num_codebook_entries=1024):
        self._device = device
        self._enable = {
            'rfid': enable_rfid,
            'inception_score': enable_inception_score,
            'codebook_usage': enable_codebook_usage_measure,
            'codebook_entropy': enable_codebook_entropy_measure
        }
        self._num_codebook_entries = num_codebook_entries

        if self._enable['rfid'] or self._enable['inception_score']:
            self._inception_model = get_inception_model().to(device).eval()
            self._rfid_num_features, self._is_num_features = 2048, 1008
        self._is_eps, self._rfid_eps = 1e-16, 1e-6
        self.reset_metrics()

    def reset_metrics(self):
        self._num_examples = self._num_updates = 0
        if self._enable['inception_score']:
            self._is_prob_total = torch.zeros(
                self._is_num_features, dtype=torch.float64, device=self._device)
            self._is_total_kl_d = torch.zeros(
                self._is_num_features, dtype=torch.float64, device=self._device)
        if self._enable['rfid']:
            self._rfid_real_sigma = torch.zeros((self._rfid_num_features, self._rfid_num_features),
                                                dtype=torch.float64, device=self._device)
            self._rfid_fake_sigma = torch.zeros_like(self._rfid_real_sigma)
            self._rfid_real_total = torch.zeros(
                self._rfid_num_features, dtype=torch.float64, device=self._device)
            self._rfid_fake_total = torch.zeros_like(self._rfid_real_total)
        if self._enable['codebook_usage']:
            self._set_of_codebook_indices = set()
        if self._enable['codebook_entropy']:
            self._codebook_frequencies = torch.zeros(
                self._num_codebook_entries, dtype=torch.float64, device=self._device)

    def update(self, real_images, fake_images, codebook_indices=None):
        self._num_examples += real_images.shape[0]
        self._num_updates += 1

        if self._enable['inception_score'] or self._enable['rfid']:
            fake_inception_images = (fake_images * 255).to(torch.uint8)
            features_fake = self._inception_model(fake_inception_images)

            if self._enable['inception_score']:
                probs = F.softmax(features_fake["logits_unbiased"], dim=-1)
                self._is_prob_total += torch.sum(probs, 0, dtype=torch.float64)
                self._is_total_kl_d += torch.sum(probs * torch.log(
                    probs + self._is_eps), 0, dtype=torch.float64)

            if self._enable['rfid']:
                real_inception_images = (real_images * 255).to(torch.uint8)
                features_real = self._inception_model(real_inception_images)

                for f_real, f_fake in zip(features_real['2048'], features_fake['2048']):
                    self._rfid_real_total += f_real
                    self._rfid_fake_total += f_fake
                    self._rfid_real_sigma += torch.outer(f_real, f_real)
                    self._rfid_fake_sigma += torch.outer(f_fake, f_fake)

        if self._enable['codebook_usage'] and codebook_indices is not None:
            self._set_of_codebook_indices.update(
                torch.unique(codebook_indices, sorted=False).tolist())

        if self._enable['codebook_entropy'] and codebook_indices is not None:
            entries, counts = torch.unique(
                codebook_indices, sorted=False, return_counts=True)
            self._codebook_frequencies.index_add_(
                0, entries.int(), counts.double())

    def result(self):
        if self._num_examples < 1:
            raise ValueError("No examples to evaluate.")

        eval_score = {}

        if self._enable['inception_score']:
            mean_probs = self._is_prob_total / self._num_examples
            avg_kl_d = torch.sum(self._is_total_kl_d -
                                 self._is_prob_total * torch.log(mean_probs + self._is_eps)) / self._num_examples
            eval_score["InceptionScore"] = torch.exp(avg_kl_d).item()

        if self._enable['rfid']:
            mu_real = (self._rfid_real_total / self._num_examples).cpu()
            mu_fake = (self._rfid_fake_total / self._num_examples).cpu()
            sigma_real = get_covariance(
                self._rfid_real_sigma, self._rfid_real_total, self._num_examples).cpu()
            sigma_fake = get_covariance(
                self._rfid_fake_sigma, self._rfid_fake_total, self._num_examples).cpu()

            covmean, _ = linalg.sqrtm(
                sigma_real.mm(sigma_fake).numpy(), disp=False)
            if np.iscomplexobj(covmean) and not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError(f"Imaginary component {
                                 np.max(np.abs(covmean.imag))}")

            tr_covmean = np.trace(
                covmean.real if np.iscomplexobj(covmean) else covmean)
            if not np.isfinite(covmean).all():
                tr_covmean = np.sum(np.sqrt((np.diag(sigma_real) * self._rfid_eps) *
                                            (np.diag(sigma_fake) * self._rfid_eps)) /
                                    (self._rfid_eps * self._rfid_eps))

            diff = mu_real - mu_fake
            rfid = float(diff.dot(diff).item() + torch.trace(sigma_real) +
                         torch.trace(sigma_fake) - 2 * tr_covmean)

            if torch.isnan(torch.tensor(rfid)) or torch.isinf(torch.tensor(rfid)):
                warnings.warn(
                    "The product of covariance of train and test features is out of bounds.")
            eval_score["rFID"] = rfid

        if self._enable['codebook_usage']:
            eval_score["CodebookUsage"] = float(
                len(self._set_of_codebook_indices)) / self._num_codebook_entries

        if self._enable['codebook_entropy']:
            probs = self._codebook_frequencies / self._codebook_frequencies.sum()
            eval_score["CodebookEntropy"] = (
                -torch.log2(probs + 1e-8) * probs).sum()

        return eval_score
