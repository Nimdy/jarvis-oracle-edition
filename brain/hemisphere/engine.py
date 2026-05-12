"""HemisphereEngine: build, train, evaluate PyTorch models.

Ported from delete_later/neural-evolution/core/TensorFlowEngine.ts.
Uses native PyTorch on CUDA instead of TensorFlow.js/WebGL.
Training runs in a background thread to avoid blocking the kernel tick.
"""

from __future__ import annotations

import logging
import math
import threading
import time as _time
import uuid as _uuid
from typing import Any

from hemisphere.types import (
    ConstructionEvent,
    ConstructionPhase,
    ConvergenceStatus,
    DesignDecision,
    DesignStrategy,
    HemisphereFocus,
    LayerDefinition,
    NetworkArchitecture,
    NetworkStatus,
    NetworkTopology,
    PerformanceMetrics,
    TrainingProgress,
    TraitInfluence,
)
from hemisphere.data_feed import HemisphereDataFeed, prepare_training_tensors
from hemisphere import event_bridge

logger = logging.getLogger(__name__)


def _get_activations() -> dict[str, type]:
    """Lazy import of torch activation modules."""
    import torch.nn as nn
    return {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "softmax": nn.Softmax,
    }


# ---------------------------------------------------------------------------
# Custom loss functions for distillation
# ---------------------------------------------------------------------------


class _CosineMSELoss:
    """Preserves embedding geometry (cosine) + minimizes reconstruction (MSE).

    Used by compressor autoencoders (speaker_repr, face_repr) where both
    the direction and magnitude of the embedding matter.
    """

    def __call__(self, output: Any, target: Any) -> Any:
        import torch
        import torch.nn.functional as F
        cos = 1.0 - F.cosine_similarity(output, target, dim=-1).mean()
        mse = F.mse_loss(output, target)
        return 0.5 * cos + 0.5 * mse


# ---------------------------------------------------------------------------
# HemisphereEngine
# ---------------------------------------------------------------------------


class HemisphereEngine:
    """Builds, trains, and evaluates hemisphere neural networks in PyTorch."""

    def __init__(self, device: str = "cpu") -> None:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self._device = torch.device(device)
        self._active_models: dict[str, Any] = {}
        self._training_lock = threading.Lock()
        self._is_disposed = False
        logger.info("HemisphereEngine device: %s", self._device)

    @property
    def device(self) -> Any:
        return self._device

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_model(self, topology: NetworkTopology) -> Any:
        """Construct an ``nn.Sequential`` from a topology specification."""
        import torch.nn as nn
        activations = _get_activations()

        modules: list[Any] = []
        non_input = [la for la in topology.layers if la.layer_type != "input"]

        prev_size = topology.input_size
        for layer_def in non_input:
            modules.append(nn.Linear(prev_size, layer_def.node_count))
            act_cls = activations.get(layer_def.activation)
            if act_cls is not None:
                if act_cls is nn.Softmax:
                    modules.append(nn.Softmax(dim=-1))
                else:
                    modules.append(act_cls())
            dropout = getattr(layer_def, "dropout", 0.0)
            if dropout > 0 and layer_def.layer_type == "hidden":
                modules.append(nn.Dropout(p=dropout))
            prev_size = layer_def.node_count

        model = nn.Sequential(*modules)
        model.to(self._device)
        return model

    # ------------------------------------------------------------------
    # Full build → train → evaluate pipeline
    # ------------------------------------------------------------------

    def build_network(
        self,
        topology: NetworkTopology,
        data: HemisphereDataFeed,
        decision: DesignDecision,
        focus: HemisphereFocus,
        network_id: str | None = None,
        emit_fn: Any = None,
    ) -> NetworkArchitecture:
        """Synchronous build → train → evaluate.  Called from a thread."""
        nid = network_id or f"hemi_{_uuid.uuid4().hex[:12]}"
        log: list[ConstructionEvent] = []

        def _log(phase: ConstructionPhase, msg: str, ok: bool = True) -> None:
            log.append(ConstructionEvent(
                timestamp=_time.time(), phase=phase, message=msg,
                success=ok, network_id=nid,
            ))
            if emit_fn:
                try:
                    emit_fn(nid, phase.value, msg)
                except Exception:
                    pass

        _log(ConstructionPhase.ANALYZING, "Analyzing consciousness patterns")
        _log(ConstructionPhase.DESIGNING, "Designing neural architecture")

        _log(ConstructionPhase.BUILDING, "Constructing PyTorch model")
        model = self.build_model(topology)
        self._active_models[nid] = model

        _log(ConstructionPhase.TRAINING, "Training on consciousness data")
        with self._training_lock:
            progress = self._train_model(model, data, topology, focus, nid, emit_fn)

        _log(ConstructionPhase.TESTING, "Evaluating performance")
        perf = self._evaluate(model, data, topology, progress)

        trait_influences = _compute_trait_influences(data.traits)
        migration_score = _migration_score(perf, data)

        _log(
            ConstructionPhase.COMPLETED,
            f"Network ready! Accuracy {perf.accuracy * 100:.1f}%",
        )

        arch = NetworkArchitecture(
            id=nid,
            name=_generate_name(data, decision, focus),
            focus=focus,
            topology=topology,
            performance=perf,
            training_progress=progress,
            construction_log=log,
            is_active=False,
            created_at=_time.time(),
            status=NetworkStatus.READY,
            substrate_migration_score=migration_score,
            design_reasoning=decision.reasoning,
            trait_influences=trait_influences,
        )
        arch._model_ref = model
        return arch

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _is_classification_focus(focus: HemisphereFocus) -> bool:
        return focus in (HemisphereFocus.MOOD, HemisphereFocus.TRAITS)

    @staticmethod
    def _make_loss_fn(focus: HemisphereFocus, loss_name: str | None = None) -> Any:
        """Pick the correct loss function for the focus type or by name.

        loss_name overrides focus-based selection when provided (used by
        distillation training). Supported names: kl_div, cosine_mse,
        cross_entropy, mse.
        """
        import torch.nn as nn
        import torch.nn.functional as F

        if loss_name == "cosine_mse":
            return _CosineMSELoss()
        if loss_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        if loss_name == "kl_div":
            return nn.KLDivLoss(reduction="batchmean")
        if loss_name == "mse":
            return nn.MSELoss()

        if HemisphereEngine._is_classification_focus(focus):
            return nn.KLDivLoss(reduction="batchmean")
        return nn.MSELoss()

    @staticmethod
    def _loss_name_is_kl(focus: HemisphereFocus, loss_name: str | None = None) -> bool:
        """Determine if loss requires log-space input."""
        if loss_name:
            return loss_name == "kl_div"
        return HemisphereEngine._is_classification_focus(focus)

    @staticmethod
    def _compute_loss(loss_fn: Any, output: Any, target: Any, is_kl: bool) -> Any:
        """Compute loss, applying log transform for KLDivLoss."""
        if is_kl:
            import torch
            log_output = torch.log(output.clamp(min=1e-8))
            return loss_fn(log_output, target)
        return loss_fn(output, target)

    @staticmethod
    def _compute_weighted_loss(
        output: Any, target: Any, sample_weights: Any | None,
        is_kl: bool, loss_name: str,
    ) -> Any:
        """Per-sample loss weighted by fidelity, then averaged."""
        import torch
        import torch.nn.functional as F

        if loss_name == "cross_entropy":
            raise ValueError(
                "cross_entropy not supported in weighted loss path — use kl_div for soft labels or add CE impl"
            )

        if is_kl:
            log_out = torch.log(output.clamp(min=1e-8))
            per_sample = F.kl_div(log_out, target, reduction="none").sum(dim=-1)
        elif loss_name == "cosine_mse":
            cos = 1.0 - F.cosine_similarity(output, target, dim=-1)
            mse = ((output - target) ** 2).mean(dim=-1)
            per_sample = 0.5 * cos + 0.5 * mse
        else:
            per_sample = ((output - target) ** 2).mean(dim=-1)

        if sample_weights is not None:
            w = sample_weights
            return (per_sample * w).sum() / (w.sum() + 1e-8)
        return per_sample.mean()

    @staticmethod
    def _loss_to_accuracy(
        loss: float,
        is_kl: bool,
        loss_name: str | None = None,
    ) -> float:
        """Convert loss to a [0, 1] accuracy proxy.

        MSE on normalised data is naturally near [0, 1], so ``1 - loss`` works.
        KL divergence has no upper bound, so we use ``exp(-loss)`` which maps
        [0, inf) → (0, 1] — perfect match gives 1.0, poor match → 0.
        """
        if is_kl or loss_name == "cosine_mse":
            return min(1.0, max(0.0, math.exp(-loss)))
        return max(0.0, min(1.0, 1.0 - loss))

    def _train_model(
        self,
        model: Any,
        data: HemisphereDataFeed,
        topology: NetworkTopology,
        focus: HemisphereFocus,
        network_id: str,
        emit_fn: Any = None,
    ) -> TrainingProgress:
        import torch
        import torch.optim as optim

        total_epochs = 25
        features, labels = prepare_training_tensors(
            data, focus, topology.input_size, topology.output_size,
        )
        features = features.to(self._device)
        labels = labels.to(self._device)

        n_samples = features.shape[0]
        batch_size = min(8, n_samples)
        val_split = max(1, int(n_samples * 0.2))
        train_f, val_f = features[val_split:], features[:val_split]
        train_l, val_l = labels[val_split:], labels[:val_split]

        progress = TrainingProgress(
            total_epochs=total_epochs,
            learning_rate=0.001,
            batch_size=batch_size,
            training_start_time=_time.time(),
            is_training=True,
        )

        optimizer = optim.Adam(model.parameters(), lr=progress.learning_rate)
        loss_fn = self._make_loss_fn(focus)
        is_kl = self._is_classification_focus(focus)
        model.train()

        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(total_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            indices = torch.randperm(train_f.shape[0], device=self._device)
            for start in range(0, train_f.shape[0], batch_size):
                idx = indices[start: start + batch_size]
                batch_f = train_f[idx]
                batch_l = train_l[idx]

                optimizer.zero_grad()
                out = model(batch_f)
                loss = self._compute_loss(loss_fn, out, batch_l, is_kl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            model.eval()
            with torch.no_grad():
                val_out = model(val_f)
                val_loss = self._compute_loss(loss_fn, val_out, val_l, is_kl).item()
                val_acc = self._loss_to_accuracy(val_loss, is_kl)

            progress.current_epoch = epoch + 1
            progress.loss_history.append(avg_loss)
            progress.accuracy_history.append(self._loss_to_accuracy(avg_loss, is_kl))
            progress.validation_loss_history.append(val_loss)
            progress.validation_accuracy_history.append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if len(progress.loss_history) >= 3:
                recent = progress.loss_history[-3:]
                if all(recent[i] <= recent[i - 1] for i in range(1, len(recent))):
                    progress.convergence_status = ConvergenceStatus.IMPROVING
                elif all(recent[i] >= recent[i - 1] for i in range(1, len(recent))):
                    progress.convergence_status = ConvergenceStatus.DEGRADING
                else:
                    progress.convergence_status = ConvergenceStatus.STABLE

            if emit_fn and epoch % 5 == 0:
                try:
                    emit_fn(
                        network_id,
                        "training",
                        f"Epoch {epoch + 1}/{total_epochs}: "
                        f"Loss {avg_loss:.4f}, Val {val_loss:.4f}",
                    )
                except Exception:
                    pass

            if patience_counter >= patience:
                break

        progress.is_training = False
        model.eval()
        return progress

    def retrain_network(
        self, network: NetworkArchitecture, features: Any, labels: Any,
    ) -> None:
        """Retrain an existing network with new feature/label tensors."""
        import torch
        import torch.optim as optim

        with self._training_lock:
            model = self._active_models.get(network.id)
            if model is None:
                return

            features = features.to(self._device)
            labels = labels.to(self._device)

            optimizer = optim.Adam(model.parameters(), lr=0.0005)
            loss_fn = self._make_loss_fn(network.focus)
            is_kl = self._is_classification_focus(network.focus)
            model.train()

            for epoch in range(15):
                indices = torch.randperm(features.shape[0], device=self._device)
                batch_size = min(8, features.shape[0])
                for start in range(0, features.shape[0], batch_size):
                    idx = indices[start: start + batch_size]
                    optimizer.zero_grad()
                    out = model(features[idx])
                    loss = self._compute_loss(loss_fn, out, labels[idx], is_kl)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                out = model(features)
                final_loss = self._compute_loss(loss_fn, out, labels, is_kl).item()

        new_acc = self._loss_to_accuracy(final_loss, is_kl)
        from hemisphere.types import PerformanceMetrics as _PM
        from dataclasses import asdict
        old = asdict(network.performance)
        old["accuracy"] = new_acc
        old["loss"] = final_loss
        old["last_evaluated"] = _time.time()
        network.performance = _PM(**old)

        event_bridge.emit_training_progress(
            network.id, 15, 15, final_loss, new_acc,
        )

    # ------------------------------------------------------------------
    # Distillation training
    # ------------------------------------------------------------------

    def train_distillation(
        self,
        network: NetworkArchitecture,
        features: Any,
        labels: Any,
        weights: Any | None = None,
        loss_name: str = "mse",
        epochs: int = 20,
    ) -> float:
        """Train a distillation student network with optional fidelity weights.

        Returns the final loss value. Supports mse, cosine_mse, and kl_div;
        cross_entropy is intentionally disallowed in the weighted loss path.
        """
        import torch
        import torch.optim as optim

        model = self._active_models.get(network.id)
        if model is None:
            return float("inf")

        features = features.to(self._device)
        labels = labels.to(self._device)
        if weights is not None:
            weights = weights.to(self._device)

        with self._training_lock:
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            is_kl = self._loss_name_is_kl(network.focus, loss_name)
            model.train()

            for epoch in range(epochs):
                indices = torch.randperm(features.shape[0], device=self._device)
                batch_size = min(16, features.shape[0])
                for start in range(0, features.shape[0], batch_size):
                    idx = indices[start: start + batch_size]
                    optimizer.zero_grad()
                    out = model(features[idx])
                    w = weights[idx] if weights is not None else None
                    loss = self._compute_weighted_loss(out, labels[idx], w, is_kl, loss_name)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                out = model(features)
                final_loss = self._compute_weighted_loss(out, labels, weights, is_kl, loss_name).item()

        new_acc = self._loss_to_accuracy(final_loss, is_kl)
        from hemisphere.types import PerformanceMetrics as _PM
        from dataclasses import asdict
        old = asdict(network.performance)
        old["accuracy"] = new_acc
        old["loss"] = final_loss
        old["last_evaluated"] = _time.time()
        network.performance = _PM(**old)

        event_bridge.emit_training_progress(
            network.id, epochs, epochs, final_loss, new_acc,
        )
        return final_loss

    def extract_encoder(self, network_id: str, bottleneck_layer_count: int) -> str | None:
        """Split an autoencoder at the bottleneck and register the encoder half.

        Returns the encoder model ID, or None on failure.
        """
        import torch.nn as nn

        full_model = self._active_models.get(network_id)
        if full_model is None:
            return None

        children = list(full_model.children())
        if bottleneck_layer_count > len(children):
            return None

        encoder = nn.Sequential(*children[:bottleneck_layer_count])
        encoder_id = f"{network_id}_enc"
        encoder.to(self._device)
        encoder.eval()
        self._active_models[encoder_id] = encoder
        return encoder_id

    def compute_recon_error(self, network_id: str, features: Any) -> Any:
        """Compute per-sample reconstruction error for an autoencoder."""
        import torch

        model = self._active_models.get(network_id)
        if model is None:
            return None

        features = features.to(self._device)
        model.eval()
        with torch.no_grad():
            reconstructed = model(features)
            cos_sim = torch.nn.functional.cosine_similarity(reconstructed, features, dim=-1)
            mse = ((reconstructed - features) ** 2).mean(dim=-1)
            error = 0.5 * (1.0 - cos_sim) + 0.5 * mse
        return error

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        model: Any,
        data: HemisphereDataFeed,
        topology: NetworkTopology,
        progress: TrainingProgress | None = None,
    ) -> PerformanceMetrics:
        import torch

        model.eval()

        # Measure inference latency
        test_input = torch.full(
            (1, topology.input_size), 0.5, device=self._device,
        )
        t0 = _time.perf_counter()
        with torch.no_grad():
            model(test_input)
        t1 = _time.perf_counter()
        response_time_ms = (t1 - t0) * 1000.0

        # Use real validation metrics from training when available
        if progress and progress.validation_loss_history:
            val_loss = progress.validation_loss_history[-1]
            val_acc = progress.validation_accuracy_history[-1] if progress.validation_accuracy_history else 0.5
            train_loss = progress.loss_history[-1] if progress.loss_history else val_loss
            train_acc = progress.accuracy_history[-1] if progress.accuracy_history else 0.5
            overfit = max(0.0, train_acc - val_acc)
        else:
            val_loss = 0.5
            val_acc = 0.5
            train_acc = 0.5
            overfit = 0.0

        param_count = sum(p.numel() for p in model.parameters())

        return PerformanceMetrics(
            accuracy=val_acc,
            loss=val_loss,
            training_accuracy=train_acc,
            validation_accuracy=val_acc,
            validation_loss=val_loss,
            convergence_rate=0.8 if progress and progress.convergence_status == ConvergenceStatus.IMPROVING else 0.4,
            overfitting_risk=min(0.5, overfit),
            response_time_ms=response_time_ms,
            memory_usage_bytes=param_count * 4,
            reliability=max(0.5, val_acc * 0.9),
            consciousness_score=val_acc * data.memory_density,
            migration_readiness=0.8 if val_acc > 0.75 else 0.3,
            last_evaluated=_time.time(),
        )

    # ------------------------------------------------------------------
    # Transfer learning
    # ------------------------------------------------------------------

    def transfer_learning(
        self,
        source: NetworkArchitecture,
        data: HemisphereDataFeed,
        target_task: str,
        target_focus: HemisphereFocus,
    ) -> NetworkArchitecture:
        """Clone source model, freeze layers, add new output head."""
        import torch.nn as nn

        src_model = source._model_ref
        if src_model is None:
            raise ValueError("Source network has no trained model")

        task_output = {"mood_prediction": 5, "trait_classification": 8}.get(target_task, 6)

        src_layers = list(src_model.children())
        frozen: list[Any] = []
        for layer in src_layers[:-2]:
            frozen.append(layer)
            for p in layer.parameters():
                p.requires_grad = False

        last_linear_input = source.topology.layers[-2].node_count
        frozen.append(nn.Linear(last_linear_input, task_output))
        frozen.append(nn.Sigmoid())

        transfer_model = nn.Sequential(*frozen).to(self._device)
        transfer_model.eval()

        nid = f"transfer_{target_task}_{_uuid.uuid4().hex[:8]}"
        self._active_models[nid] = transfer_model

        new_layers = list(source.topology.layers[:-1]) + (
            LayerDefinition(
                id="transfer_output",
                layer_type="output",
                node_count=task_output,
                activation="sigmoid",
            ),
        )
        new_topo = NetworkTopology(
            input_size=source.topology.input_size,
            layers=tuple(new_layers),
            output_size=task_output,
            total_parameters=sum(p.numel() for p in transfer_model.parameters()),
            activation_functions=tuple(l.activation for l in new_layers),
        )

        arch = NetworkArchitecture(
            id=nid,
            name=f"Transfer-{target_task}-{source.name}",
            focus=target_focus,
            topology=new_topo,
            performance=PerformanceMetrics(last_evaluated=_time.time()),
            training_progress=TrainingProgress(
                total_epochs=15,
                learning_rate=0.0001,
                batch_size=4,
            ),
            status=NetworkStatus.READY,
            substrate_migration_score=source.substrate_migration_score * 0.8,
            design_reasoning=f"Transfer learning from {source.name} for {target_task}",
            trait_influences=list(source.trait_influences),
        )
        arch._model_ref = transfer_model
        return arch

    # ------------------------------------------------------------------
    # Inference (for hemisphere signals)
    # ------------------------------------------------------------------

    def infer(self, network_id: str, input_vec: list[float]) -> list[float]:
        """Run a single forward pass and return output as a list."""
        import torch

        model = self._active_models.get(network_id)
        if model is None:
            return []
        with self._training_lock:
            model.eval()
            t = torch.tensor([input_vec], dtype=torch.float32, device=self._device)
            with torch.no_grad():
                out = model(t)
            return out.detach().cpu().squeeze().tolist()

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def save_model(self, network_id: str, path: str) -> bool:
        import torch

        model = self._active_models.get(network_id)
        if model is None:
            return False
        torch.save(model.state_dict(), path)
        return True

    def load_model(self, network_id: str, topology: NetworkTopology, path: str) -> Any:
        import torch

        model = self.build_model(topology)
        model.load_state_dict(torch.load(path, map_location=self._device, weights_only=True))
        model.eval()
        self._active_models[network_id] = model
        return model

    def remove_model(self, network_id: str) -> None:
        self._active_models.pop(network_id, None)

    def dispose(self) -> None:
        self._is_disposed = True
        self._active_models.clear()

    def get_model_count(self) -> int:
        return len(self._active_models)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_trait_influences(traits: tuple[str, ...] | list[str]) -> list[TraitInfluence]:
    return [
        TraitInfluence(
            trait=t,
            influence=0.3 + (hash(t) % 40) / 100.0,
            manifestation=f"{t} trait influences network topology",
            architectural_impact=(f"{t.lower()}_layer_sizing", f"{t.lower()}_activation_choice"),
        )
        for t in traits
    ]


def _migration_score(perf: PerformanceMetrics, data: HemisphereDataFeed) -> float:
    base = perf.accuracy
    stability_bonus = perf.reliability * 0.2
    complexity_bonus = min(0.1, data.memory_density * 0.2)
    return min(1.0, base + stability_bonus + complexity_bonus)


def _generate_name(
    data: HemisphereDataFeed,
    decision: DesignDecision,
    focus: HemisphereFocus,
) -> str:
    dominant = data.traits[0] if data.traits else "Neural"
    mood = data.mood.capitalize()
    strat = decision.design_strategy.value.capitalize()
    return f"{dominant}-{mood}-{strat}-{focus.value.capitalize()}"
