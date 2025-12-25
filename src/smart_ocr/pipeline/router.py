"""Routing logic for selecting OCR engines."""

from typing import Callable, Mapping

from smart_ocr.core.config import AgentConfig, EngineType


class EngineRouter:
    """Decide which engine to use for primary and fallback stages."""

    def __init__(
        self,
        config: AgentConfig,
        engines: Mapping[EngineType, object],
    ) -> None:
        self.config = config
        self.engines = engines

    def _enabled(self, engine: EngineType) -> bool:
        """Whether an engine is enabled in config."""
        try:
            return bool(self.config.get_engine_config(engine).enabled)
        except Exception:
            # If config doesn't know about the engine, treat it as disabled.
            return False

    def _available(self, engine: EngineType) -> bool:
        """Whether an engine is enabled and currently available."""
        if not self._enabled(engine):
            return False
        impl = self.engines.get(engine)
        if not impl:
            return False
        try:
            return bool(impl.is_available())
        except Exception:
            return False

    def select_primary(
        self,
        doc_type,
        warn: Callable[[str], None] | None = None,
    ) -> EngineType:
        """Select primary engine based on doc type and overrides."""
        if self.config.use_primary_override:
            override = self.config.primary_engine
            if self._available(override):
                return override
            if warn:
                warn(f"Primary override '{override.value}' not available; using automatic selection")

        from smart_ocr.core.document import DocumentType

        # Prefer local engines first; academic docs prefer Nougat when available.
        preference: list[EngineType] = []
        if doc_type == DocumentType.ACADEMIC:
            preference.extend([EngineType.NOUGAT, EngineType.DEEPSEEK])
        else:
            preference.extend([EngineType.DEEPSEEK, EngineType.NOUGAT])
        # Then cheap cloud, then other cloud.
        preference.extend([EngineType.GEMINI, EngineType.MISTRAL])

        for engine_type in preference:
            if self._available(engine_type):
                return engine_type

        # Nothing available.
        enabled = [e.value for e in EngineType if self._enabled(e)]
        raise RuntimeError(
            "No OCR engines are available. "
            f"Enabled in config: {enabled or 'none'}. "
            "Check dependencies/API keys/Ollama, or enable at least one engine."
        )

    def select_fallback(
        self,
        primary: EngineType,
        warn: Callable[[str], None] | None = None,
    ) -> EngineType | None:
        """Select fallback engine (different from primary, prefer cheaper)."""
        if self.config.use_fallback_override:
            override = self.config.fallback_engine
            if override == primary:
                if warn:
                    warn("Fallback override matches primary; using automatic fallback")
            elif self._available(override):
                return override
            elif warn:
                warn(f"Fallback override '{override.value}' not available; using automatic fallback")

        preference = [EngineType.GEMINI, EngineType.MISTRAL, EngineType.DEEPSEEK, EngineType.NOUGAT]

        for engine_type in preference:
            if engine_type != primary and self._available(engine_type):
                return engine_type

        return None

    def select_cross_check(self, primary: EngineType) -> EngineType | None:
        """Select a secondary local engine for cross-checking flagged pages."""
        candidates = [EngineType.NOUGAT, EngineType.DEEPSEEK]
        for engine_type in candidates:
            if engine_type == primary:
                continue
            if not self._enabled(engine_type):
                continue
            engine = self.engines.get(engine_type)
            if not engine:
                continue
            caps = getattr(engine, "capabilities", None)
            if caps and getattr(caps, "is_local", False) and self._available(engine_type):
                return engine_type
        return None
