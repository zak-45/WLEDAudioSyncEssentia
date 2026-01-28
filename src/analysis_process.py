# src/analysis_process.py
"""Worker process that performs all heavy audio analysis and OSC output.

This module defines the multiprocessing entry point used to run genre
classification, mood estimation, and lighting control logic in a separate
process. By isolating the analysis pipeline here, the main application can
capture audio and handle UI or networking without being blocked by model
inference or visualization work.
"""

def run_analysis_process(
    audio_queue,
    cfg_path,
    osc_ip,
    osc_port,
    osc_path,
    use_macro,
    macro_agg,
    debug,
    visual_enabled,
    aux,
    aux_mood,
    shutdown_event,
    silent_event,
):
    """
    Windows-safe multiprocessing entry point.
    ALL heavy objects are created inside this function.
    """

    # --- imports MUST be inside ---
    from src.runtime_config import RuntimeConfig
    from src.osc_sender import OSCSender
    from src.visual_debug import VisualDebugOverlay
    from src.analysis_process_core import AnalysisCore

    #import xtra.tracetool.tracetool

    cfg = RuntimeConfig(cfg_path)

    osc = OSCSender(
        ip=osc_ip,
        port=osc_port,
        path=osc_path,
    )

    visual = VisualDebugOverlay() if visual_enabled else None

    analysis = AnalysisCore(
        audio_queue=audio_queue,
        cfg=cfg,
        osc=osc,
        visual=visual,
        use_macro=use_macro,
        macro_agg=macro_agg,
        debug=debug,
        aux=aux,
        activate_buffer=cfg.ACTIVATE_BUFFER,
        aux_mood=aux_mood,
        shutdown_event=shutdown_event,
        silent_event=silent_event
    )

    print("🧠 Analysis process ready")
    analysis.run()
