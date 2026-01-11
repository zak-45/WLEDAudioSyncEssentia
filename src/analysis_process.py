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
    color1,
    debug,
    visual_enabled,
    aux,
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
        color1=color1,
        debug=debug,
        aux=aux,
        activate_buffer=cfg.ACTIVATE_BUFFER,
        rt_mood_lift=cfg.RT_MOOD_LIFT
    )

    print("🧠 Analysis process ready")
    analysis.run()
