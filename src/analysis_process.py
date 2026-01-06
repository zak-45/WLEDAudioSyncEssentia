# src/analysis_process.py

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
    last_beat_time,
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
        last_beat_time=last_beat_time,
    )

    print("🧠 Analysis process ready")
    analysis.run()
