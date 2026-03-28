"""
Herebot Curiosity Engine v0.4 - Information-Theoretic Curiosity & Intrinsic Motivation
Integrated with real CI/Phonotactic/Diagnostic engines from `backend.ci`.

This system implements cutting-edge, human-level curiosity:

1. **Hierarchical Dirichlet Process** - latent concept sharing.
2. **Gaussian Process Uncertainty** - active learning via predictive variance.
3. **Bayesian Surprise** - KL divergence between prior and posterior.
4. **Intrinsic Reward** - prediction error / knowledge gain as reinforcement.
5. **Multi-factor meta-curiosity** - adaptively weighted by learned utility.
6. **Reasoning engine** - co-occurrence hypotheses and query planning.
7. **Persistent state** - lifelong growth.

All maths is probabilistic; decisions are sampled from posteriors.
# Based on concepts from:
# - Horswill, I. (2015). "Computational Guilt". AIIDE Workshop.
# - Christiano et al. (2017). "Deep Reinforcement Learning from Human Preferences". NIPS.
# - Oudeyer, P-Y. & Kaplan, F. (2007). "What is intrinsic motivation? A typology of computational approaches".
# - Schmidhuber, J. (1991). "A possibility for implementing curiosity and boredom in model-building neural controllers".
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from importglobal import *
from functionsglobal import *
from concepts.concepts_global import *
from traits.traits_global import *
from components_global import *
from core.mind import *

# ------------------------------------------------------------
# Idle Reasoner (background thread) – now with long-idle introspection
# ------------------------------------------------------------
class IdleReasoner(threading.Thread):
    def __init__(self, engine: CuriosityEngine,
                 short_idle_threshold: float = None,
                 long_idle_threshold: float = None):
        super().__init__(daemon=True)
        self.engine = engine
        # Read thresholds from config if available
        if engine.config_manager:
            self.short_thresh = engine.config_manager.get('idle_reasoner.short_idle_threshold', 8.0)
            self.long_thresh = engine.config_manager.get('idle_reasoner.long_idle_threshold', 600.0)
        else:
            self.short_thresh = short_idle_threshold if short_idle_threshold is not None else 8.0
            self.long_thresh = long_idle_threshold if long_idle_threshold is not None else 600.0
        self.running = True
        self.long_idle_done = False   # reset after interaction

    def update_parameters_from_config(self):
        """Reload thresholds from config (hot reload)."""
        if self.engine.config_manager:
            self.short_thresh = self.engine.config_manager.get('idle_reasoner.short_idle_threshold', 8.0)
            self.long_thresh = self.engine.config_manager.get('idle_reasoner.long_idle_threshold', 600.0)

    def run(self):
        while self.running:
            try:
                time.sleep(2)
                idle_time = time.time() - self.engine.last_interaction_time

                # Short idle → simple meta-thought
                if idle_time > self.short_thresh:
                    thought = self.engine.idle_reflection()
                    print(f"\n⏳ [Idle at {time.strftime('%H:%M:%S')}]\n{thought}\n")
                    self.engine.tracker.record(self.engine, trigger="idle_short")
                    time.sleep(self.short_thresh)

                # Long idle → deep state introspection AND all idle tasks
                if idle_time > self.long_thresh and not self.long_idle_done:
                    self.engine.perform_state_introspection()
                    self.engine.perform_all_idle_tasks()
                    try:
                        self.engine._save_state()
                    except Exception:
                        pass
                    self.engine.tracker.record(self.engine, trigger="idle_long")
                    self.long_idle_done = True
            except Exception:
                print("\n[IdleReasoner ERROR — Full Traceback]")
                print(traceback.format_exc())
                time.sleep(5)

    def reset_long_idle_flag(self):
        """Call when user interacts to allow a new long-idle cycle."""
        self.long_idle_done = False
# ------------------------------------------------------------
# Command-Line Interface
# ------------------------------------------------------------
def main():
    print("=" * 80)
    print(" Herebot Curiosity Engine v0.9 - Information-Theoretic & Intrinsically Motivated")
    print("Integrated with backend.ci ")
    print("State persists in ./curiosity_state/ - the system grows over time.")
    print("with: Gaussian Process uncertainty, Bayesian surprise, Intrinsic Reward.")
    print("Type your message or press Ctrl+C to exit.")
    print("=" * 80)

    engine = CuriosityEngine(load_state=True)
    idle_reasoner = IdleReasoner(engine, short_idle_threshold=8.0, long_idle_threshold=600.0)
    engine._idle_reasoner = idle_reasoner 
    idle_reasoner.start() 
    try:
        while True:
            try:
                user_input = input("\n Herebot> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    print("Shutting down.")
                    break

                result = engine.process_input(user_input)

                print("\n--- Curiosity Output ---")
                for line in result["curiosity_artifacts"]:
                    print(line)
                print("------------------------\n")
            except Exception:
                import traceback
                traceback.print_exc()
                print("Attempting to recover and continue...")
                # Optionally save emergency state
                try:
                    engine._save_state()
                except:
                    pass
                continue
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
    finally:
        idle_reasoner.running = False
        engine.tracker.save_data()
        engine.tracker.plot_all(engine)
if __name__ == "__main__":
    main()