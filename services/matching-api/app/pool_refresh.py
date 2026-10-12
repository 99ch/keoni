"""Mise à jour du vivier de CV juste avant un scoring déclenché à la main.

Volontairement sans import lourd (comme app/scoring.py) : la réindexation
incrémentale elle-même vit dans main.py (run_full_reindex), ce module ne
fait que l'orchestrer -- attendre, lancer, re-attendre -- à partir de
fonctions injectées, ce qui permet de tester toute la logique d'attente sans
installer la pile ML (voir tests/test_pool_refresh.py).
"""

import time
from typing import Callable


def refresh_pool(
    is_running: Callable[[], bool],
    run_pass: Callable[[], None],
    *,
    max_wait_seconds: float,
    poll_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Garantit qu'un passage de réindexation incrémental a eu lieu APRÈS
    l'appel, puis qu'aucun n'est plus en cours. Renvoie True si c'est le cas,
    False si `max_wait_seconds` s'est écoulé avant (le scoring peut alors
    continuer, mais sur un vivier possiblement incomplet -- à journaliser).

    Pourquoi attendre avant de lancer : run_full_reindex() rend la main
    immédiatement si un passage tourne déjà (verrou d'état). Or ce passage a
    pu démarrer AVANT l'ajout des CV qui nous intéressent -- le laisser finir
    puis en lancer un nouveau est la seule façon d'être sûr de les voir.

    Pourquoi ré-attendre après : un autre passage (cycle autonome, autre
    clic) a pu démarrer entre-temps, auquel cas notre propre appel a été
    ignoré ; on attend alors qu'il se termine plutôt que de scorer pendant
    qu'il écrit encore.
    """
    deadline = clock() + max_wait_seconds

    def wait_until_idle() -> bool:
        while is_running():
            if clock() >= deadline:
                return False
            sleep(poll_seconds)
        return True

    if not wait_until_idle():
        return False

    run_pass()

    return wait_until_idle()
