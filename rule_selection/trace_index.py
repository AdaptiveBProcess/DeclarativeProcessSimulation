"""
Evaluador de cumplimiento de reglas Declare, precomputado por traza.

Reimplementa la semantica de las 8 ramas de
`support_modules.traces_evaluation.evaluate_condition_list` (required, not_allowed,
eventually, directly, precedence, succession, coexistence, NotChainSuccession), pero
contra estructuras precomputadas una sola vez por traza (nodos, sucesores directos,
alcanzabilidad hacia adelante, posiciones de ocurrencia) en vez de reconstruir un
`networkx.DiGraph` en cada llamada.

No modifica `evaluate_condition`/`evaluate_condition_list` -- el pipeline de
alucinacion LSTM depende de esas funciones tal cual (ver CLAUDE.md Seccion 12.2).

A diferencia del `ac_index` (nombre->entero) que usa el modulo original, aca se
trabaja directamente con nombres de actividad como strings: el indice entero del
modulo original nunca se usa fuera de esas dos funciones, asi que no aporta nada
mantenerlo aca.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Los 8 valores de `rule` que entiende TraceIndex.evaluate -- exactamente los mismos
# strings que espera `support_modules.traces_evaluation.evaluate_condition_list`.
SUPPORTED_RULES = frozenset(
    {
        "required",
        "not_allowed",
        "eventually",
        "directly",
        "precedence",
        "succession",
        "coexistence",
        "NotChainSuccession",
    }
)


def build_trace_sequences(df_log: pd.DataFrame) -> dict[str, list[str]]:
    """
    A partir de un log normalizado (columnas `case:concept:name`, `concept:name`,
    `time:timestamp`, `lifecycle:transition` en minusculas, con pares start/complete
    ya completados via `corregir_lifecycle_incompleto`), reconstruye por traza la
    secuencia de actividades con **una sola entrada por ocurrencia**.

    El log trae 2 filas por ocurrencia de actividad (start y complete). Si se
    tomaran las filas tal cual ordenadas por timestamp, cada actividad quedaria
    duplicada en la secuencia, lo que corrompe toda regla basada en el grafo
    (directly, NotChainSuccession, auto-loops). Se replica el mismo pivot por
    `cumcount()` que ya usa `generar_analisis_por_traza`
    (MINERful/modelo_declarativo_enriquecedio_v2.py, ~lineas 464-475) sin modificar
    esa funcion, porque es compartida con el pipeline legacy.
    """
    df_temp = df_log.copy()
    df_temp["instance"] = df_temp.groupby(
        ["case:concept:name", "concept:name", "lifecycle:transition"]
    ).cumcount()

    df_pivoted = df_temp.pivot_table(
        index=["case:concept:name", "concept:name", "instance"],
        columns="lifecycle:transition",
        values="time:timestamp",
        aggfunc="first",
    ).reset_index()

    if "start" not in df_pivoted.columns:
        raise ValueError(
            "El log no tiene columna 'start' tras el pivot -- confirma que "
            "'lifecycle:transition' ya esta en minusculas y que se corrio "
            "corregir_lifecycle_incompleto antes de llamar build_trace_sequences."
        )

    df_pivoted = df_pivoted.sort_values(["case:concept:name", "start"])

    sequences: dict[str, list[str]] = {}
    for case_id, group in df_pivoted.groupby("case:concept:name"):
        sequences[str(case_id)] = group["concept:name"].tolist()
    return sequences


@dataclass
class TraceIndex:
    """Estructuras precomputadas de una traza, reutilizables para todas las candidatas."""

    sequence: list[str]
    nodes: set[str] = field(default_factory=set)
    direct_succ: dict[str, set[str]] = field(default_factory=dict)
    reach: dict[str, set[str]] = field(default_factory=dict)
    positions: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def from_sequence(cls, seq: list[str]) -> "TraceIndex":
        nodes = set(seq)

        direct_succ: dict[str, set[str]] = {n: set() for n in nodes}
        for a, b in zip(seq[:-1], seq[1:]):
            direct_succ[a].add(b)

        # Alcanzabilidad hacia adelante (clausura de direct_succ), una vez por nodo.
        # No incluye el propio nodo salvo que exista un ciclo real de vuelta a el --
        # coincide exactamente con la semantica de nx.ancestors (excluye el nodo).
        reach: dict[str, set[str]] = {}
        for n in nodes:
            seen: set[str] = set()
            stack = list(direct_succ[n])
            while stack:
                x = stack.pop()
                if x not in seen:
                    seen.add(x)
                    stack.extend(direct_succ.get(x, set()) - seen)
            reach[n] = seen

        positions: dict[str, list[int]] = {}
        for i, act in enumerate(seq):
            positions.setdefault(act, []).append(i)

        return cls(
            sequence=seq,
            nodes=nodes,
            direct_succ=direct_succ,
            reach=reach,
            positions=positions,
        )

    def _has_path(self, a: str, b: str) -> bool:
        """Equivalente a nx.has_path(G, a, b): True tambien si a == b (camino trivial)."""
        return a == b or b in self.reach.get(a, set())

    def evaluate(self, rule: str, arg0: str, arg1: str | None = None) -> bool:
        if rule == "eventually":
            positions_a = self.positions.get(arg0)
            if not positions_a:
                # No-vacuo: si arg0 nunca ocurre, la regla se rechaza (no se cumple vacuamente).
                return False
            last_a = positions_a[-1]
            # Basta revisar la ULTIMA ocurrencia de arg0: si arg1 aparece despues de
            # ella, aparece despues de todas las ocurrencias anteriores tambien.
            positions_b = self.positions.get(arg1, [])
            return any(idx > last_a for idx in positions_b)

        elif rule == "not_allowed":
            return arg0 not in self.nodes

        elif rule == "directly":
            if arg0 not in self.nodes:
                return False
            return arg1 in self.direct_succ.get(arg0, set())

        elif rule == "required":
            return arg0 in self.nodes

        elif rule == "precedence":
            # Se activa por la presencia de arg1 (B); si B existe, A debe existir y
            # ser su ancestro (A ->...-> B, sin contar A==B, igual que nx.ancestors).
            if arg1 not in self.nodes:
                return False
            if arg0 not in self.nodes:
                return False
            return arg1 in self.reach.get(arg0, set())

        elif rule == "succession":
            if arg0 not in self.nodes or arg1 not in self.nodes:
                return False
            path_a_b = self._has_path(arg0, arg1)
            path_b_a = self._has_path(arg1, arg0)
            return path_a_b and not path_b_a

        elif rule == "coexistence":
            return arg0 in self.nodes and arg1 in self.nodes

        elif rule == "NotChainSuccession":
            if arg0 not in self.nodes:
                return False
            is_direct_successor = arg1 in self.direct_succ.get(arg0, set())
            return not is_direct_successor

        raise ValueError(f"Regla no soportada: {rule!r} (validas: {sorted(SUPPORTED_RULES)})")


def build_trace_indexes(sequences: dict[str, list[str]]) -> dict[str, TraceIndex]:
    """Construye un TraceIndex por traza, una sola vez, para reutilizar en todas las candidatas."""
    return {case_id: TraceIndex.from_sequence(seq) for case_id, seq in sequences.items()}
