"""
Log de juguete de 12 trazas usado como ejemplo trabajado en CLAUDE.md, Seccion 12.4.1.

Sirve para validar a mano el pipeline de calculo de uplift/score sin depender de
MINERful, Simod ni Prosimos -- las secuencias y el PCE de cada traza estan dados
directamente (no se derivan de timestamps), tal como se documento en la conversacion
con el tutor.
"""

TOY_SEQUENCES: dict[str, list[str]] = {
    "t1": ["A", "B", "C", "D", "F"],
    "t2": ["A", "C", "B", "D", "F"],
    "t3": ["A", "B", "D", "C", "F"],
    "t4": ["A", "C", "D", "B", "F"],
    "t5": ["A", "B", "C", "D", "E", "F"],
    "t6": ["A", "D", "C", "B", "F"],
    "t7": ["A", "B", "C", "F"],
    "t8": ["A", "C", "B", "F"],
    "t9": ["A", "B", "D", "F"],
    "t10": ["A", "D", "B", "C", "F"],
    "t11": ["A", "B", "C", "D", "E", "F"],
    "t12": ["A", "C", "B", "D", "E", "F"],
}

TOY_PCE: dict[str, float] = {
    "t1": 82,
    "t2": 40,
    "t3": 75,
    "t4": 35,
    "t5": 88,
    "t6": 30,
    "t7": 90,
    "t8": 38,
    "t9": 70,
    "t10": 33,
    "t11": 95,
    "t12": 92,
}

# Grupos "cumple"/"no cumple" documentados a mano en CLAUDE.md Seccion 12.4.1, para
# la candidata Precedence(B, D) -- usados como oraculo en los tests.
PRECEDENCE_B_D_CUMPLE = {"t1", "t2", "t3", "t5", "t9", "t11", "t12"}
PRECEDENCE_B_D_NO_CUMPLE = {"t4", "t6", "t7", "t8", "t10"}
