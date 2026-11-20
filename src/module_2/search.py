"""Búsqueda híbrida, similitud por ChromaDB y consulta de detalles.

La lógica es una transcripción de las funciones finales de
``semanticGameIntelligence2.ipynb``. El notebook no se ejecuta ni se modifica.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from src.module_2.loader import Module2Artifacts
from src.module_2.schemas import COLUMNAS_CANDIDATO, Intencion, TituloAmbiguo
from src.utils import to_builtin

COLS_OUT = [
    "game_id",
    "title",
    "genre",
    "release_year",
    "price_usd",
    "file_size_gb",
    "steam_positive_pct",
    "num_reviews",
    "metacritic_score",
    "owners_k",
    "peak_concurrent_players",
    "has_multiplayer",
    "is_indie",
    "early_access",
    "score",
    "sim",
    "score_pref",
    "score_base",
]

PREF_MARK = (
    r"\b(prefiero|prefiera|idealmente|mejor si|prioriza\w*|de ser posible|"
    r"ojala|me gustaria|si es posible|preferentemente|de preferencia)\b"
)
OBL_MARK = (
    r"\b(solamente|solo|unicamente|exclusivamente|debe\w*|obligatorio|"
    r"necesito|sin|con|que tenga\w*|que sea\w*|menos de|mayor a|hasta|entre|no)\b"
)
OBL_MARK_CUAL = (
    r"\b(solamente|solo|unicamente|exclusivamente|debe\w*|"
    r"obligatorio|necesito|siempre)\b"
)

BOOL_REGLAS = [
    ("has_multiplayer", 0, r"(?:sin|no)\s+(?:\w+\s+){0,2}multi\w*", "obligatorio"),
    ("has_multiplayer", 1, r"multi\w*|co-?op|jugar con amigos|en linea|online", "obligatorio"),
    ("early_access", 0, r"(?:sin|no)\s+(?:\w+\s+){0,2}acceso anticipado|(?:sin|no)\s+\w*\s*early access", "obligatorio"),
    ("early_access", 1, r"acceso anticipado|early access", "obligatorio"),
    ("is_indie", 0, r"no indie|no independiente|triple a|grandes estudios", "obligatorio"),
    ("is_indie", 1, r"indie|independientes?", "preferencia"),
    ("controller_support", 0, r"sin control|sin compatibilidad con controles|sin soporte de control", "obligatorio"),
    ("controller_support", 1, r"compatible con controles?|soporte para controles?|con control", "obligatorio"),
]

CUALITATIVAS = [
    (r"muy barat\w*|muy economic\w*|regalad\w*", "price_usd", "<=", "P10", "preferencia"),
    (r"barat\w*|economic\w*|asequible\w*", "price_usd", "<=", "P25", "preferencia"),
    (r"\bcar[oa]s?\b|\bpremium\b", "price_usd", ">=", "P75", "preferencia"),
    (r"muy bien valorad\w*|muy buenas valoraciones|excelentes valoraciones|mejor valorad\w*",
     "steam_positive_pct", ">=", "P90", "preferencia"),
    (r"bien valorad\w*|buenas valoraciones|buena valoracion|bien puntuad\w*",
     "steam_positive_pct", ">=", "P75", "preferencia"),
    (r"gran cantidad de resenas|muchas resenas|muchos reviews|gran cantidad de reviews",
     "num_reviews", ">=", "P90", "preferencia"),
    (r"bastantes resenas|bastantes reviews", "num_reviews", ">=", "P75", "preferencia"),
    (r"muy popular|muy jugad\w*|los mas jugados", "peak_concurrent_players", ">=", "P90", "preferencia"),
    (r"gran cantidad de jugadores concurrentes|muchos jugadores concurrentes|"
     r"muchos jugadores en linea|gran cantidad de jugadores",
     "peak_concurrent_players", ">=", "P90", "preferencia"),
    (r"muchos jugadores|popular|jugados?", "peak_concurrent_players", ">=", "P75", "preferencia"),
    (r"no ocupen demasiado|no ocupe demasiado|no ocupen mucho|no ocupe mucho|"
     r"que ocupen poco|que ocupe poco|poco espacio|pequen\w*|liger\w*|livian\w*|poco peso",
     "file_size_gb", "<=", "P25", "preferencia"),
]

NUM_G = r"(?P<num>\d+(?:[.,]\d+)?)"
NUM2_G = r"(?P<num2>\d+(?:[.,]\d+)?)"
UNIDAD = r"(?:usd|dolares?|gb|gigas?|anos?)"
PREFIJO1 = rf"(?P<pref>{UNIDAD})?\s*"
PREFIJO2 = rf"(?P<pref2>{UNIDAD})?\s*"
SUFIJO = rf"(?P<unidad>{UNIDAD})?"

PATRONES_NUM = [
    (rf"(?:menos de|menor a|por debajo de|debajo de|bajo)\s*(?:de\s*)?{PREFIJO1}{NUM_G}\s*{SUFIJO}", "<"),
    (rf"(?:mas de|mayor a|superior a|por encima de|encima de)\s*(?:de\s*)?{PREFIJO1}{NUM_G}\s*{SUFIJO}", ">"),
    (rf"(?:hasta|como maximo|maximo)\s*{PREFIJO1}{NUM_G}\s*{SUFIJO}", "<="),
    (rf"(?:al menos|minimo|desde|a partir de)\s*{PREFIJO1}{NUM_G}\s*{SUFIJO}", ">="),
    (rf"entre\s*{PREFIJO1}{NUM_G}\s*(?:y|a|-)\s*{PREFIJO2}{NUM2_G}\s*{SUFIJO}", "between"),
]


class SemanticSearch:
    def __init__(self, artifacts: Module2Artifacts) -> None:
        self.a = artifacts
        self.df = artifacts.dataset
        self.base = artifacts.dataset
        self.P = artifacts.percentiles
        self.embeddings = artifacts.embeddings
        self.id_a_pos = artifacts.id_a_pos
        self.missing_mode = "neutral"
        self.topes = {
            col: float(self.df[col].max())
            for col in self.df.select_dtypes(include=[np.number]).columns
        }
        self.genero_alias = [
            ("juegos de rol", "RPG"), ("role playing", "RPG"), ("rpg", "RPG"), ("rol", "RPG"),
            ("tiro", "Shooter"), ("disparos", "Shooter"), ("accion", "Action"),
            ("aventura", "Adventure"), ("simulacion", "Simulation"),
            ("estrategia", "Strategy"), ("plataformas", "Platformer"),
            ("supervivencia", "Survival"), ("rompecabezas", "Puzzle"), ("puzzle", "Puzzle"),
            ("terror", "Horror"), ("deportes", "Sports"), ("carreras", "Racing"),
        ] + [(self.normalizar(genre), genre) for genre in self.df["genre"].unique()]
        self.genero_alias = sorted(set(self.genero_alias), key=lambda item: -len(item[0]))

    @staticmethod
    def normalizar(texto: str) -> str:
        texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")
        texto = re.sub(r"[^a-z0-9]+", " ", texto.lower()).strip()
        return re.sub(r"\s+", " ", texto)

    def detectar_titulo(self, consulta: str) -> str | None:
        normalizada = self.normalizar(consulta)
        coincidencias = [
            titulo for titulo in self.df["title"].unique()
            if self.normalizar(titulo) in normalizada
        ]
        return max(coincidencias, key=len) if coincidencias else None

    def resolver_titulo(
        self,
        titulo: str,
        game_id: str | None = None,
        genero: str | None = None,
        anio: int | None = None,
        precio_max: float | None = None,
    ) -> dict:
        tit_norm = self.normalizar(titulo)
        candidatos = self.df[self.df["title"].map(self.normalizar) == tit_norm].copy()
        if game_id is not None:
            candidatos = candidatos[candidatos["game_id"] == game_id]
        if genero is not None:
            candidatos = candidatos[candidatos["genre"] == genero]
        if anio is not None:
            candidatos = candidatos[candidatos["release_year"] == anio]
        if precio_max is not None:
            candidatos = candidatos[candidatos["price_usd"] <= precio_max]
        if candidatos.empty:
            raise ValueError(f"No encontré el título '{titulo}' con esos criterios.")
        if len(candidatos) > 1:
            raise TituloAmbiguo(titulo, candidatos[COLUMNAS_CANDIDATO])
        return candidatos.iloc[0].to_dict()

    @staticmethod
    def _modo_local(txt: str, inicio: int, por_defecto: str, oblig: str = OBL_MARK) -> str:
        ctx = txt[max(0, inicio - 45):inicio]
        if re.search(PREF_MARK, ctx):
            return "preferencia"
        if re.search(oblig, ctx):
            return "obligatorio"
        return por_defecto

    def _poner(self, it: Intencion, var: str, op: str, valor, modo: str, peso: float = 1.0) -> None:
        destino = it.obligatorios if modo == "obligatorio" else it.preferencias
        regla: dict[str, Any] = {"op": op, "valor": valor, "peso": peso}
        if (
            var in self.df.columns
            and pd.api.types.is_numeric_dtype(self.df[var])
            and isinstance(valor, (int, float, np.integer, np.floating))
        ):
            regla["max"] = float(self.df[var].max())
        destino.setdefault(var, []).append(regla)

    @staticmethod
    def _marcar(txt: str, patron: str):
        match = re.search(patron, txt, re.I)
        if match:
            return match, txt[:match.start()] + " " * (match.end() - match.start()) + txt[match.end():]
        return None, txt

    @staticmethod
    def _variable_numerica(unidad: str | None, ctx: str, numeros: list[float]) -> str | None:
        u = (unidad or "").lower()
        if u.startswith("usd") or u.startswith("dol"):
            return "price_usd"
        if u.startswith("gb") or u.startswith("gig"):
            return "file_size_gb"
        if u.startswith("ano"):
            return "release_year"
        if numeros and all(1900 <= n <= 2100 for n in numeros):
            return "release_year"
        if re.search(r"precio|usd|dolar|costar|pagar", ctx):
            return "price_usd"
        if re.search(r"gb|giga|tamano|ocupar|peso|espacio", ctx):
            return "file_size_gb"
        if re.search(r"ano|lanzamiento|estreno|reciente|antigu", ctx):
            return "release_year"
        if re.search(r"concurren|jugando|en linea", ctx):
            return "peak_concurrent_players"
        if re.search(r"resenas|reviews", ctx):
            return "num_reviews"
        if re.search(r"propietarios|owners", ctx):
            return "owners_k"
        if re.search(r"metacritic", ctx):
            return "metacritic_score"
        return None

    def parsear(self, consulta: str) -> Intencion:
        txt = self.normalizar(consulta)
        it = Intencion(consulta=consulta)
        it.titulo = self.detectar_titulo(consulta)

        semantic_mode = bool(re.search(
            r"\b(parecid\w*|similar\w*|semejante|tipo|del mismo estilo)\b", txt
        ))
        sin_necesidad = bool(re.search(
            r"(?:aunque|pero)\s+no|no necesariamente|sin importar|abierto a|"
            r"no tiene que|no requiere", txt
        ))
        resto = txt

        for alias, gen in self.genero_alias:
            match, resto2 = self._marcar(resto, rf"\b{re.escape(alias)}\b")
            if match:
                destino = it.genero_semantico if semantic_mode else it.generos
                if gen not in destino:
                    destino.append(gen)
                resto = resto2

        for var, valor, patron, default in BOOL_REGLAS:
            match, resto2 = self._marcar(resto, patron)
            if match and var not in it.obligatorios and var not in it.preferencias:
                if semantic_mode and sin_necesidad:
                    it.atributos_semanticos.append((var, valor))
                else:
                    self._poner(it, var, "=", valor, self._modo_local(txt, match.start(), default))
                resto = resto2

        for patron, var, op, pk, default in CUALITATIVAS:
            match, resto2 = self._marcar(resto, patron)
            if match and var not in it.obligatorios and var not in it.preferencias:
                self._poner(
                    it, var, op, self.P[var][pk],
                    self._modo_local(txt, match.start(), default, oblig=OBL_MARK_CUAL),
                )
                resto = resto2

        for patron, op in PATRONES_NUM:
            for match in list(re.finditer(patron, resto, re.I)):
                partes = match.groupdict()
                numeros = [
                    float(partes[key].replace(",", "."))
                    for key in ("num", "num2")
                    if partes.get(key)
                ]
                unidad = partes.get("pref") or partes.get("pref2") or partes.get("unidad") or ""
                ctx = txt[max(0, match.start() - 40):match.end() + 20]
                var = self._variable_numerica(unidad, ctx, numeros)
                if var is None:
                    it.advertencias.append(f"No pude asignar un atributo a '{match.group(0)}'.")
                    continue
                valor = (numeros[0], numeros[1]) if op == "between" and len(numeros) > 1 else numeros[0]
                self._poner(it, var, op, valor, self._modo_local(txt, match.start(), "obligatorio"))
                resto = resto[:match.start()] + " " * (match.end() - match.start()) + resto[match.end():]

        if "steam_positive_pct" in it.preferencias or "steam_positive_pct" in it.obligatorios:
            it.preferencias.setdefault("num_reviews", []).append(
                {"op": ">=", "valor": self.P["num_reviews"]["P25"], "peso": 0.5}
            )
            it.advertencias.append("Valoración contextualizada con num_reviews >= P25.")
        if "peak_concurrent_players" in it.preferencias:
            it.advertencias.append(
                "'Jugadores' interpretado como concurrentes; aclarar si significa propietarios."
            )
        return it

    @staticmethod
    def _comparar(serie: pd.Series, op: str, valor) -> pd.Series:
        if op == "=":
            return serie == valor
        if op == "<":
            return serie < valor
        if op == "<=":
            return serie <= valor
        if op == ">":
            return serie > valor
        if op == ">=":
            return serie >= valor
        if op == "between":
            return serie.between(valor[0], valor[1])
        raise ValueError(f"Operador no soportado: {op}")

    def aplicar_filtros(self, df_base: pd.DataFrame, it: Intencion):
        mask = pd.Series(True, index=df_base.index)
        if it.generos:
            mask &= df_base["genre"].isin(it.generos)
        for var, reglas in it.obligatorios.items():
            for regla in reglas:
                mask &= self._comparar(df_base[var], regla["op"], regla["valor"])
        return df_base[mask].copy(), mask

    def _satisfaccion(self, x, regla: dict, var: str | None = None) -> float:
        op, v = regla["op"], regla["valor"]
        if pd.isna(x):
            return 0.5 if self.missing_mode == "neutral" else 0.0
        if op == "=":
            return float(x == v)
        if op == "between":
            lo, hi = v
            if lo <= x <= hi:
                return 1.0
            ancho = hi - lo
            if ancho <= 0:
                return 0.0
            dist = lo - x if x < lo else x - hi
            return float(max(0.0, 1 - dist / ancho))
        if not isinstance(v, (int, float)):
            return 0.5

        tope_ref = regla.get("max") or self.topes.get(var)
        tope = min(2 * v, tope_ref) if tope_ref else 2 * v

        if op in ("<", "<="):
            if var == "price_usd" and x < self.P["price_usd"]["P10"]:
                return 0.90
            if x <= v:
                return 1.0
            if tope <= v:
                return 0.0
            return float(np.clip(1 - (x - v) / (tope - v), 0, 1))
        if op in (">", ">="):
            if x >= v:
                return 1.0
            if v <= 0:
                return 1.0
            return float(np.clip(x / v, 0, 1))
        return 0.5

    def _score_preferencias(self, fila: pd.Series, it: Intencion) -> float:
        valores, pesos = [], []
        for var, reglas in it.preferencias.items():
            for regla in reglas:
                valores.append(self._satisfaccion(fila[var], regla, var))
                pesos.append(regla["peso"])
        return float(np.average(valores, weights=pesos)) if valores else 0.5

    def _score_estructural(self, cand: pd.DataFrame, it: Intencion) -> pd.Series:
        usadas = set(it.preferencias) | set(it.obligatorios)
        pop = [v for v in ["owners_k", "num_reviews", "peak_concurrent_players"] if v not in usadas]
        cal = [v for v in ["steam_positive_pct", "metacritic_score"] if v not in usadas]

        def media_rank(cols: list[str]) -> pd.Series:
            if not cols:
                return pd.Series(0.5, index=cand.index)
            return pd.concat([cand[c].rank(pct=True) for c in cols], axis=1).mean(axis=1)

        return 0.5 * media_rank(pop) + 0.5 * media_rank(cal)

    def _explicar(self, fila: pd.Series, it: Intencion) -> str:
        partes = []
        if it.generos:
            partes.append("género " + "/".join(it.generos))
        for var, reglas in it.obligatorios.items():
            for regla in reglas:
                partes.append(f"{var} {regla['op']} {regla['valor']}")
        for var, reglas in it.preferencias.items():
            for regla in reglas:
                partes.append(
                    f"{var} {regla['op']} {regla['valor']} "
                    f"(cumple {self._satisfaccion(fila[var], regla, var):.2f})"
                )
        return " · ".join(partes)

    def ranking(
        self,
        estrategia: str,
        consulta: str,
        ref: str | None = None,
        n: int = 10,
        w_base: float = 0.5,
    ) -> pd.DataFrame:
        it = self.parsear(consulta)

        if estrategia == "semantica":
            cand = self.base.copy()
        else:
            cand, _ = self.aplicar_filtros(self.base, it)
        if cand.empty:
            vacio = self.base.iloc[0:0].copy()
            vacio.attrs["candidatos"] = 0
            return vacio

        if ref or it.titulo:
            gid = ref or self.resolver_titulo(it.titulo)["game_id"]
            if gid not in self.id_a_pos:
                raise ValueError(f"game_id desconocido: {gid}")
            q = self.embeddings[self.id_a_pos[gid]]
        else:
            if self.a.embedder is None:
                raise RuntimeError(
                    "El embedder no está cargado; se necesita para consultas sin game_id de referencia."
                )
            q = self.a.embedder.encode(
                "query: " + consulta,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

        pos = cand.index.to_numpy()
        sims = self.embeddings[pos] @ q
        n_pref = sum(len(v) for v in it.preferencias.values())

        if n_pref:
            s_pref = cand.apply(lambda f: self._score_preferencias(f, it), axis=1).to_numpy()
        else:
            s_pref = np.full(len(cand), 0.5)
        s_base = self._score_estructural(cand, it).to_numpy()

        if estrategia == "estructurada":
            w_sem, w_pref = 0.0, (2.0 if n_pref else 0.0)
        elif estrategia == "semantica":
            w_sem, w_pref, w_base = 1.0, 0.0, 0.001
        elif estrategia == "hibrida":
            w_sem = 2.0 if (ref or it.titulo) else 1.0
            w_pref = min(1.0 + 0.5 * n_pref, 3.0) if n_pref else 0.0
        elif estrategia == "estatica":
            w_sem, w_pref = 1.0, (1.0 if n_pref else 0.0)
        else:
            raise ValueError(f"Estrategia desconocida: {estrategia}")

        denominador = w_sem + w_pref + w_base
        score = (w_sem * sims + w_pref * s_pref + w_base * s_base) / denominador

        out = cand.copy()
        out["sim"], out["score_pref"], out["score_base"], out["score"] = (
            sims, s_pref, s_base, score
        )
        if ref:
            out = out[out["game_id"] != ref]
        out = out.sort_values("score", ascending=False).head(n)
        out.attrs["candidatos"] = len(cand)
        return out

    def search(self, query: str, top_k: int = 10, game_id_referencia: str | None = None) -> dict:
        if game_id_referencia is not None and game_id_referencia not in self.id_a_pos:
            return {
                "status": "error",
                "error": f"game_id desconocido: {game_id_referencia}",
            }
        try:
            ranked = self.ranking("hibrida", query, ref=game_id_referencia, n=top_k)
        except TituloAmbiguo as exc:
            return {
                "status": "ambiguous",
                "message": str(exc),
                "candidates": to_builtin(exc.candidatos.to_dict("records")),
            }

        if ranked.empty:
            return {
                "status": "ok",
                "query": query,
                "strategy": "hibrida",
                "n_candidates": 0,
                "results": [],
                "message": "Ningún juego cumple las condiciones obligatorias.",
            }

        intencion = self.parsear(query)
        resultados = []
        for _, fila in ranked.iterrows():
            item = {col: to_builtin(fila[col]) for col in COLS_OUT if col in fila.index}
            explicacion = self._explicar(fila, intencion)
            if not explicacion:
                if game_id_referencia:
                    explicacion = (
                        f"similitud semántica con el juego de referencia {game_id_referencia}"
                    )
                elif intencion.titulo:
                    explicacion = f"similitud semántica con '{intencion.titulo}'"
                else:
                    explicacion = (
                        "ranking híbrido: similitud semántica, preferencias y "
                        "puntuación estructural"
                    )
            item["por_que"] = explicacion
            resultados.append(item)

        return {
            "status": "ok",
            "query": query,
            "strategy": "hibrida",
            "n_candidates": int(ranked.attrs.get("candidatos", 0)),
            "results": resultados,
            "advertencias": intencion.advertencias,
        }

    def similar(self, game_id_referencia: str, top_k: int = 10) -> dict:
        gid = str(game_id_referencia)
        if gid not in self.id_a_pos:
            raise ValueError(f"game_id desconocido: {gid}")

        query_vector = self.embeddings[self.id_a_pos[gid]].tolist()
        respuesta = self.a.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k + 1, self.a.collection.count()),
            include=["metadatas", "distances", "documents"],
        )

        resultados = []
        for rid, dist, meta, doc in zip(
            respuesta["ids"][0],
            respuesta["distances"][0],
            respuesta["metadatas"][0],
            respuesta["documents"][0],
        ):
            if rid == gid:
                continue
            resultados.append(
                {
                    "game_id": rid,
                    "title": meta.get("title"),
                    "genre": meta.get("genre"),
                    "similarity": round(1.0 - float(dist), 6),
                    "distance": round(float(dist), 6),
                    "metadata": to_builtin(meta),
                    "document": doc,
                }
            )
            if len(resultados) >= top_k:
                break

        return {
            "status": "ok",
            "source": "chromadb",
            "collection": "videojuegos_e5",
            "reference": gid,
            "results": resultados,
            "note": "Similitud coseno sobre embeddings normalizados; se excluye el juego de referencia.",
        }

    def get_details(
        self,
        game_id: str | None = None,
        title: str | None = None,
        genero: str | None = None,
        anio: int | None = None,
        precio_max: float | None = None,
    ) -> dict:
        if game_id is not None:
            filas = self.df.loc[self.df["game_id"] == str(game_id)]
            if filas.empty:
                raise ValueError(f"game_id desconocido: {game_id}")
            return {"status": "ok", "game": to_builtin(filas.iloc[0].to_dict())}

        if title is not None:
            try:
                juego = self.resolver_titulo(title, None, genero, anio, precio_max)
            except TituloAmbiguo as exc:
                return {
                    "status": "ambiguous",
                    "message": str(exc),
                    "candidates": to_builtin(exc.candidatos.to_dict("records")),
                }
            return {"status": "ok", "game": to_builtin(juego)}

        raise ValueError("Pasá game_id o title para get_game_details")
