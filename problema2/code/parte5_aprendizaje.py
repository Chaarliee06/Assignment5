"""
parte5_aprendizaje.py - Clientes que APRENDEN (bandits y Q-learning) y un
guardia que aprende su politica con PPO (Stable-Baselines3).

(a) Bandits. Un cliente elige comprar, robar o irse sin conocer la
    probabilidad p de que lo atrapen:
        comprar: v - precio = L = 1 (+ ruido N(0,1))
        robar:   v = 10, o v - f = -10 si lo atrapan (prob. p)
        irse:    0 (+ ruido N(0,1))
    Robar conviene si 10 - 20 p > 1, es decir si p < p* = (v - L)/f = 0.45.
    Se comparan e-greedy, UCB1 y Thompson sampling con p = 0.30 y p = 0.60.

(b) Q-learning. La tienda tiene 3 zonas (v = 10, 6, 3). Al entrar, el
    cliente VE en que zona esta el guardia (g0); va a una zona y luego decide
    comprar, robar o irse. Mientras tanto el guardia se mueve a g1 segun la
    politica de la tienda. Si roba en la zona del guardia lo atrapan con
    prob. 0.9; en las demas, la camara lo atrapa con prob. 0.05.

(c) PPO. La tienda aprende la politica del guardia (a que zona moverse, dado
    donde esta) contra clientes que siguen aprendiendo con Q-learning.
"""
import os
import resource
import sys
import time

import numpy as np
import pandas as pd

from comun import SEMILLAS, COLORES, banda, guardar_csv, guardar_fig, guardar_json, memoria_pico, plt

T_VISITAS = 20_000
V, PRECIO, MULTA = 10.0, 9.0, 20.0
L = V - PRECIO                                    # lo que gana comprando = 1
P_ESTRELLA = (V - L) / MULTA                      # = 0.45


# ---------------------------------------------------------------------------
# (a) Bandits
# ---------------------------------------------------------------------------
def medias_brazos(p):
    return np.array([V - PRECIO, V - MULTA * p, 0.0])          # comprar, robar, irse


def recompensa(a, p, rng):
    if a == 0:
        return V - PRECIO + rng.normal()
    if a == 1:
        return V - MULTA * (rng.random() < p)
    return rng.normal()


def bandit(algoritmo, p, T, seed, eps=0.1):
    rng = np.random.default_rng(seed)
    n, suma = np.zeros(3), np.zeros(3)
    mu = medias_brazos(p)
    regret, robos = np.empty(T), np.empty(T)
    acum = 0.0
    for t in range(T):
        if t < 3:
            a = t                                               # probar cada brazo una vez
        elif algoritmo == "e-greedy":
            a = rng.integers(3) if rng.random() < eps else int(np.argmax(suma / n))
        elif algoritmo == "UCB1":
            # recompensas reescaladas a [0, 1] (van de -10 a 10 aprox.)
            media01 = (suma / n + 12) / 24
            a = int(np.argmax(media01 + np.sqrt(2 * np.log(t) / n)))
        else:                                                   # Thompson (Gaussiano, sigma = 10)
            a = int(np.argmax(rng.normal(suma / n, 10.0 / np.sqrt(n))))
        r = recompensa(a, p, rng)
        n[a] += 1; suma[a] += r
        acum += mu.max() - mu[a]
        regret[t], robos[t] = acum, a == 1
    return regret, robos


# ---------------------------------------------------------------------------
# (b) Tienda con 3 zonas y clientes Q-learning
# ---------------------------------------------------------------------------
VALORES = np.array([10.0, 6.0, 3.0])
P_GUARDIA, P_CAMARA = 0.9, 0.05
UMBRAL = (VALORES - L) / MULTA                 # (v - L)/f: prob. minima para disuadir


class ClienteQ:
    """Q-learning tabular. Estado 1: zona del guardia g0 (3). Accion 1: ir a la
    zona 0, 1, 2 o irse (3). Estado 2: (g0, zona). Accion 2: comprar, robar, irse."""

    def __init__(self, rng, alfa_min=0.02, eps=0.1):
        self.Q1, self.N1 = np.zeros((3, 4)), np.zeros((3, 4))
        self.Q2, self.N2 = np.zeros((3, 3, 3)), np.zeros((3, 3, 3))
        self.rng, self.alfa_min, self.eps = rng, alfa_min, eps

    def _paso(self, N, idx):
        """Tasa de aprendizaje 1/n (promedio), con minimo alfa_min para poder
        seguir adaptandose si la tienda cambia su politica."""
        N[idx] += 1
        return max(1.0 / N[idx], self.alfa_min)

    def _elige(self, q):
        return self.rng.integers(len(q)) if self.rng.random() < self.eps else int(np.argmax(q))

    def visita(self, g0, mover_guardia):
        z = self._elige(self.Q1[g0])
        if z == 3:                                                  # se va sin entrar a ninguna zona
            self.Q1[g0, 3] += self._paso(self.N1, (g0, 3)) * (0.0 - self.Q1[g0, 3])
            return dict(zona=-1, robo=False, perdida=0.0, g1=mover_guardia())
        a = self._elige(self.Q2[g0, z])
        g1 = mover_guardia()
        perdida = 0.0
        if a == 0:
            r = 1.0 + self.rng.normal()                             # comprar: v - precio = 1
        elif a == 1:
            atrapado = self.rng.random() < (P_GUARDIA if g1 == z else P_CAMARA)
            r = VALORES[z] - MULTA * atrapado
            perdida = 0.0 if atrapado else VALORES[z]
        else:
            r = self.rng.normal()
        # actualizacion de Q-learning (gamma = 1, episodio de 2 pasos)
        self.Q2[g0, z, a] += self._paso(self.N2, (g0, z, a)) * (r - self.Q2[g0, z, a])
        self.Q1[g0, z] += self._paso(self.N1, (g0, z)) * (self.Q2[g0, z].max() - self.Q1[g0, z])
        return dict(zona=z, robo=a == 1, perdida=perdida, g1=g1)


def politica_stackelberg():
    """Cobertura minima para disuadir todas las zonas, repartiendo lo que sobra.
    Prob. de atrapar en la zona z: P_GUARDIA c_z + P_CAMARA (1 - c_z) >= umbral_z."""
    c = np.maximum((UMBRAL - P_CAMARA) / (P_GUARDIA - P_CAMARA), 0)
    return c / c.sum()


POLITICAS = {
    "fijo en zona 1": lambda g, rng: 0,
    "rotación 1→2→3": lambda g, rng: (g + 1) % 3,
    "uniforme": lambda g, rng: int(rng.integers(3)),
    "Stackelberg (mixta)": lambda g, rng: int(rng.choice(3, p=politica_stackelberg())),
}


def simular_politica(nombre, seed, T):
    rng = np.random.default_rng(seed)
    cli = ClienteQ(rng)
    g = 0
    robos, perdidas = np.empty(T), np.empty(T)
    for t in range(T):
        res = cli.visita(g, lambda: POLITICAS[nombre](g, rng))
        g = res["g1"]
        robos[t], perdidas[t] = res["robo"], res["perdida"]
    return robos, perdidas


# ---------------------------------------------------------------------------
# (c) PPO para el guardia
# ---------------------------------------------------------------------------
def crear_entorno(seed, largo_ep=200):
    import gymnasium as gym
    from gymnasium import spaces

    class TiendaGuardia(gym.Env):
        """Cada paso es una visita. Observacion: zona actual del guardia
        (one-hot) + tasa reciente de robos por zona. Accion: zona a la que se
        mueve el guardia. Recompensa: -valor robado."""

        def __init__(self):
            self.observation_space = spaces.Box(0, 1, (6,), np.float32)
            self.action_space = spaces.Discrete(3)
            self.rng = np.random.default_rng(seed)
            self.cli = ClienteQ(self.rng)                 # el cliente aprende durante todo el entrenamiento
            self.g, self.t, self.reciente = 0, 0, np.zeros(3)
            self.log_robos, self.log_perdidas = [], []

        def _obs(self):
            return np.concatenate([np.eye(3)[self.g], self.reciente]).astype(np.float32)

        def reset(self, seed=None, options=None):
            self.t = 0
            return self._obs(), {}

        def step(self, accion):
            res = self.cli.visita(self.g, lambda: int(accion))
            self.g = res["g1"]
            self.reciente *= 0.99
            if res["robo"]:
                self.reciente[res["zona"]] += 0.01
            self.log_robos.append(res["robo"]); self.log_perdidas.append(res["perdida"])
            self.t += 1
            return self._obs(), -res["perdida"], False, self.t >= largo_ep, {}

    return TiendaGuardia()


def entrenar_ppo(seed, pasos):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    env = crear_entorno(seed)
    # gamma = 0: cada visita se paga sola (no hay recompensa futura que planear)
    modelo = PPO("MlpPolicy", env, n_steps=1000, batch_size=200, n_epochs=10, learning_rate=1e-3,
                 ent_coef=0.02, gamma=0.0, seed=seed, verbose=0, device="cpu")
    modelo.learn(total_timesteps=pasos)
    # distribucion final de la politica para cada zona actual del guardia
    dist = []
    for g in range(3):
        obs = np.concatenate([np.eye(3)[g], env.reciente]).astype(np.float32)
        d = modelo.policy.get_distribution(modelo.policy.obs_to_tensor(obs)[0])
        dist.append(d.distribution.probs.detach().cpu().numpy()[0])
    return np.array(env.log_robos[:pasos], float), np.array(env.log_perdidas[:pasos]), np.array(dist)


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 2 ** 10      # macOS da bytes, Linux KB


def movil(x, w=500):
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[w:] - c[:-w]) / w


# ---------------------------------------------------------------------------
def correr(sintetico=False, rapido=False):
    print("[Parte 5] Bandits, Q-learning y PPO ...", flush=True)
    semillas = SEMILLAS[:3] if rapido else SEMILLAS
    T = 5000 if rapido else T_VISITAS
    resumen = dict(T=T, p_estrella=P_ESTRELLA, semillas=len(semillas))

    # ---------------- (a) bandits ----------------
    algs = ["e-greedy", "UCB1", "Thompson"]
    filas, reg, rob = [], {}, {}
    for p in (0.30, 0.60):
        for alg in algs:
            for s in semillas:
                t0 = time.perf_counter()
                r, b = bandit(alg, p, T, s)
                filas.append(dict(p=p, algoritmo=alg, semilla=s, regret_final=r[-1], tasa_robo=b.mean(),
                                  tasa_robo_final=b[-1000:].mean(), tiempo_s=time.perf_counter() - t0))
                reg.setdefault((p, alg), []).append(r)
                rob.setdefault((p, alg), []).append(movil(b))
            _, mem = memoria_pico(bandit, alg, p, T, 0)
            for f in filas[-len(semillas):]:
                f["memoria_mb"] = mem
    band = pd.DataFrame(filas)
    guardar_csv(band, "p5_bandits")
    # barrido de p: ¿cuando deja de robar el cliente? (UCB1)
    ps = np.round(np.arange(0.10, 0.81, 0.05), 2)
    barrido = pd.DataFrame([dict(p=p, semilla=s, tasa_robo=bandit("UCB1", p, 5000, s)[1][-1000:].mean())
                            for p in ps for s in semillas])
    guardar_csv(barrido, "p5_barrido_p")
    resumen["bandits"] = band.groupby(["p", "algoritmo"]).agg(
        regret=("regret_final", "mean"), regret_std=("regret_final", "std"),
        robo=("tasa_robo", "mean"), robo_final=("tasa_robo_final", "mean"),
        tiempo=("tiempo_s", "mean"), memoria=("memoria_mb", "mean")).round(4).reset_index().to_dict("records")
    print("  bandits: listo", flush=True)

    # ---------------- (b) Q-learning con politicas fijas ----------------
    filas, curvas = [], {}
    for nombre in POLITICAS:
        for s in semillas:
            t0 = time.perf_counter()
            b, l = simular_politica(nombre, s, T)
            filas.append(dict(politica=nombre, semilla=s, tasa_robo=b.mean(), tasa_robo_final=b[-2000:].mean(),
                              perdida_media=l.mean(), perdida_final=l[-2000:].mean(),
                              tiempo_s=time.perf_counter() - t0))
            curvas.setdefault(nombre, []).append(movil(b))
    ql = pd.DataFrame(filas)
    resumen["stackelberg_cobertura"] = politica_stackelberg().round(4)
    resumen["umbrales"] = UMBRAL
    print("  Q-learning: listo", flush=True)

    # ---------------- (c) PPO ----------------
    try:
        import stable_baselines3  # noqa: F401
        hay_sb3 = True
    except ImportError:
        hay_sb3 = False
        print("  (no esta instalado stable-baselines3: se omite PPO)")
    if hay_sb3:
        dists = []
        for s in semillas:
            t0 = time.perf_counter()
            b, l, dist = entrenar_ppo(s, T)
            filas.append(dict(politica="PPO (aprendida)", semilla=s, tasa_robo=b.mean(),
                              tasa_robo_final=b[-2000:].mean(), perdida_media=l.mean(),
                              perdida_final=l[-2000:].mean(), tiempo_s=time.perf_counter() - t0,
                              memoria_rss_mb=rss_mb()))
            curvas.setdefault("PPO (aprendida)", []).append(movil(b))
            dists.append(dist)
            print(f"    PPO semilla {s}: robo final {b[-2000:].mean():.3f}", flush=True)
        resumen["ppo_distribucion_media"] = np.mean(dists, 0).round(3)
        resumen["ppo_cobertura_media"] = np.mean(dists, 0).mean(0).round(3)
    ql = pd.DataFrame(filas)
    guardar_csv(ql, "p5_guardia")
    resumen["guardia"] = ql.groupby("politica").agg(
        robo=("tasa_robo", "mean"), robo_std=("tasa_robo", "std"),
        robo_final=("tasa_robo_final", "mean"), robo_final_std=("tasa_robo_final", "std"),
        perdida_final=("perdida_final", "mean"), perdida_final_std=("perdida_final", "std"),
        tiempo=("tiempo_s", "mean")).round(4).reset_index().to_dict("records")

    # ---------------- Figuras ----------------
    x = np.arange(1, T + 1)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.8))
    for k, alg in enumerate(algs):
        banda(ax[0], x, reg[(0.30, alg)], f"{alg}", COLORES[k])
        banda(ax[1], x, reg[(0.60, alg)], f"{alg}", COLORES[k])
    ax[0].set_title(f"(a) regret, p = 0.30 < p* = {P_ESTRELLA:.2f}"); ax[1].set_title("(b) regret, p = 0.60 > p*")
    for a in ax[:2]:
        a.set_xlabel("visitas"); a.set_ylabel("regret acumulado")
    ax[0].legend(fontsize=7)
    g = barrido.groupby("p").tasa_robo.agg(["mean", "std"])
    ax[2].errorbar(g.index, g["mean"], g["std"], marker="o", ms=3, capsize=2)
    ax[2].axvline(P_ESTRELLA, color="k", ls="--", lw=1, label="p* = 0.45")
    ax[2].set_xlabel("probabilidad de ser atrapado p"); ax[2].set_ylabel("fracción de robos (final)")
    ax[2].set_title("(c) UCB1: ¿cuándo deja de robar?"); ax[2].legend(fontsize=7)
    fig.tight_layout()
    guardar_fig(fig, "p5_bandits")

    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    for k, (nombre, cs) in enumerate(curvas.items()):
        banda(ax, np.arange(500, T + 1), cs, nombre, COLORES[k])
    ax.set_xlabel("visitas de clientes"); ax.set_ylabel("tasa de robo (ventana de 500)")
    ax.legend(fontsize=7)
    guardar_fig(fig, "p5_guardia")
    guardar_json(resumen, "p5_resumen")
    return resumen


if __name__ == "__main__":
    correr(sintetico="--sintetico" in sys.argv, rapido="--rapido" in sys.argv)
