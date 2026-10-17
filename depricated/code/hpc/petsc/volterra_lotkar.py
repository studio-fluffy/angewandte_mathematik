"""
Lotka-Volterra Räuber-Beute-Modell mit PETSc Time Stepper

Löst das klassische Lotka-Volterra System von Differentialgleichungen:
    dx/dt = α*x - β*x*y  (Beute-Gleichung)
    dy/dt = δ*x*y - γ*y  (Räuber-Gleichung)

Wobei:
    x = Beute-Population
    y = Räuber-Population
    α = Wachstumsrate der Beute
    β = Prädationsrate (Räuber auf Beute)
    δ = Effizienz der Räuber (Umwandlung Beute -> Räuber)
    γ = Sterberate der Räuber
"""

from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

# Modellparameter für das Lotka-Volterra System
α, β, δ, γ = 1.5, 1.0, 1.0, 3.0  # Wachstum, Prädation, Effizienz, Sterberate

def rhs(ts, t, u, F):
    """
    Right-hand side (RHS) Funktion für das Lotka-Volterra System.
    
    Args:
        ts: PETSc Time Stepper Objekt (nicht verwendet, aber erforderlich)
        t: Aktuelle Zeit (nicht verwendet in diesem autonomen System)
        u: Aktueller Lösungsvektor [x, y] (Beute, Räuber)
        F: Output-Vektor für die Ableitungen [dx/dt, dy/dt]
    """
    x, y = u[0], u[1]  # Beute- und Räuber-Population extrahieren
    
    # Lotka-Volterra Gleichungen
    F[0] = α * x - β * x * y      # dx/dt: Beute wächst, wird aber gefressen
    F[1] = δ * x * y - γ * y      # dy/dt: Räuber profitieren von Beute, sterben natürlich

def main():
    """
    Hauptfunktion: Löst das Lotka-Volterra System und erstellt einen Plot.
    """
    # Zeitintegrations-Parameter
    t0, tf = 0.0, 20.0    # Anfangs- und Endzeit
    dt = 0.01             # Zeitschrittweite

    # PETSc Time Stepper konfigurieren
    ts = PETSc.TS().create()                           # Time Stepper erstellen
    ts.setType(PETSc.TS.Type.RK)                      # Runge-Kutta Verfahren wählen
    ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR) # Nichtlineares Problem

    # Lösungsvektor erstellen und Anfangsbedingungen setzen
    u = PETSc.Vec().createSeq(2)          # Sequentieller Vektor der Länge 2
    u.setValues([0, 1], [10.0, 5.0])      # u[0]=10 (Beute), u[1]=5 (Räuber)
    
    # Time Stepper mit Problemdefinition verbinden
    ts.setSolution(u)                      # Lösungsvektor zuweisen
    ts.setRHSFunction(rhs)                 # Rechte Seite der DGL setzen

    # Zeitintegrations-Parameter setzen
    ts.setTime(t0)                         # Startzeit
    ts.setTimeStep(dt)                     # Zeitschrittweite
    ts.setMaxTime(tf)                      # Endzeit

    # Arrays für Speicherung der Lösung initialisieren
    times = [t0]                           # Zeitpunkte
    sol = [u.getValues([0, 1])]           # Lösungswerte [x, y]

    def monitor(ts, step, time, u):
        """
        Monitor-Funktion: Wird nach jedem Zeitschritt aufgerufen.
        Speichert die aktuelle Lösung für spätere Auswertung.
        
        Args:
            ts: Time Stepper Objekt
            step: Aktueller Zeitschritt (Nummer)
            time: Aktuelle Zeit
            u: Aktueller Lösungsvektor
        """
        vals = u.getValues([0, 1])         # Aktuelle Populationswerte extrahieren
        times.append(time)                 # Zeit speichern
        sol.append(vals)                   # Lösung speichern

    ts.setMonitor(monitor)                 # Monitor-Funktion registrieren
    ts.solve(u)                           # Zeitintegration durchführen

    # Konvertierung zu NumPy Arrays für einfachere Handhabung
    sol = np.array(sol)                    # Lösungsmatrix: (n_timesteps, 2)
    times = np.array(times)                # Zeitvektor: (n_timesteps,)

    # Visualisierung der Populationsdynamik
    plt.figure(figsize=(8, 5))
    plt.plot(times, sol[:, 0], label='Beute (x)', linewidth=2, color='blue')
    plt.plot(times, sol[:, 1], label='Räuber (y)', linewidth=2, color='red')
    plt.xlabel('Zeit t')
    plt.ylabel('Populationsgröße')
    plt.title(f'Lotka–Volterra Modell (α={α}, β={β}, δ={δ}, γ={γ})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot als PNG-Datei speichern (für headless Umgebungen geeignet)
    plt.savefig('lotka_volterra.png', dpi=150, bbox_inches='tight')
    plt.close()  # Speicher freigeben
    
    print(f"Simulation abgeschlossen. Plot gespeichert als 'lotka_volterra.png'")
    print(f"Simulierte Zeit: {t0} bis {tf} mit Schrittweite {dt}")
    print(f"Anzahl Zeitschritte: {len(times)-1}")

if __name__ == "__main__":
    main()
