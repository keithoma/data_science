import numpy as np
from scipy.stats import t, mannwhitneyu

# -------------------------------
# Abgabegruppe: G2
# Personen: Kei Thoma, Marina Steflyuk, Ardit Lushaj
# HU-Accountname: 574613, 572453, 617482
# -------------------------------

def teilaufgabe_a():
    return """
    H_0: Die neue Version ist im Durchschnitt nicht langsamer als die alte Version.\n
    H_A: Die neue Version ist im Durchschnitt langsamer als die alte Version.
    """

def teilaufgabe_b(samples_a, samples_b):
    """
    Führen Sie einen T-Test für die beiden Stichproben mit einem Signifikanzniveau von 5% durch.
    Kann die Nullhypothese zugunsten der Alternativhypothese verworfen werden?
    Geben Sie die Differenz der Mittelwerte, den p-value und das Testergebnis (boolean) zurück.
    """

    # Implementieren Sie hier Ihre Lösung.
    mean_diff = np.mean(samples_a) - np.mean(samples_b)

    size_n = len(samples_a)

    a_std = np.std(samples_a, ddof=1)
    b_std = np.std(samples_b, ddof=1)

    pooled_std = np.sqrt((a_std ** 2 + b_std ** 2) / size_n)
    t_value = mean_diff / pooled_std

    p_value = 1 - t.cdf(t_value, 2 * size_n - 2)
    decision = p_value < 0.05

    """
    Da der errechnete p-Wert mit 0.038 unter dem angegebenen Signifikanzniveau
    von 5% ist, darf die Nullhypothese verworfen werden. Die
    Alternativhypothese ist damit zumutbar und wir inferieren, dass die neue
    Version tatsächlich langsamer ist.
    """

    return mean_diff, p_value, decision

def teilaufgabe_c(samples_a, samples_b):
    """
    Führen Sie den Mann-Whitney U Test für die beiden Stichproben mit einem Signifikanzniveau von 5% durch.
    Kann die Nullhypothese zugunsten der Alternativhypothese verworfen werden?
    Geben Sie den p-value und das Testergebnis (boolean) zurück.
    """
    p_value = mannwhitneyu(samples_a, samples_b, alternative='greater')
    decision = p_value < 0.05
    """
    Argumente: Formulieren Sie hier ihre Antwort. 
    """
    return p_value, decision

if __name__ == "__main__":
    samples_a = np.array([0.24, 0.22, 0.20, 0.25], dtype=np.float64)
    samples_b = np.array([0.2, 0.19, 0.22, 0.18], dtype=np.float64)

    print("Teilaufgabe b)")

    mean_diff, p_value, decision = teilaufgabe_b(samples_a, samples_b)
    print(f"{mean_diff=}")  # ~ 0.03
    print(f"{p_value=}")  # ~ 0.038
    print(f"{decision=}")  # ~ True

    print()
    print("Teilaufgabe c)")
    p_value, decision = teilaufgabe_c(samples_a, samples_b)
    print(f"{p_value=}")  # ~ 0.054
    print(f"{decision=}")  # ~ False
