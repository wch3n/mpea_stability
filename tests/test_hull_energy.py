import unittest

from run_stability import predict


class HullEnergyTest(unittest.TestCase):
    def test_energies_for_the_reported_solid_solution(self):
        # Reference energies in eV/atom at 0.9 Tm, first for solid solutions
        # alone and then with the competing intermetallics included.
        cases = {
            "AlCo": ("HCP", "unstable", -0.333995321, 0.278350294),
            "CrCu": ("FCC", "unstable", 0.181007262, 0.181007262),
            "Al3Ni": ("HCP", "unstable", -0.116042118, 0.052090142),
            "CoCrFeMnNi": ("FCC", "stable", -0.029529527, -0.029529527),
        }
        results = predict(list(cases), 0.9)
        self.assertEqual(len(results), len(cases))
        for (formula, expected), (result, _, _) in zip(cases.items(), results):
            with self.subTest(formula=formula):
                phase, stability, ss_energy, im_energy = expected
                self.assertEqual(result["phase"], phase)
                self.assertEqual(result["stability"], stability)
                self.assertAlmostEqual(result["e_above"], ss_energy)
                self.assertAlmostEqual(result["e_above_im"], im_energy)
