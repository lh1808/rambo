econml/validate/utils.py in Funktion `calculate_dr_outcomes' : y_dr_0 = reg_preds[:, 0] + (d0_mask / np.clip(prop_preds[:, 0], .01, np.inf)) * (y - reg_preds[:, 0]) 
Mögliche Support Probleme und Regionen, wo der Treatment Effekt nicht identifiziert sind, werden einfach weggeklippt. Link zur Funktion EconML/econml/validate/utils.py at main · py-why/EconML. 
