import numpy as np
import sympy
import re
import json
from astropy.utils.misc import JsonCustomEncoder
import qsymm
from icecream import ic

def bloch_model_to_json(model, name):
    name = name + '.json'
    model_dict = {str(key) : model[key].tolist() for key in model.keys()}
    with open(name, 'w') as f:
        json.dump(model_dict, f, cls = JsonCustomEncoder)

def bloch_model_from_json(name):
    with open("json_models/" + name + ".json", "r") as f:
        model = json.load(f, cls=JsonCustomDecoderExp)
    return qsymm.Model(model, momenta=["k_x", "k_y"])


class JsonCustomDecoderExp(json.JSONDecoder):
    """
    A decoder function tailored specifically to our purposes: take a dictionary of strings, output a dictionary
    of symbolic keys (Mul types etc.) and arrays.
    """

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        new_dict = dict()
        for key in obj.keys():
            # back-convert the key from string to symbols
            symbolic = []
            # find exponents, works for arbitrary number of exponents
            exp_full = re.findall('\*?e\*\*\(-?I\*k_\w\)\*?', key) # gives list with exponential terms in key
            if exp_full:
                coef = key
                for exp in exp_full:
                    coef = coef.replace(exp, '') # constant coefficient 
                    
                exp_power_pl = re.findall('\*?e\*\*\(I\*(k_\w)\)\*?', key) # gives list of positive exponents
                exp_power_mn = re.findall('\*?e\*\*\(-I\*(k_\w)\)\*?', key) # gives list of negative exponents
                
                if len(exp_power_pl)>=1:
                    sym_exp_pls = []
                    for exp in exp_power_pl:
                        sym_exp_pls.append(sympy.Pow(sympy.Symbol('e'), sympy.I*sympy.Symbol(exp)))
                    if len(exp_power_mn)==0: # no negative powers
                        symbolic_key = sympy.Mul(*sym_exp_pls, sympy.Symbol(coef))
                    
                if len(exp_power_mn)>=1:
                    sym_exp_mns = []
                    for exp in exp_power_mn:
                        sym_exp_mns.append(sympy.Pow(sympy.Symbol('e'), -sympy.I*sympy.Symbol(exp)))
                    if len(exp_power_pl)==0: # no positive powers
                        symbolic_key = sympy.Mul(*sym_exp_mns, sympy.Symbol(coef))
                    else: # both positive and negative powers
                        symbolic_key = sympy.Mul(*sym_exp_mns, *sym_exp_pls, sympy.Symbol(coef))
            else:
                symbolic_key = sympy.Symbol(key)
                
            assert str(key)==str(symbolic_key) # make sure converter works
            array = []
            for line in obj[key]:
                lineski = []
                for entry in line:
                    lineski.append(entry[0] + 1j * entry[1])
                array.append(lineski)
            new_dict[symbolic_key] = np.array(array)
        return new_dict



