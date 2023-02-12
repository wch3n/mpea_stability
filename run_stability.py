#!/usr/bin/env python3.11

from pymatgen.core import Element, Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.analysis.cost import CostAnalyzer, CostDBCSV
from math import log, sqrt
import json
import itertools
import numpy as np
import pandas as pd
import warnings
from timeit import default_timer
from multiprocessing import Pool
from functools import partial
from os.path import join

K=8.61733262E-5
EV2KJMOL = 96.48534
ORDER_IM = 3

def pretty(struct_str):
    structs = struct_str.split("_")
    if structs[1] == 'SS':
        comp = Composition(structs[0])
        c = list(comp.to_reduced_dict.values())
        if c.count(c[0]) == len(c):
            formula = ''.join(structs[0].split("1"))
        else:
            formula = structs[0]
        return formula+'_SS'+'('+structs[-1]+')'
    elif structs[-1] == 'none': 
        return structs[0]
    elif len(structs) == 3:
        return structs[0]+'('+structs[1]+'_'+structs[2]+')'
    else:
        return structs[0]+'('+structs[1]+')'

def min_eform_im_pair(comp, im_eform):
    # equiatomic by definition!
    elems = comp.elements
    norm_dict = comp.get_el_amt_dict()
    eform = []
    for pair in itertools.combinations(elems, 2):
        ei, ej = Element(pair[0]).symbol, Element(pair[1]).symbol
        ci, cj = norm_dict[ei], norm_dict[ej]
        formula = '-'.join(sorted([ei,ej]))
        eform.append(im_eform[formula])
    return min(eform)

def calc_eform_im_pair(comp, im_eform):
    elems = comp.elements
    norm_dict = comp.get_el_amt_dict()
    eform = 0
    for pair in itertools.combinations(elems, 2):
        ei, ej = Element(pair[0]).symbol, Element(pair[1]).symbol
        ci, cj = norm_dict[ei], norm_dict[ej]
        formula = '-'.join(sorted([ei,ej]))
        eform += im_eform[formula]*ci*cj
    return eform*4

def calc_deltaR(comp_frac):
    r = np.array([Element(i).atomic_radius for i in comp_frac.elements])
    c = np.array([i for i in comp_frac.to_reduced_dict.values()])
    r_mean = np.sum(r*c)
    # deltaR
    delta = np.sum(c * (1.0 - r/r_mean)**2)
    return sqrt(delta) if delta > 1E-3 else 1E-3

def calc_gamR(comp_frac):
    # equiatomic only
    r = np.array([Element(i).atomic_radius for i in comp_frac.elements])
    r_mean = r.mean()
    gam_s = ((r.min()+r_mean)**2 - r_mean**2)/(r.min()+r_mean)**2
    gam_s = 1.0 - sqrt(gam_s)
    gam_l = ((r.max()+r_mean)**2 - r_mean**2)/(r.max()+r_mean)**2
    gam_l = 1.0 - sqrt(gam_l)
    return gam_s/gam_l 

def model_others(comp_frac, hmix_m, tm, t, im_eform):
    deltaR = calc_deltaR(comp_frac)
    gamR = calc_gamR(comp_frac)
    c = np.array([i for i in comp_frac.to_reduced_dict.values()])
    h = 0
    elems = [i.symbol for i in comp_frac.elements]
    for i in elems:
        if not i in hmix_m.keys():
            return 'none', 'none'
    for i in range(len(elems)):
        for j in range(i+1, len(elems)):
            h += hmix_m.at[elems[i],elems[j]]*c[i]*c[j]
    h *= 4.0 # in kJ/mol
    if abs(h) < 1E-6:
        h = 1E-6
    smix = -np.sum(c*np.log(c))
    smix *= EV2KJMOL*K #  kJ/mol
    eform_im = calc_eform_im_pair(comp_frac, im_eform)*EV2KJMOL # kJ/mol
    eform_im_min = min_eform_im_pair(comp_frac, im_eform)*EV2KJMOL
    kappa = t * smix*(1.0-0.6)/abs(h) + 1.0
    tann_s = -t * smix
    m1_stability = 'stable' if (smix*tm/abs(h) >= 1.1) and (deltaR  <= 0.066) else 'unstable'
    m2_stability = 'stable' if (-11.6 < h < 3.2)  and (deltaR  < 0.066) else 'unstable'
    m3_stability = 'stable' if (-11.6 < h < 3.2)  and (gamR  < 1.175) else 'unstable'
    m4_stability = "stable" if smix/(deltaR)**2/10 > 0.96 else "unstable"
    m6_stability = "stable" if (eform_im_min > tann_s) and (eform_im_min < 0.00357*EV2KJMOL) else'unstable' # equiatomic only
    m7_stability = "stable" if eform_im/h < kappa else "unstable"
    return m1_stability, m2_stability, m3_stability, m4_stability, m6_stability, m7_stability

def is_equimolar(comp):
    c = [i for i in comp.to_reduced_dict.values()]
    return c.count(c[0]) == len(c)

def is_ss(pd_entry):
    name = pd_entry.name
    return name.split('_')[-2] == 'SS'
    
def sane(formula):
    try:
        comp = Composition(formula)
    except:
        return False

    allowed = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Hf', 'Mn', 'Mo', 'Nb', 'Ni', 'Ta', 'Ti', 'W', \
               'Zr', 'V', 'Mg', 'Re', 'Os', 'Rh', 'Ir', 'Pd', 'Pt', 'Ag', 'Au', 'Zn', 'Cd', \
               'Hg', 'Si', 'Ge', 'Ga', 'In', 'Sn', 'Sb', 'As', 'Te', 'Pb', 'Bi', 'Y', 'Sc', 'Ru']
    return all([i.symbol in allowed for i in comp.elements])

def predict(formulas, t_fac, temperature=-1, expt_phase=None, file_out=None, nproc=1):
    omegas, im, cost, hmix_m, im_eform = init_params()
    if not type(formulas) is list:
        formulas = [formulas]
    if expt_phase == None:
        expt_phase = ['none']*len(formulas)
    func = partial(model,t_fac,temperature,omegas,im,cost,hmix_m,im_eform,file_out)
    with Pool(nproc) as pool:
        res = pool.starmap(func, zip(formulas, expt_phase))
    return res

def model(t_fac, temperature, omegas, im, cost, hmix_m, im_eform, file_out, formula, expt_phase):
    time_0 = default_timer()
    comp_raw = Composition(formula)
    comp = comp_raw.fractional_composition
    tm = np.sum([Element(el).melting_point*comp.get_atomic_fraction(el) for el in comp.elements])
    t = temperature if temperature >= 0 else t_fac*tm
    chemsys_list=[]
    ncomp = len(comp)
    norm_dict = comp.get_el_amt_dict()
    formula_norm = ''
    for i in sorted(norm_dict.keys()):
        formula_norm += '{0}{1:.2f} '.format(i, norm_dict[i])

    # check if we have elements beyond the omegas table
    for el in comp.elements:
        if not el.symbol in omegas['elements']['BCC'].keys():
            return

    for i in range(ncomp):
        for combi in itertools.combinations(comp.elements, i + 1):
            chemsys = "-".join(sorted([x.symbol for x in combi]))
            chemsys_list.append(chemsys)
    entries=[]
    for j in chemsys_list:
        entries.extend(compute_ss_equimolar(omegas,j,t))

    # for non-equimolar alloy
    equimolar = is_equimolar(comp)
    if not equimolar:
        entries_target, conf_entropy = compute_ss(omegas, comp, t)
        entries.extend(entries_target)
    else:
        conf_entropy = -K*t*log(ncomp)
    
    # convex hull analysis
    pd_ss=PhaseDiagram(entries)
    stability="unstable"
    for e in pd_ss.stable_entries:
        if e.composition.fractional_composition == comp:
            e_above=pd_ss.get_equilibrium_reaction_energy(e)
            stability="stable"
    for e in pd_ss.all_entries:
        if e.composition.fractional_composition == comp:
            if e.name == "SS_BCC":
                bcc_energy = e.energy_per_atom
            if e.name == "SS_FCC":
                fcc_energy = e.energy_per_atom
            if e.name == "SS_HCP":
                hcp_energy = e.energy_per_atom
            if stability == "unstable":
                e_above=pd_ss.get_e_above_hull(e)

    # now include the IM up to ternary
    for j in chemsys_list:
        if (len(j.split("-"))<ORDER_IM+1) & (j in im.keys()):
            for r in im[j]:
                im_energy = r['total_energy']
                im_name = Composition(r['unit_cell_formula']).reduced_formula+"_"+r["type_im"]
                entries.append(PDEntry(r['unit_cell_formula'], im_energy, name=im_name))
    pd_im=PhaseDiagram(entries)
    stability_im="unstable"
    for e in pd_im.stable_entries:
        if e.composition.fractional_composition == comp and is_ss(e):
            e_above_im=pd_im.get_equilibrium_reaction_energy(e)
            stability_im="stable"
    for e in pd_im.all_entries:
        if (e.composition.fractional_composition == comp) and is_ss(e) and (stability_im == "unstable"): 
            e_above_im=pd_im.get_e_above_hull(e)

    res_m1, res_m2, res_m3, res_m4, res_m6, res_m7 = model_others(comp, hmix_m, tm, t, im_eform)

    # dump results
    decomp=pd_im.get_decomposition(comp)
    struct = ['BCC','FCC','HCP'][np.argmin([bcc_energy,fcc_energy,hcp_energy])]
    _system = comp_raw.reduced_formula
    _cost = cost.get_cost_per_mol(comp)
    _delta_bcc = bcc_energy-fcc_energy
    _delta_hcp = hcp_energy-fcc_energy
    _decomp = str([x.name for x in decomp]).replace(' ','')
    _decomp = _decomp.strip("[]").replace("'","").split(",") 
    _decomp_pretty = [pretty(i) for i in _decomp]
    _decomp_string = '+'.join(_decomp_pretty)
    hmix = enthalpy_mixing(omegas, comp, struct)
    out = {'system':_system, 'formula_norm': formula_norm, 'e_above':e_above, 'e_above_im':e_above_im, 'hmix': hmix, 'ts_conf': conf_entropy, 'stability':stability_im, \
          'phase': struct, 'cost':_cost, 'delta_bcc':_delta_bcc, 'delta_hcp':_delta_hcp, 'decomp':_decomp_pretty, 'tm':tm, 't':t, 'expt_phase':expt_phase, 'm1':res_m1, 'm2':res_m2, "m3": res_m3, "m4": res_m4, "m6": res_m6, "m7": res_m7}
    msg = "%s %s %6.3f %6.3f %6.3f %s %s %6.2f %6.3f %6.3f %s %6.0f %6.0f %s %s %s %s %s %s %s" \
          %(formula.replace(' ',''), _system, e_above, e_above_im, hmix, stability_im, struct, _cost, _delta_bcc, _delta_hcp, _decomp_string, tm, t, res_m1, res_m2, res_m3, res_m4, res_m6, res_m7, expt_phase)
    print(msg, file=open(file_out, "a"), flush=True) if not file_out == None else print(json.dumps(out))
    return out, default_timer() - time_0, len(entries)

def compute_ss(omegas, comp, t):
    entries = []
    bcc = 0.0; fcc = 0.0; hcp = 0.0
    for i in itertools.combinations(comp.elements, 2):
        chemsys = '-'.join(sorted([el.symbol for el in i]))
        c = [comp.get_atomic_fraction(el) for el in i]
        cicj = np.prod(c)
        bcc += omegas['omegas']['BCC'][chemsys]*cicj
        fcc += omegas['omegas']['FCC'][chemsys]*cicj
        hcp += omegas['omegas']['HCP'][chemsys]*cicj
    element_ref_fcc = 0.0; element_ref_bcc = 0.0; element_ref_hcp = 0.0
    conf_entropy = 0.0
    for el in comp.elements:
        chemsys = el.symbol
        ci = comp.get_atomic_fraction(el)
        element_ref_fcc += ci*(omegas['elements']['FCC'][chemsys])
        element_ref_bcc += ci*(omegas['elements']['BCC'][chemsys])
        element_ref_hcp += ci*(omegas['elements']['HCP'][chemsys])
        conf_entropy += -ci*log(ci)
    conf_entropy *= -K*t
    entries.append(PDEntry(comp, (bcc+element_ref_bcc+conf_entropy),name="SS_BCC"))
    entries.append(PDEntry(comp, (fcc+element_ref_fcc+conf_entropy),name="SS_FCC"))
    entries.append(PDEntry(comp, (hcp+element_ref_hcp+conf_entropy),name="SS_HCP"))
    return entries, conf_entropy

def compute_ss_equimolar(omegas, chemsys, t):
    e=chemsys.split("-")
    entries=[]
    n=len(e)
    if n==1:
        entries.append(PDEntry(e[0]+"1",omegas['elements']['FCC'][e[0]],name=e[0]+"_FCC"))
        entries.append(PDEntry(e[0]+"1",omegas['elements']['BCC'][e[0]],name=e[0]+"_BCC"))
        entries.append(PDEntry(e[0]+"1",omegas['elements']['HCP'][e[0]],name=e[0]+"_HCP"))
    else:
        bcc=0.0; fcc=0.0; hcp=0.0
        for i in itertools.combinations(e, 2):
            bcc+=omegas['omegas']['BCC']['-'.join(i)]*(1.0/n)**2
            fcc+=omegas['omegas']['FCC']['-'.join(i)]*(1.0/n)**2
            hcp+=omegas['omegas']['HCP']['-'.join(i)]*(1.0/n)**2
        element_ref_fcc=0.0; element_ref_bcc=0.0; element_ref_hcp=0.0
        for i in e:
            element_ref_fcc+=(1.0/n)*(omegas['elements']['FCC'][i])
            element_ref_bcc+=(1.0/n)*(omegas['elements']['BCC'][i])
            element_ref_hcp+=(1.0/n)*(omegas['elements']['HCP'][i])
        ideal_entropy = -K*t*log(n)
        entries.append(PDEntry('1'.join(e)+"1",n*(bcc+element_ref_bcc+ideal_entropy),name="SS_BCC"))
        entries.append(PDEntry('1'.join(e)+"1",n*(fcc+element_ref_fcc+ideal_entropy),name="SS_FCC"))
        entries.append(PDEntry('1'.join(e)+"1",n*(hcp+element_ref_hcp+ideal_entropy),name="SS_HCP"))
    return entries

def enthalpy_mixing(omegas, comp, struct):
    hmix = 0
    for i in itertools.combinations(comp.elements, 2):
        chemsys = '-'.join(sorted([el.symbol for el in i]))
        c = [comp.get_atomic_fraction(el) for el in i]
        cicj = np.prod(c)
        hmix += cicj*omegas['omegas'][struct][chemsys]
    return hmix

def init_params(workdir='./model_params'):
    with open(join(workdir,'im_eform.json')) as f:
        im_eform = json.load(f)
    with open(join(workdir,'omegas.json')) as f:
        omegas = json.load(f)
    with open(join(workdir,'im_aflow_icsd.json')) as f:
        im_icsd=json.load(f)
    with open(join(workdir,'im_aflow_lib.json')) as f:
        im_lib = json.load(f)
    im = {}
    keys = im_icsd.keys() | im_lib.keys()
    for k in keys:
        if im_icsd.get(k,{}) and im_lib.get(k,{}):
            im[k] = im_icsd.get(k, {}) +  im_lib.get(k,{})
        elif im_icsd.get(k,{}):
            im[k] = im_icsd.get(k, {})
        else:
            im[k] = im_lib.get(k, {})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cost=CostAnalyzer(CostDBCSV('costdb_elements.csv'))
    hmix_m = pd.read_csv(join(workdir,'./expt/takeuchi.csv'), index_col=0)
    return omegas, im, cost, hmix_m, im_eform

if __name__ == "__main__":
    t_fac = 0.9
    formulas = ['CoCrFeMnNi']
    #file_out = 'mpea_{0}Tm_im{1}.csv'.format(t_fac, ORDER_IM)
    res = predict(formulas, t_fac, temperature=-1, expt_phase=None, file_out=None, nproc=2)
