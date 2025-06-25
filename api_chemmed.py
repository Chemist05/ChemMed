from rdkit.Chem import Descriptors
from rdkit.Chem import GraphDescriptors, rdmolops, rdMolDescriptors
from rdkit import Chem
import math

from flask import Flask, request, jsonify 

app = Flask(__name__)

@app.route("/prop1")
def prop1():

    data = request.get_json()
    smiles = data.get("smiles")

    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        return jsonify({"error" : "SMILES is missing or is wrong!"}), 400

    molw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    numhdon = Descriptors.NumHDonors(mol)
    numhacp = Descriptors.NumHAcceptors(mol)

    return jsonify({
        "smiles" : smiles,
        "molw" : molw,
        "logp" : logp,
        "rotb" : rotb,
        "tpsa" : tpsa,
        "numhdon" : numhdon,
        "numhacp" : numhacp
    })

@app.route("/prop2")
def prop2():

    data = request.get_json()
    smiles = data.get("smiles")

    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        return jsonify({"error" : "SMILES is missing or is wrong!"}), 400

    bal_index = GraphDescriptors.BalabanJ(mol)
    ran_index = GraphDescriptors.Chi0(mol)

    dist = rdmolops.GetDistanceMatrix(mol)
    wie_index = dist.sum() / 2

    degr = [atom.GetDegree() for atom in mol.GetAtoms()]
    zgr_m1 = sum(d**2 for d in degr)
    zgr_m2 = sum(degr[b.GetBeginAtomIdx()] * degr[b.GetEndAtomIdx()] for b in mol.GetBonds())


    def topol_pol():

        tp = 0
        leng = len(degr)
        for i in range(leng):
            for j in range(i + 1, leng):
                d_ij = dist[i][j]
                tp += math.sqrt(degr[i] * degr[j]) / (d_ij**2)

        return tp

    tp = topol_pol()

    surf_area = rdMolDescriptors.CalcLabuteASA(mol)

    return jsonify({
        "smiles" : smiles,
        "wie_index" : wie_index,
        "bal_index" : bal_index,
        "ran_index" : ran_index,
        "zgr_m1" : zgr_m1,
        "zgr_m2" : zgr_m2,
        "tp" : tp,
        "surf_area" : surf_area
    })


if __name__ == "__main__":
    app.run(debug=True)
