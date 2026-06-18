import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RYGate

########################################################################

# Please see:
### https://doi.org/10.1007/978-3-030-25027-0_9

### this code is based off and modified from:
### https://github.com/andre-juan/dicke_states_preparation/
### in particular the dicke_states.py file has been modified to work with later versions of Qiskit.
### which is shared under an Apache License 2.0
### to see the license please follow: https://github.com/andre-juan/dicke_states_preparation/blob/main/LICENSE
### a copy has been included in the directory of this file
### if using this codebase please reference the paper: https://doi.org/10.1007/978-3-030-25027-0_9 
### and the repo: https://github.com/andre-juan/dicke_states_preparation/ 

########################################################################

def gate_i(n: int, draw: bool = False) -> QuantumCircuit:
    """
    Returns the 2-qubit gate (i), introduced in Sec. 2.2
    
    This gate is simply a cR_y between 2 CNOTs, with
    rotation angle given by 2*np.arccos(np.sqrt(1/n))
    
    Args:
        n: int, defining the rotation angle
        draw: bool, determines if the circuit is drawn
    """
    qc_i = QuantumCircuit(2, name=f"$(i)_{n}$")

    qc_i.cx(0, 1)

    theta = 2 * np.arccos(np.sqrt(1/n))
    cry = RYGate(theta).control(1, ctrl_state="1")
    qc_i.append(cry, [1, 0])

    qc_i.cx(0, 1)

    if draw:
        print(qc_i.draw())
        
    return qc_i.to_gate()

def gate_ii_l(l: int, n: int, draw: bool = False) -> QuantumCircuit:
    """
    Returns the 2-qubit gate (ii)_l, introduced in Sec. 2.2
    
    This gate is simply a ccR_y between 2 CNOTs, with
    rotation angle given by 2*np.arccos(np.sqrt(l/n))
    
    Args:
        l: int, defining the rotation angle
        n: int, defining the rotation angle
        draw: bool, determines if the circuit is drawn
    """
    qc_ii = QuantumCircuit(3, name=f"$(ii)_{{{l},{n}}}$")

    qc_ii.cx(0, 2)

    theta = 2 * np.arccos(np.sqrt(l/n))
    ccry = RYGate(theta).control(2, ctrl_state="11")
    qc_ii.append(ccry, [2, 1, 0])

    qc_ii.cx(0, 2)

    if draw:
        print(qc_ii.draw())
        
    return qc_ii.to_gate()

def gate_scs_nk(n: int, k: int, draw: bool = False) -> QuantumCircuit:
    """
    Returns the SCS_{n,k} gate, which is introduced in Definition 3 and
    is built of one 2-qubits (i) gate and k+1 3-qubits gates (ii)_l
    
    Args:
        n: int, used as argument of gate_i() and gate_ii_l()
        k: int, defines the size of the gate, as well as its topology
        draw: bool, determines if the circuit is drawn
    """
    qc_scs = QuantumCircuit(k + 1, name=f"SCS$_{{{n},{k}}}$")

    qc_scs.append(gate_i(n), [k-1, k])

    for l in range(2, k + 1):
        qc_scs.append(gate_ii_l(l, n), [k-l, k-l+1, k])

    if draw:
        print(qc_scs.decompose().draw())

    return qc_scs.to_gate()

def first_block(n: int, k: int, l: int, draw: bool = False) -> QuantumCircuit:
    """
    Returns the first block of unitaries products in Lemma 2.
    
    Args:
        n: int, used to calculate indices and as argument of gate_scs_nk()
        k: int, used to calculate indices and as argument of gate_scs_nk()
        l: int, used to calculate indices and as argument of gate_scs_nk()
        draw: bool, determines if the circuit is drawn
    """
    qr = QuantumRegister(n, "q")
    qc_first_block = QuantumCircuit(qr)

    # indices of qubits to which we append identities.
    n_first = l - k - 1
    n_last = n - l
    
    # qubits indices to which we append the SCS_{n,k} gate.
    idxs_scs = list(range(n))

    if n_first > 0:
        idxs_scs = idxs_scs[n_first:]
        qc_first_block.id(qr[:n_first])  # Changed from i() to id()

    if n_last > 0:
        idxs_scs = idxs_scs[:-n_last]
        qc_first_block.append(gate_scs_nk(l, k), idxs_scs)
        qc_first_block.id(qr[-n_last:])  # Changed from i() to id()
    else:
        qc_first_block.append(gate_scs_nk(l, k), idxs_scs)
        
    str_operator = f"$1^{{\\otimes {n_first}}} \\otimes$ SCS$_{{{l},{k}}} \\otimes 1^{{\\otimes {n_last}}}$"
    
    gate = qc_first_block.to_gate()
    gate.name = str_operator

    if draw:
        print(qc_first_block.decompose().draw())
        
    return gate

def second_block(n: int, k: int, l: int, draw: bool = False) -> QuantumCircuit:
    """
    Returns the second block of unitaries products in Lemma 2.
    
    Args:
        n: int, used to calculate indices and as argument of gate_scs_nk()
        k: int, used to calculate indices and as argument of gate_scs_nk()
        l: int, used to calculate indices and as argument of gate_scs_nk()
        draw: bool, determines if the circuit is drawn
    """
    qr = QuantumRegister(n, "q")
    qc_second_block = QuantumCircuit(qr)

    n_last = n - l
    idxs_scs = list(range(n))

    if n_last > 0:
        idxs_scs = idxs_scs[:-n_last]
        qc_second_block.append(gate_scs_nk(l, l-1), idxs_scs)
        qc_second_block.id(qr[-n_last:])  # Changed from i() to id()
    else:
        qc_second_block.append(gate_scs_nk(l, l-1), idxs_scs)
        
    str_operator = f"SCS$_{{{l},{l-1}}} \\otimes 1^{{\\otimes {n_last}}}$"
    
    gate = qc_second_block.to_gate()
    gate.name = str_operator

    if draw:
        print(qc_second_block.decompose().draw())
        
    return gate

def dicke_state(
    n: int, 
    k: int, 
    draw: bool = False, 
    barrier: bool = False, 
    only_decomposed: bool = False
) -> QuantumCircuit:
    """
    Returns a quantum circuit to prepare a Dicke state with arbitrary n, k

    Based on: https://arxiv.org/pdf/1904.07358
    
    This function essentially calls previous functions, wrapping the products in Lemma 2
    within the appropriate for loops for the construction of U_{n, k}, which is
    applied to the initial state as prescribed.
    
    Args:
        n: int, number of qubits for the Dicke state
        k: int, hamming weight of states to be included in the superposition
        draw: bool, determines if the circuit is drawn
        barrier: bool, determines if barriers are added to the circuit
        only_decomposed: bool, determines if only the 3-fold decomposed circuit is drawn
    """
    qr = QuantumRegister(n, "q")
    qc = QuantumCircuit(qr)

    # Initial state
    qc.x(qr[-k:])

    if barrier:
        qc.barrier()

    # Apply first term in Lemma 2
    for l in range(k+1, n+1)[::-1]:
        qc.append(first_block(n, k, l), range(n))
        if barrier:
            qc.barrier()

    # Apply second term in Lemma 2
    for l in range(2, k+1)[::-1]:
        qc.append(second_block(n, k, l), range(n))
        if barrier:
            qc.barrier()

    # Drawing
    if draw:
        if only_decomposed:
            print(qc.decompose().decompose().decompose().draw())
        else:
            print(qc.draw())
            print()
            print(qc.decompose().draw())
            print()
            print(qc.decompose().decompose().draw())
            print()
            print(qc.decompose().decompose().decompose().draw())
    
    return qc
