import numpy as np 
import json
import os 

from symmer.operators.base import PauliwordOp, QuantumState
from symmer.utils import exact_gs_energy

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import QFT, ZGate, CRZGate
from qiskit.circuit.gate import Gate

from qiskit import QuantumRegister #, ClassicalRegister
from typing import Dict, Tuple
from qiskit import transpile
from .dicke import dicke_state
##########################
#### Binary functions ####
##########################

def get_binary(num, width=None):
    nbits = np.ceil(np.log2(num))
    if width is None:
        pass
    elif width<nbits:
        pass
    else:
        nbits = width
    return np.array(list(np.binary_repr(num, width=nbits))).astype(np.uint8)


def bin_to_int(binary_list):
    bitshift = 1<<np.arange(len(binary_list))[::-1]
    return (bitshift*binary_list).sum()


def bin_to_int_twos_compliment(binary_list):
    if binary_list[0]==1:
        # flip bits then add one then taken negative sign
        outint = - (bin_to_int((binary_list+1)%2) + 1)
    else:
        outint = bin_to_int(binary_list)
    return outint


##########################
### circuit  functions ###
##########################


def adder_circuit(nq_sum_register:int, k:int, label:str) -> Gate:
    """
    Build adder circuit based on quantum fourier transform (QFT)

    Args:
        nq_sum_register (int): number of qubits in adder circuit only
        k (int): positive or negative number 
    Returns:
        gate (Gate): gate primative for particular adder circuit

    """
    sub_circuit = QuantumCircuit(nq_sum_register)
    phases = [(k*np.pi/(2**nq_sum_register)) * 2**(i+1) for i in range(nq_sum_register)]
    # https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.QuantumCircuit
    for qi, phi in enumerate(phases):
        sub_circuit.p(phi, qi)
    gate = sub_circuit.to_gate(label=label)
    return gate


def dicke_circuit_chem(nq_qubo:int, na_target:int, nb_target) -> QuantumCircuit:
    """
    Function assumes the encoding of spin-up indices being even and spin-down indices being odd

    Args:
        nq_qubo: number of problem qubits  
        na_target (int): number of alpha electrons
        nb_target (int): number of beta  electrons
    Returns:
        full_dicke (QuantumCircuit): circuit preparing (n/2 choose na) * (n/2 choose nb) superposition state (each of Hamming weight na + nb).

    """
    alph_dicke = dicke_state(nq_qubo//2, na_target).to_gate()
    beta_dicke = dicke_state(nq_qubo//2, nb_target).to_gate()

    full_dicke = QuantumCircuit(nq_qubo)
    full_dicke.append(alph_dicke, [2*qi for qi in range(nq_qubo//2)])
    full_dicke.append(beta_dicke, [2*qi+1 for qi in range(nq_qubo//2)])
    full_dicke = transpile(full_dicke, basis_gates=['id', 'ry', 'rx', 'cx'], optimization_level=1, seed_transpiler=1)
    return full_dicke


class GAS_HW():
    """
    Grover Adaptive Search (GAS) with Hamming Weights

    Search space is uniform superposition over all computational basis states
    """
    def __init__(self,nq_qubo:int, 
                  nq_fx:int,
                    nq_na:int, 
                    nq_nb:int,
                    singles_dict:Dict[int, Tuple],
                    doubles_dict:Dict[Tuple[int,int], Tuple],
                    na_target:int, 
                    nb_target:int) -> None:
        """
        Args:
            nq_qubo (int): number of qubits QUBO is defined on
            nq_fx (int):   number of qubits to store QUBO function on (np.ciel(np.log2(max-qubo-value)))
            nq_na (int):   number of qubits to store even occupancy
            nq_nb (int):   number of qubits to store odd  occupancy
            singles_dict (dict): bit-index, coefficient for singles term in QUBO
            doubles_dict (dict): bit-index1 and bit-index2 for doubles term in QUBO
            na_target (int): target even/alpha occupancy
            nb_target (int): target  odd/beta  occupancy
            
        
        """
        
        assert np.ceil(np.log2(na_target))<=nq_qubo//2 , "target na value not valid for number of qubits in QUBO problem"
        assert np.ceil(np.log2(nb_target))<=nq_qubo//2 , "target nb value not valid for number of qubits in QUBO problem"

        qr_fx = QuantumRegister(nq_fx, 'f(x)')
        qr_sys = QuantumRegister(nq_qubo, 'sys')
        qr_fx_na  = QuantumRegister(  nq_na, 'na_count')
        qr_fx_nb  = QuantumRegister(  nq_nb, 'nb_count')
        qr_na_flag = QuantumRegister(  1, 'na_flag')
        qr_nb_flag = QuantumRegister(  1, 'nb_flag')
        qr_n_flag = QuantumRegister(  1, 'n_ab_flag')

        fx_qbits = [q for q in qr_fx]

        ## note no initial grover state in qr_sys register (as can be changed)
        ## also qc starting value in qr_fx register is zero here (should be changed during algs progress!)

        

        ##### first build cost function (qr_sys QUBO result put onto qr_fx ancilla register!)
        ##### QUBO start
        qc_QUBO = QuantumCircuit(qr_sys, qr_fx, qr_fx_na, qr_fx_nb, qr_na_flag,qr_nb_flag, qr_n_flag)
        
        # for qidx in range(nq_fx): ## this is done in initial Y reg now!
        #     qc_QUBO.h(qr_fx[qidx])

        for q_idx, coeff in singles_dict.items():
            g = adder_circuit(nq_fx,coeff, f"({coeff})x{q_idx}").control(1)
            qc_QUBO.append(g, [qr_sys[q_idx], *fx_qbits])

        for (q_idx1, q_idx2), coeff in doubles_dict.items():
            g = adder_circuit(nq_fx,coeff, f"({coeff})x{q_idx1}x{q_idx2}").control(2)
            qc_QUBO.append(g, [qr_sys[q_idx1],qr_sys[q_idx2], *fx_qbits])


        QFT_dag = QFT(num_qubits=nq_fx, inverse=True).to_gate()
        qc_QUBO.append(QFT_dag, fx_qbits)
        ### QUBO end


        #### for symmetry add two ancilla registers to count the hamming weight of the resulting solutions
        ### "c" for counting number of bits START
        qc_elec_count = QuantumCircuit(qr_sys, qr_fx, qr_fx_na, qr_fx_nb, qr_na_flag,qr_nb_flag, qr_n_flag)
        for qidx in range(qr_fx_na.size):
            qc_elec_count.h(qr_fx_na[qidx])
        for qidx in range(qr_fx_nb.size):
            qc_elec_count.h(qr_fx_nb[qidx])

        for q_idx in range(nq_qubo):
            if q_idx%2:
                g = adder_circuit(nq_na, 1, f"c{q_idx}").control(1)
                qc_elec_count.append(g, [qr_sys[q_idx], *[q for q in qr_fx_nb]]) # qr_fx_nb rather than qr_fx_na as qubit order is flipped in qiksit
            else:
                g = adder_circuit(nq_nb, 1, f"c{q_idx}").control(1)
                qc_elec_count.append(g, [qr_sys[q_idx], *[q for q in qr_fx_na]]) # qr_fx_na rather than qr_fx_nb as qubit order is flipped in qiksit


        QFT_dag1 = QFT(num_qubits=qr_fx_na.size, inverse=True).to_gate()
        qc_elec_count.append(QFT_dag1, qr_fx_na)
        QFT_dag2 = QFT(num_qubits=qr_fx_nb.size, inverse=True).to_gate()
        qc_elec_count.append(QFT_dag2, qr_fx_nb)

        ### "c" for counting number of bits END
                
        # number check START!
        bin_target_a = get_binary(na_target, width=qr_fx_na.size)[::-1]
        bin_target_b = get_binary(nb_target, width=qr_fx_nb.size)[::-1]

        for qi, b in enumerate(bin_target_a):
            if b==0:
                qc_elec_count.x(qr_fx_na[qi])
        qc_elec_count.mcx([q for q in qr_fx_na], qr_na_flag[0])
        for qi, b in enumerate(bin_target_a):
            if b==0:
                qc_elec_count.x(qr_fx_na[qi])

        for qi, b in enumerate(bin_target_b):
            if b==0:
                qc_elec_count.x(qr_fx_nb[qi])
        qc_elec_count.mcx([q for q in qr_fx_nb], qr_nb_flag[0])
        for qi, b in enumerate(bin_target_b):
            if b==0:
                qc_elec_count.x(qr_fx_nb[qi])

        qc_elec_count.ccx(qr_na_flag[0], qr_nb_flag[0], qr_n_flag[0])
        # number check END!

        ## copy reverse circuit here!
        qc_before_marking_gate_ADJOINT = (qc_QUBO.compose(qc_elec_count)).inverse()


        ### mark states START!
        qc_mark = QuantumCircuit(qr_sys, qr_fx, qr_fx_na, qr_fx_nb, qr_na_flag,qr_nb_flag, qr_n_flag)
        for qi in range(nq_qubo-1):
            qc_mark.cx(qr_sys[qi], qr_sys[qi+1])
        ccrz_gate = CRZGate(-2*np.pi).control(1)
        qc_mark.append(ccrz_gate, [qr_n_flag[0], fx_qbits[-1], qr_sys[-1]])
        # qc_diffusion.crz(-2*np.pi, fx_qbits[-1], qr_sys[-1])
        # qr_n_flag
        for qi in np.arange(nq_qubo-1-1, -1, -1):
            qc_mark.cx(qr_sys[qi], qr_sys[qi+1])
        ## mark states END!

        # #### add dagger circuit before marking!
        # qc = qc.compose(qc_before_marking_gate_ADJOINT)

        ## finally reflect
        qc_reflect = QuantumCircuit(qr_sys, qr_fx, qr_fx_na, qr_fx_nb, qr_na_flag,qr_nb_flag, qr_n_flag)
        for qi in range(nq_qubo):
            qc_reflect.h(qr_sys[qi])
            qc_reflect.x(qr_sys[qi])
        mcz_gate = ZGate().control(nq_qubo-1)
        qc_reflect.append(mcz_gate, [q for q in qr_sys])
        for qi in range(nq_qubo):
            qc_reflect.x(qr_sys[qi])
            qc_reflect.h(qr_sys[qi])
        qc_reflect.barrier()

        ## circuit pieces
        self.qc_QUBO = qc_QUBO
        self.qc_elec_count = qc_elec_count
        self.qc_mark = qc_mark
        self.qc_reflect = qc_reflect

        ## store different registers:        
        self.qr_fx      = qr_fx       
        self.qr_sys     = qr_sys    
        self.qr_fx_na   = qr_fx_na  
        self.qr_fx_nb   = qr_fx_nb  
        self.qr_na_flag = qr_na_flag
        self.qr_nb_flag = qr_nb_flag
        self.qr_n_flag  = qr_n_flag 

        self.nq_qubo = nq_qubo
        self.nq_fx   = nq_fx

    def build_circuit(self, fx_initial:int, grover_repeats:int) -> QuantumCircuit:
        """
        """
        assert grover_repeats>0, "need to have at least 1 Grover repeat!"

        ## grover initial state |s>
        s_state_qc = QuantumCircuit(self.qr_sys, 
                                    self.qr_fx,
                                    self.qr_fx_na, 
                                    self.qr_fx_nb, 
                                    self.qr_na_flag,
                                    self.qr_nb_flag, 
                                    self.qr_n_flag)
        for qidx in range(self.nq_qubo):
            s_state_qc.h(self.qr_sys[qidx])
        

        ## inital qubo value (best current result here!)
        y_subtractor = fx_initial_qc = QuantumCircuit(self.qr_sys, 
                        self.qr_fx,
                        self.qr_fx_na, 
                        self.qr_fx_nb, 
                        self.qr_na_flag,
                        self.qr_nb_flag, 
                        self.qr_n_flag)
        if fx_initial!=0:
            for qidx in range(self.nq_fx):
                fx_initial_qc.h(self.qr_fx[qidx])
            y_subtractor.append(adder_circuit(self.nq_fx, fx_initial, f"initial: {fx_initial}"),
                              [q for q in self.qr_fx])
            # for qidx in range(self.nq_fx):
            #     fx_initial_qc.h(self.qr_fx[qidx])

        qc_full = s_state_qc
        first = y_subtractor.compose(self.qc_QUBO).compose(self.qc_elec_count)
        for _ in range(grover_repeats):   
            qc_full = qc_full.compose(first)
            qc_full = qc_full.compose(self.qc_mark)
            qc_full = qc_full.compose(first.inverse())
            qc_full = qc_full.compose(self.qc_reflect)

        return qc_full


class GAS_dicke():
    """
    Grover Adaptive Search (GAS) over Dicke states
    """
    def __init__(self,
                    nq_qubo:int, 
                    nq_fx:int,
                    singles_dict:Dict[int, Tuple],
                    doubles_dict:Dict[Tuple[int,int], Tuple],
                    dicke_circuit:QuantumCircuit) -> None:
        """
        Args:
            nq_qubo (int): number of qubits QUBO is defined on
            nq_fx (int):   number of qubits to store QUBO function on (np.ciel(np.log2(max-qubo-value)))
            singles_dict (dict): bit-index, coefficient for singles term in QUBO
            doubles_dict (dict): bit-index1 and bit-index2 for doubles term in QUBO
            dicke_circuit (QuantumCircuit): Quantum circuit preparing Dicke State
            
        """
        
        assert dicke_circuit.num_qubits == nq_qubo, "dicke circuit not defined on correct number of qubits"

        qr_sys = QuantumRegister(nq_qubo, 'sys')
        qr_fx = QuantumRegister(nq_fx, 'f(x)')
        

        fx_qbits = [q for q in qr_fx]
        sy_qbits = [q for q in qr_sys]

        ## note no initial grover state in qr_sys register (as can be changed)
        ## also qc starting value in qr_fx register is zero here (should be changed during algs progress!)

        self.qc_QUBO = QuantumCircuit(qr_sys, qr_fx)

        ##### first build cost function (qr_sys QUBO result put onto qr_fx ancilla register!)
        ##### QUBO start
        # for qidx in range(nq_fx):
        #     self.qc_QUBO.h(qr_fx[qidx])

        for q_idx, coeff in singles_dict.items():
            g = adder_circuit(nq_fx,coeff, f"({coeff})x{q_idx}").control(1)
            self.qc_QUBO.append(g, [qr_sys[q_idx], *fx_qbits])

        for (q_idx1, q_idx2), coeff in doubles_dict.items():
            g = adder_circuit(nq_fx,coeff, f"({coeff})x{q_idx1}x{q_idx2}").control(2)
            self.qc_QUBO.append(g, [qr_sys[q_idx1],qr_sys[q_idx2], *fx_qbits])


        QFT_dag = QFT(num_qubits=nq_fx, inverse=True).to_gate()
        self.qc_QUBO.append(QFT_dag, fx_qbits)
        ### QUBO end

        ### mark states START!
        self.qc_mark = QuantumCircuit(qr_sys, qr_fx)

        for qi in range(nq_qubo-1):
            self.qc_mark.cx(qr_sys[qi], qr_sys[qi+1])
        self.qc_mark.crz(-2*np.pi, fx_qbits[-1], qr_sys[-1])
        for qi in np.arange(nq_qubo-1-1, -1, -1):
            self.qc_mark.cx(qr_sys[qi], qr_sys[qi+1])
        ## mark states END!


        ## finally reflect
        # i.e. do: U(2∣0⟩⟨0∣−I)U†∣ψ⟩ where U|0> = |d>... Here ∣ψ⟩ is an arbitrary state in the system register 
        # therefore apply Udagger, then reflect around |0>, then apply U
        self.qc_reflect = QuantumCircuit(qr_sys, qr_fx)
        dicke_dag_gate = dicke_circuit.inverse().to_gate(label="dicke-dag")
        self.qc_reflect.append(dicke_dag_gate, [q for q in sy_qbits])

        for qi in range(nq_qubo):
            self.qc_reflect.x(qr_sys[qi])
        mcz_gate = ZGate().control(nq_qubo-1)
        self.qc_reflect.append(mcz_gate, [q for q in qr_sys])
        for qi in range(nq_qubo):
            self.qc_reflect.x(qr_sys[qi])
        
        dicke_gate = dicke_circuit.to_gate(label="dicke")
        self.qc_reflect.append(dicke_gate, [q for q in sy_qbits])

        ## store different registers:        
        self.qr_fx      = qr_fx       
        self.qr_sys     = qr_sys    

        self.nq_qubo = nq_qubo
        self.nq_fx   = nq_fx

        self.dicke_circuit = dicke_circuit


    def build_circuit(self, fx_initial:int, grover_repeats:int) -> QuantumCircuit:
        """
        """
        assert grover_repeats>0, "need to have at least 1 Grover repeat!"

        ## grover initial state |d> = |dicke>
        s_state_qc = QuantumCircuit(self.qr_sys, 
                                    self.qr_fx)
        
        dicke_gate = self.dicke_circuit.to_gate(label="dicke")
        s_state_qc.append(dicke_gate, [q for q in self.qr_sys])
        

        ## inital qubo value (best current result here!)
        y_subtractor = fx_initial_qc = QuantumCircuit(self.qr_sys, 
                        self.qr_fx)
        
        if fx_initial!=0:
            for qidx in range(self.nq_fx):
                fx_initial_qc.h(self.qr_fx[qidx])
            y_subtractor.append(adder_circuit(self.nq_fx, fx_initial, f"initial: {fx_initial}"),
                              [q for q in self.qr_fx])
            # for qidx in range(self.nq_fx):
            #     fx_initial_qc.h(self.qr_fx[qidx])

        qc_full = s_state_qc
        first = y_subtractor.compose(self.qc_QUBO)
        for _ in range(grover_repeats):
            qc_full = qc_full.compose(first)
            qc_full = qc_full.compose(self.qc_mark)
            qc_full = qc_full.compose(first.inverse())
            qc_full = qc_full.compose(self.qc_reflect)

        return qc_full

