import random

import numpy as np
from Bio.Align import substitution_matrices
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from ghostfold.core.logging import get_logger

logger = get_logger("mutator")

# Placeholder for 3Di matrix, for a future feature.
three_di_matrix = {
    ('A', 'A'): 5, ('A', 'C'): -1, ('A', 'D'): -2, ('A', 'E'): -1, ('A', 'F'): -2,
    ('C', 'A'): -1, ('C', 'C'): 9, ('C', 'D'): -3, ('C', 'E'): -4, ('C', 'F'): -2,
    ('G', 'G'): 6, ('P', 'P'): 7, ('S', 'S'): 4, ('T', 'T'): 5, ('V', 'V'): 4,
    ('W', 'W'): 11, ('Y', 'Y'): 7, ('X', 'X'): 0,
}


class MSA_Mutator:
    """A module to introduce mutations into an MSA based on substitution matrices."""

    def __init__(self, mutation_rates=None):
        self.MATRIX_NAMES = [
            "BLOSUM62",
            "PAM250",
            "MEGABLAST",
        ]

        self.STANDARD_AA = list("ARNDCQEGHILKMFPSTWYV")

        self.MATRICES = {}
        for name in self.MATRIX_NAMES:
            if name != "3Di":
                try:
                    matrix_data = substitution_matrices.load(name)
                    self.MATRICES[name] = {
                        "matrix": matrix_data,
                        "dict": {
                            tuple(key): float(val)
                            for key, val in matrix_data.items()
                        },
                        "alphabet": list(matrix_data.alphabet),
                    }
                except ValueError as e:
                    logger.warning(f"Could not load matrix '{name}': {e}. Skipping this matrix.")
            else:
                if not three_di_matrix:
                    raise ValueError("3Di matrix data is missing. Please define `three_di_matrix` in mutator.py.")
                alphabet_3di = sorted(list(set([aa for pair in three_di_matrix for aa in pair])))
                self.MATRICES["3Di"] = {
                    "matrix": None,
                    "dict": three_di_matrix,
                    "alphabet": alphabet_3di,
                }

        self.MATRIX_NAMES = list(self.MATRICES.keys())

        self.mutation_rates = {}
        if mutation_rates:
            for matrix_name in self.MATRIX_NAMES:
                rate = mutation_rates.get(matrix_name)
                if rate is not None:
                    self.mutation_rates[matrix_name] = float(rate)
                else:
                    self.mutation_rates[matrix_name] = 5.0

            for key_in_input_rates in mutation_rates.keys():
                if key_in_input_rates not in self.MATRIX_NAMES:
                    logger.warning(f"Mutation rate provided for '{key_in_input_rates}', but this matrix could not be loaded or is not recognized. Ignoring this rate.")
        else:
            self.mutation_rates = {name: 5.0 for name in self.MATRIX_NAMES}

        logger.info(f"Initialized MSA_Mutator with matrices: {self.MATRIX_NAMES} and mutation rates: {self.mutation_rates}")

    def get_substitution_probs(self, residue, matrix_name):
        """Return substitution probabilities for a given residue using specified matrix."""
        matrix_data = self.MATRICES.get(matrix_name)
        if not matrix_data:
            return {}

        scores = {}
        valid_aas = [aa for aa in self.STANDARD_AA if aa in matrix_data["alphabet"]]

        for aa in valid_aas:
            score = matrix_data["dict"].get(
                (residue, aa), matrix_data["dict"].get((aa, residue), None)
            )
            if score is None:
                continue
            scores[aa] = score

        if not scores:
            return {}

        exp_scores = {aa: np.exp(val) for aa, val in scores.items()}
        total = sum(exp_scores.values())
        return {aa: val / total for aa, val in exp_scores.items()} if total > 0 else {}

    def substitute_residue(self, residue, matrix_name):
        if residue not in self.STANDARD_AA:
            return residue
        probs = self.get_substitution_probs(residue, matrix_name)
        if not probs:
            return residue
        return random.choices(list(probs), weights=list(probs.values()), k=1)[0]

    def _apply_mutations(self, sequence, matrix_name, mutation_rate):
        """Applies mutations to a single sequence."""
        seq_list = list(sequence)
        seq_length = len(seq_list)
        num_positions_to_change = int(seq_length * mutation_rate / 100)

        if num_positions_to_change > seq_length:
            num_positions_to_change = seq_length

        if seq_length == 0:
            return ""

        mutable_indices = [i for i, char in enumerate(seq_list) if char in self.STANDARD_AA]

        if not mutable_indices:
            return sequence

        if num_positions_to_change > len(mutable_indices):
            num_positions_to_change = len(mutable_indices)

        positions_to_change_actual = random.sample(mutable_indices, num_positions_to_change)
        positions_to_change_actual.sort()

        for pos in positions_to_change_actual:
            seq_list[pos] = self.substitute_residue(seq_list[pos], matrix_name)
        return "".join(seq_list)

    def evolve_msa(self, input_fasta_path, output_fasta_path, sample_percentage=0.5):
        """Reads an MSA, randomly samples sequences, introduces mutations,
        and writes the evolved MSA to a new FASTA file."""
        logger.info(f"Starting MSA evolution for {input_fasta_path}")
        original_records = list(SeqIO.parse(input_fasta_path, "fasta"))
        if not original_records:
            logger.info(f"No sequences found in {input_fasta_path}. Skipping mutation.")
            with open(output_fasta_path, "w") as handle:
                pass
            return

        num_total_sequences = len(original_records)
        num_sequences_to_sample = max(1, int(num_total_sequences * sample_percentage))

        if num_sequences_to_sample > num_total_sequences:
            num_sequences_to_sample = num_total_sequences

        sampled_records_list = random.sample(original_records, num_sequences_to_sample)
        sampled_record_ids_set = {record.id for record in sampled_records_list}

        evolved_records_output = []

        for record in original_records:
            if record.id in sampled_record_ids_set:
                if not self.MATRIX_NAMES:
                    logger.warning("No valid substitution matrices loaded. Cannot evolve sequences. Adding original record.")
                    evolved_records_output.append(record)
                    continue

                matrix_name = random.choice(self.MATRIX_NAMES)
                mutation_rate = self.mutation_rates.get(matrix_name, 5.0)

                evolved_seq_str = self._apply_mutations(str(record.seq), matrix_name, mutation_rate)

                evolved_record = SeqRecord(
                    Seq(evolved_seq_str),
                    id=f"{record.id}_evolved_{matrix_name.replace(' ', '_')}",
                    description=f"evolved from {record.id} using {matrix_name} with {mutation_rate}% mutations"
                )
                evolved_records_output.append(evolved_record)
            else:
                evolved_records_output.append(record)

        with open(output_fasta_path, "w") as handle:
            SeqIO.write(evolved_records_output, handle, "fasta")
        logger.info(f"Evolved MSA saved to {output_fasta_path}. {num_sequences_to_sample} sequences were sampled and evolved.")
