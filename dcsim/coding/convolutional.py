"""Rate-1/2 Constraint Length K=3 Convolutional Code with Hard-Decision Viterbi Decoder.

Owner: Student 3 (Module C: Channel, Coding and Integration Lab)

Specifications:
- Constraint length K = 3 (Shift register states: 4)
- Rate R = 1/2
- Generator polynomials:
    g1 = 7_8 = 111_2 -> v1 = u_t ^ u_{t-1} ^ u_{t-2}
    g2 = 5_8 = 101_2 -> v2 = u_t ^ u_{t-2}
- Maximum Likelihood (ML) Viterbi decoding with traceback.
"""

from __future__ import annotations
import numpy as np


class ConvolutionalCodec:
    """Rate-1/2 K=3 (7,5)_8 Convolutional Encoder and Viterbi Decoder."""

    def __init__(self) -> None:
        self.K = 3
        self.num_states = 4  # 2^(K-1)
        self.rate = 0.5
        self.flush_bits = self.K - 1  # 2 flush bits

        # Precompute trellis state transitions and branch outputs
        # State representation: integer in [0..3], where state = 2*s1 + s0 = (u_{t-1}, u_{t-2})
        # For each state and input bit (0 or 1):
        #   next_state = 2*u + s1
        #   out1 = u ^ s1 ^ s0
        #   out2 = u ^ s0
        self.next_state = np.zeros((self.num_states, 2), dtype=int)
        self.output_bits = np.zeros((self.num_states, 2, 2), dtype=np.int8)

        for s in range(self.num_states):
            s1 = (s >> 1) & 1
            s0 = s & 1
            for u in (0, 1):
                ns = (u << 1) | s1
                v1 = u ^ s1 ^ s0
                v2 = u ^ s0
                self.next_state[s, u] = ns
                self.output_bits[s, u] = [v1, v2]

        # Precompute predecessor states for each destination state in the trellis
        # For dest state S = (s1', s0'):
        # Input u must be s1'. Predecessor s1 must be s0'. Predecessor s0 can be 0 or 1.
        self.predecessors: list[list[tuple[int, int, np.ndarray]]] = []
        for ns in range(self.num_states):
            s1_prime = (ns >> 1) & 1
            s0_prime = ns & 1
            u = s1_prime
            preds = []
            for s0 in (0, 1):
                ps = (s0_prime << 1) | s0
                v1 = u ^ s0_prime ^ s0
                v2 = u ^ s0
                preds.append((ps, u, np.array([v1, v2], dtype=np.int8)))
            self.predecessors.append(preds)

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """Encode binary sequence with zero-flushing termination.

        Output length = 2 * (len(info_bits) + flush_bits).
        """
        b = np.asarray(info_bits, dtype=np.int8).ravel()
        # Append K-1 = 2 zero flush bits to terminate encoder in state 0
        b_flushed = np.append(b, np.zeros(self.flush_bits, dtype=np.int8))
        total_steps = len(b_flushed)

        coded_stream = np.zeros(2 * total_steps, dtype=np.int8)
        state = 0

        for t, bit in enumerate(b_flushed):
            out = self.output_bits[state, bit]
            coded_stream[2 * t : 2 * t + 2] = out
            state = self.next_state[state, bit]

        return coded_stream

    def decode(self, coded_bits: np.ndarray, num_info_bits: int | None = None) -> np.ndarray:
        """Perform hard-decision Viterbi decoding over trellis.

        Traceback from state 0 gives the ML decoded information sequence.
        """
        c = np.asarray(coded_bits, dtype=np.int8).ravel()
        total_steps = len(c) // 2
        r_pairs = c[: 2 * total_steps].reshape(total_steps, 2)

        INF = 1000000.0
        path_metrics = np.full(self.num_states, INF, dtype=np.float64)
        path_metrics[0] = 0.0  # Encoder always starts in state 0

        # Trellis traceback pointers: survivor_predecessor[t, state]
        survivor_state = np.zeros((total_steps, self.num_states), dtype=int)
        survivor_input = np.zeros((total_steps, self.num_states), dtype=np.int8)

        # Forward Viterbi recursion (Add-Compare-Select)
        for t in range(total_steps):
            r = r_pairs[t]
            new_metrics = np.full(self.num_states, INF, dtype=np.float64)

            for ns in range(self.num_states):
                best_metric = INF
                best_pred = 0
                best_u = 0

                for ps, u, expected_out in self.predecessors[ns]:
                    # Hamming distance branch metric
                    branch_metric = float(np.sum(r != expected_out))
                    candidate_metric = path_metrics[ps] + branch_metric

                    if candidate_metric < best_metric:
                        best_metric = candidate_metric
                        best_pred = ps
                        best_u = u

                new_metrics[ns] = best_metric
                survivor_state[t, ns] = best_pred
                survivor_input[t, ns] = best_u

            path_metrics = new_metrics

        # Traceback: terminated encoder finishes in state 0
        curr_state = 0
        decoded_reversed = []

        for t in range(total_steps - 1, -1, -1):
            u = survivor_input[t, curr_state]
            decoded_reversed.append(u)
            curr_state = survivor_state[t, curr_state]

        decoded_bits = np.array(decoded_reversed[::-1], dtype=np.int8)

        # Strip the flush bits
        info_len = total_steps - self.flush_bits
        if num_info_bits is not None:
            info_len = min(info_len, num_info_bits)

        return decoded_bits[:info_len]
