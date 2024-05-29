"""
A MIDI-sequencer based on Conway's Game of Life.
"""

import os
import random
import time
from typing import Iterator

import mido
import numpy as np
from mido import Message
from numpy.typing import NDArray


State = NDArray[np.bool_]

# grid size
WIDTH = 20 * 3
HEIGHT = 20 * 3

# probability skew towards diatonic notes
DIATONICITY = 4.0

# how unlikely are for notes to be played
SPARSENESS = 2.0

BPM = 60

initial_state = np.random.randint(0, 2, size=(HEIGHT, WIDTH)).astype(np.bool_)


def print_state(state: State) -> None:
    for i in range(HEIGHT):
        for j in range(WIDTH):
            print("*" if state[i, j] else " ", end="")
        print()


def update_state(state: State) -> State:
    combined_state = -state.astype(np.int32)

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            combined_state += np.roll(
                np.roll(state.astype(np.int32), dx, axis=0),
                dy,
                axis=1,
            )

    return (combined_state == 3) | (state & (combined_state == 2))


def find_notes(state: State) -> Iterator[Message]:
    n, m = state.shape[0] // 3, state.shape[1] // 3
    reshaped_state = state.reshape(n, 3, m, 3)
    reduced_state = reshaped_state.sum(axis=(1, 3)).reshape(-1)

    scale = list(range(48, 72))
    notes = []
    i = 0
    for _ in range(len(scale)):
        notes.append(scale[i % len(scale)])
        i += 7
    weights = [1 / DIATONICITY**i for i in range(len(notes))]

    for i, group in enumerate(reduced_state):
        random.seed(i)

        if random.random() > 1 / SPARSENESS**2:
            continue

        velocity = int((group / 10) * 127)
        message = "note_on" if velocity > 0 else "note_off"
        note = random.choices(notes, weights)[0]

        yield Message(message, channel=0, note=note, velocity=velocity)


state = initial_state

port_name = mido.get_output_names()[0]

with mido.open_output(port_name) as port:
    while True:
        os.system("clear")
        print_state(state)

        for message in find_notes(state):
            port.send(message)

        time.sleep(60.0 / BPM)
        state = update_state(state)
