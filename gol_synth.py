#Code by Vasilii
#It is stupid but it works

import tkinter as tk
import numpy as np
import sounddevice as sd
from scipy.signal import convolve2d
from scipy.io.wavfile import write
import os

# Config for 3 octaves of piano (C3 to B5)
ROWS, COLS = 36, 36
CELL_SIZE = 18
SAMPLE_RATE = 44100
FRAME_DURATION = 0.1
UPDATE_DELAY = int(FRAME_DURATION * 1000)

def piano_frequencies(start_note=48, num_notes=36):
    """Generate frequencies for MIDI note numbers (C3 = 48 to B5 = 83)"""
    return [440.0 * (2 ** ((n - 69) / 12.0)) for n in range(start_note, start_note + num_notes)]

FREQS = piano_frequencies()

class SoundLife:
    def __init__(self, master):
        self.master = master
        self.grid = np.zeros((ROWS, COLS), dtype=int)
        self.canvas = tk.Canvas(master, width=COLS * CELL_SIZE, height=ROWS * CELL_SIZE, bg="black")
        self.canvas.pack()

        self.start_button = tk.Button(master, text="Start", command=self.toggle_running)
        self.start_button.pack(pady=10)

        self.running = False
        self.canvas.bind("<Button-1>", self.toggle_cell)
        self.draw_grid()
        self.samplerate = SAMPLE_RATE
        self.recording_buffer = []

    def toggle_running(self):
        if self.running:
            self.running = False
            self.start_button.config(text="Start")
            self.save_recording()
        else:
            self.running = True
            self.recording_buffer = []
            self.start_button.config(text="Stop")
            self.update_loop()

    def toggle_cell(self, event):
        if not self.running:
            row = event.y // CELL_SIZE
            col = event.x // CELL_SIZE
            if 0 <= row < ROWS and 0 <= col < COLS:
                self.grid[row, col] ^= 1
                self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")
        for r in range(ROWS):
            for c in range(COLS):
                x0, y0 = c * CELL_SIZE, r * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE
                fill = "green" if self.grid[r, c] == 1 else "black"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="gray")

    def update_loop(self):
        if self.running:
            self.grid = self.next_generation(self.grid)
            self.draw_grid()
            audio = self.play_sounds(self.grid)
            self.recording_buffer.append(audio)
            sd.play(audio, SAMPLE_RATE, blocking=True)
            self.master.after(UPDATE_DELAY, self.update_loop)

    def next_generation(self, grid):
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        neighbors = convolve2d(grid, kernel, mode='same', boundary='wrap')
        return ((grid == 1) & ((neighbors == 2) | (neighbors == 3))) | \
               ((grid == 0) & (neighbors == 3))

    def play_sounds(self, grid):
        duration = FRAME_DURATION
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r, c] == 1:
                    duration = duration + FRAME_DURATION / duration
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        output = np.zeros_like(t)

        for r in range(ROWS):
            volume = r / (ROWS - 1)
            for c in range(COLS):
                if grid[r, c] == 1:
                    freq1 = FREQS[c % len(FREQS)]
                    freq2 = FREQS[r % len(FREQS)]
                    wave = np.sin(2 * np.pi * freq1 * t * abs(r - c) * 2 * np.pi / (r + c + 1)) + \
                           np.sin(2 * np.pi * freq2 * t * abs(r - c) * 2 * np.pi / (r + c + 1))
                    output += wave

        if np.max(np.abs(output)) > 0:
            output /= np.max(np.abs(output))
        self.duration = duration
        output = output * self.generate_envelope(t, self.duration / 4, 0, 1, self.duration / 4)
        return output.astype(np.float32)

    def generate_envelope(self, t, attack_time, decay_time, sustain_level, release_time):
        envelope = np.zeros_like(t)

        attack_samples = int(self.duration * attack_time * self.samplerate)
        decay_samples = int(self.duration * decay_time * self.samplerate)
        release_samples = int(self.duration * release_time * self.samplerate)
        sustain_samples = len(t) - (attack_samples + decay_samples + release_samples)
        if sustain_samples < 0:
            sustain_samples = 0

        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if decay_samples > 0:
            envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples)
        if sustain_samples > 0:
            envelope[attack_samples + decay_samples:attack_samples + decay_samples + sustain_samples] = sustain_level
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)

        return envelope

    def save_recording(self):
        if not self.recording_buffer:
            return
        full_audio = np.concatenate(self.recording_buffer)
        full_audio = np.int16(full_audio / np.max(np.abs(full_audio)) * 32767)

        base_name = "lifegame"
        i = 0
        while os.path.exists(f"{base_name}_{i}.wav"):
            i += 1
        filename = f"{base_name}_{i}.wav"
        write(filename, SAMPLE_RATE, full_audio)
        print(f"Saved: {filename}")

# Run
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Game of Life with Piano Sound (36x36 Grid)")
    app = SoundLife(root)
    root.mainloop()



