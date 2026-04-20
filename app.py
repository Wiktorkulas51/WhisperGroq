import os
import threading
import tempfile
import queue
import time
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
import keyboard
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from dotenv import load_dotenv
import pystray
from pystray import MenuItem as item
from PIL import Image, ImageDraw

# load environment variables from .env (if present)
load_dotenv()

try:
    from groq import Groq
except Exception:
    Groq = None

API_KEY = os.getenv("GROQ_API_KEY")
HOTKEY = os.getenv("HOTKEY", "f9")  # default hotkey: F9 (press-and-hold)

# audio defaults
SR = 44100
CHANNELS = 1


class Recorder:
    def __init__(self, samplerate=SR, channels=CHANNELS):
        self.sr = samplerate
        self.channels = channels
        self._frames = []
        self._rec_thread = None
        self._running = threading.Event()
        self.last_error = None
        self._started_event = threading.Event()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Recording status:", status, file=sys.stderr)
        # copy because indata is reused by sounddevice
        self._frames.append(indata.copy())

    def start(self):
        self._frames = []
        self.last_error = None
        self._started_event.clear()
        self._running.set()
        def _run():
            try:
                with sd.InputStream(samplerate=self.sr, channels=self.channels, callback=self._callback):
                    # signal that the input stream opened successfully
                    self._started_event.set()
                    while self._running.is_set():
                        sd.sleep(100)
            except Exception as e:
                # capture any device/opening errors
                self.last_error = str(e)
                self._running.clear()

        self._rec_thread = threading.Thread(target=_run, daemon=True)
        self._rec_thread.start()

    def stop(self, outpath: str):
        self._running.clear()
        if self._rec_thread is not None:
            self._rec_thread.join()
        if self.last_error:
            raise RuntimeError(self.last_error)
        if not self._frames:
            raise RuntimeError("No audio recorded")
        data = np.concatenate(self._frames, axis=0)
        # ensure shape (n, channels)
        sf.write(outpath, data, self.sr)
        return outpath


# list of models to alternate between
MODELS = ["whisper-large-v3", "whisper-large-v3-turbo"]
_model_index = 0
_model_lock = threading.Lock()

def transcribe_with_groq(audio_path: str):
    global _model_index
    if Groq is None:
        raise RuntimeError("Groq package not installed or failed to import")
    if not API_KEY:
        raise RuntimeError("Set GROQ_API_KEY environment variable first")

    with _model_lock:
        model_name = MODELS[_model_index]
        _model_index = (_model_index + 1) % len(MODELS)

    print(f"Transcribing using model: {model_name}")

    client = Groq(api_key=API_KEY)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    resp = client.audio.transcriptions.create(
        file=(os.path.basename(audio_path), audio_bytes),
        model=model_name,
        temperature=0,
        response_format="verbose_json",
    )
    # object shape depends on SDK; try common access patterns
    if hasattr(resp, "text"):
        return resp.text
    if isinstance(resp, dict):
        return resp.get("text") or resp.get("transcription") or str(resp)
    return str(resp)


class GUIManager:
    def __init__(self):
        self._queue = queue.Queue()
        self.root = tk.Tk()
        # keep the root window hidden; use Toplevel windows for status
        self.root.withdraw()
        self._window = None
        self._label = None
        self._poll()

    def _poll(self):
        try:
            while True:
                cmd, arg = self._queue.get_nowait()
                if cmd == "show":
                    self._do_show(arg)
                elif cmd == "update":
                    self._do_update(arg)
                elif cmd == "close":
                    self._do_close()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _do_show(self, text: str):
        # Simplified dark rounded text-only popup placed bottom-center.
        # Accept either a plain text or a (mode, text) tuple; we only use the text here.
        body = text[1] if (isinstance(text, (tuple, list)) and len(text) > 1) else text

        if self._window:
            self._do_update(body)
            return

        self._window = tk.Toplevel(self.root)
        try:
            self._window.overrideredirect(True)
        except Exception:
            pass
        self._window.wm_attributes("-topmost", True)

        # Create a unique transparent color key for the window background
        transparent_key = "#123456"
        bg_color = "#222222"
        fg_color = "#ffffff"
        pad_x = 18
        pad_y = 10

        # set the window background to the transparent key
        try:
            self._window.configure(bg=transparent_key)
        except Exception:
            pass

        # Canvas will draw the rounded rectangle and the text
        font_obj = tkfont.Font(family=None, size=11)
        text_width = font_obj.measure(body)
        text_height = font_obj.metrics("linespace")
        width = text_width + pad_x * 2
        height = text_height + pad_y * 2

        canvas = tk.Canvas(self._window, width=width, height=height, highlightthickness=0, bg=transparent_key)
        canvas.pack()

        # draw rounded rectangle using ovals and rectangles
        r = 12
        color = bg_color
        # center rectangle parts
        canvas.create_rectangle(r, 0, width - r, height, fill=color, outline=color)
        canvas.create_rectangle(0, r, width, height - r, fill=color, outline=color)
        # four corner ovals
        canvas.create_oval(0, 0, 2 * r, 2 * r, fill=color, outline=color)
        canvas.create_oval(width - 2 * r, 0, width, 2 * r, fill=color, outline=color)
        canvas.create_oval(0, height - 2 * r, 2 * r, height, fill=color, outline=color)
        canvas.create_oval(width - 2 * r, height - 2 * r, width, height, fill=color, outline=color)

        # draw text
        self._text_id = canvas.create_text(width // 2, height // 2, text=body, fill=fg_color, font=font_obj)

        # try to make the background transparent (Windows supports -transparentcolor)
        try:
            self._window.wm_attributes("-transparentcolor", transparent_key)
        except Exception:
            try:
                self._window.configure(bg=color)
            except Exception:
                pass

        # position bottom-center
        self._window.update_idletasks()
        ws = self._window.winfo_screenwidth()
        hs = self._window.winfo_screenheight()
        x = (ws // 2) - (width // 2)
        y = hs - height - 60
        self._window.geometry(f"{width}x{height}+{x}+{y}")
        self._mode = "textpopup"
        self._canvas = canvas
        self._font_obj = font_obj

    def _do_update(self, text: str):
        # accept either a plain text or a (mode, text) tuple
        mode = None
        body = text
        if isinstance(text, tuple) or isinstance(text, list):
            mode, body = text[0], text[1]

        if mode and mode != self._mode:
            # recreate window with new mode
            self._do_close()
            self._do_show((mode, body))
            return

            body = text[1] if (isinstance(text, (tuple, list)) and len(text) > 1) else text
            if hasattr(self, "_canvas") and self._canvas and hasattr(self, "_text_id"):
                # update text and resize background if needed
                self._canvas.itemconfig(self._text_id, text=body)
                # recompute size
                text_width = self._font_obj.measure(body)
                text_height = self._font_obj.metrics("linespace")
                pad_x = 18
                pad_y = 10
                width = text_width + pad_x * 2
                height = text_height + pad_y * 2
                self._canvas.config(width=width, height=height)
                # redraw rounded rect by clearing and recreating shapes
                self._canvas.delete("all")
                r = 12
                color = "#222222"
                self._canvas.create_rectangle(r, 0, width - r, height, fill=color, outline=color)
                self._canvas.create_rectangle(0, r, width, height - r, fill=color, outline=color)
                self._canvas.create_oval(0, 0, 2 * r, 2 * r, fill=color, outline=color)
                self._canvas.create_oval(width - 2 * r, 0, width, 2 * r, fill=color, outline=color)
                self._canvas.create_oval(0, height - 2 * r, 2 * r, height, fill=color, outline=color)
                self._canvas.create_oval(width - 2 * r, height - 2 * r, width, height, fill=color, outline=color)
                self._text_id = self._canvas.create_text(width // 2, height // 2, text=body, fill="#ffffff", font=self._font_obj)
                # reposition window bottom-center
                self._window.update_idletasks()
                ws = self._window.winfo_screenwidth()
                hs = self._window.winfo_screenheight()
                x = (ws // 2) - (width // 2)
                y = hs - height - 60
                self._window.geometry(f"{width}x{height}+{x}+{y}")

    def _do_close(self):
        if self._window:
            try:
                self._window.destroy()
            except Exception:
                pass
            self._window = None
            self._label = None

    def show(self, text: str):
        self._queue.put(("show", text))

    def update(self, text: str):
        self._queue.put(("update", text))

    def close(self):
        self._queue.put(("close", None))


def main():
    def print_banner(hotkey: str):
        banner = r"""
 __        ___     _                      ____                  
 \ \      / / |__ (_)___ _ __   ___ _ __ / ___|_ __ ___   __ _  
  \ \ /\ / /| '_ \| / __| '_ \ / _ \ '__| |  _| '__/ _ \ / _` | 
   \ V  V / | | | | \__ \ |_) |  __/ |  | |_| | | | (_) | (_| | 
    \_/\_/  |_| |_|_|___/ .__/ \___|_|   \____|_|  \___/ \__, | 
                        |_|                                 |_| 
"""
        print(banner)
        print(f"Developer: Rohan Sharvesh")
        print("License: MIT (see LICENSE file)")
        print("")
        print(f"Hold '{hotkey}' to record; release to send audio for transcription.")
        print("")

    print_banner(HOTKEY)
    # Use ctrl+alt as default if not in .env
    current_hotkey = os.getenv("HOTKEY", "ctrl+alt")
    print_banner(current_hotkey)
    recorder = Recorder()
    gui = GUIManager()

    recording_state = {"is_recording": False, "last_toggle_time": 0}
    state_lock = threading.Lock()

    def toggle_recording(e=None):
        with state_lock:
            now = time.time()
            # Debounce: ignore presses within 500ms
            if now - recording_state["last_toggle_time"] < 0.5:
                return
            recording_state["last_toggle_time"] = now

            if not recording_state["is_recording"]:
                # START RECORDING
                print("Start recording (toggle)")
                try:
                    gui.show(("record", f"Recording... [Press {current_hotkey} to stop]"))
                except Exception:
                    pass
                recorder.last_error = None
                recorder.start()
                started = recorder._started_event.wait(timeout=1.0)
                if not started:
                    msg = recorder.last_error or "Timeout opening audio input device"
                    gui.update(f"Recording error: {msg}")
                    time.sleep(1.5)
                    gui.close()
                    return
                recording_state["is_recording"] = True
            else:
                # STOP RECORDING
                print("Stop recording (toggle)")
                recording_state["is_recording"] = False
                try:
                    fd, path = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)
                    gui.update("Stopping... saving audio")
                    audio_path = recorder.stop(path)
                except Exception as ex:
                    print("Recording error:", ex, file=sys.stderr)
                    gui.update(f"Recording error: {ex}")
                    time.sleep(2)
                    gui.close()
                    return

                def _process():
                    try:
                        gui.update(("processing", "Processing audio..."))
                        text = transcribe_with_groq(audio_path)
                        if not text:
                            text = "(no transcription returned)"
                        print("Transcription:\n", text)

                        try:
                            import pyperclip
                            pyperclip.copy(text)
                        except Exception:
                            try:
                                def _cb():
                                    try:
                                        gui.root.clipboard_clear()
                                        gui.root.clipboard_append(text)
                                        gui.root.update()
                                    except Exception:
                                        pass
                                gui.root.after(0, _cb)
                            except Exception:
                                pass

                        try:
                            snippet = text if len(text) <= 300 else text[:300] + "..."
                            gui.show(("done", snippet))
                        except Exception:
                            pass

                        time.sleep(0.2)
                        try:
                            keyboard.press_and_release('ctrl+v')
                        except Exception:
                            pass

                        time.sleep(1.2)
                        try:
                            gui.close()
                        except Exception:
                            pass

                    except Exception as ex:
                        gui.update(f"Transcription error: {ex}")
                        print("Transcription error:", ex, file=sys.stderr)
                    finally:
                        try:
                            os.remove(audio_path)
                        except Exception:
                            pass

                t = threading.Thread(target=_process, daemon=True)
                t.start()

    # Register toggle handler
    try:
        keyboard.add_hotkey(current_hotkey, toggle_recording)
    except Exception as ex:
        print(f"Error registering hotkey: {ex}")

    def create_image():
        # Create a simple icon: a circle with a letter 'W'
        width = 64
        height = 64
        color1 = "#222222"
        color2 = "#ffffff"
        image = Image.new('RGB', (width, height), color1)
        dc = ImageDraw.Draw(image)
        dc.ellipse((8, 8, 56, 56), fill="#50C878") # emerald green
        return image

    def on_quit(icon, item):
        icon.stop()
        gui.root.after(0, gui.root.quit)
        os._exit(0)

    # Setup tray icon
    menu = (item('WhisperGroq (Active)', lambda: None, enabled=False), 
            item('Quit', on_quit))
    icon = pystray.Icon("WhisperGroq", create_image(), "WhisperGroq", menu)
    
    # Run icon in a separate thread
    threading.Thread(target=icon.run, daemon=True).start()

    print(f"Ready. Press '{current_hotkey}' to start/stop recording.")
    try:
        # Show a quick startup notification
        gui.root.after(500, lambda: gui.show(("info", "WhisperGroq is running in background")))
        gui.root.after(3000, gui.close)
        
        gui.root.mainloop()
    except KeyboardInterrupt:
        print("Exiting")
        try:
            gui.root.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
