"""Explicitly started, cancellable stereo sessions, streamed in small PCM blocks."""
import secrets
import threading
import time
from PySide6.QtCore import QObject, QTimer, Signal
from speech.asmr import PRESETS, RATE, TextureLibrary, parse_command
from pet.settings import project_root


class ASMRController(QObject):
    finished = Signal(int, str)

    def __init__(self, pet):
        super().__init__(pet)
        self.pet = pet
        self.active = False
        self.paused = False
        self.epoch = 0
        self.elapsed = 0.0
        self.duration = 0
        self.preset = 'mixed'
        self.volume = float(pet.settings.get('asmr_volume') or .3)
        self.minutes = int(pet.settings.get('asmr_minutes') or 15)
        self.cancel = threading.Event()
        self.thread = None
        self.finished.connect(self._finished)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(pet.update)

    @property
    def label(self):
        seconds = max(0, int(self.duration - self.elapsed))
        return ('ASMR 已暂停' if self.paused else 'ASMR 无语音') + f'\n{seconds // 60:02}:{seconds % 60:02}'

    def start(self, preset='mixed', seconds=None):
        if preset not in PRESETS:
            return
        pet = self.pet
        if pet._chat_busy or pet._recording:
            pet.bubble.show_text('等这次聊天或录音结束后，再开始 ASMR。', autohide_s=4)
            return
        self.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=.35)
        if self.thread and self.thread.is_alive():
            pet.bubble.show_text('正在结束上一段，请稍后再试。', autohide_s=3)
            return
        pet._voice_epoch += 1
        pet.player.stop()
        if pet.hands_free:
            pet.toggle_hands_free()
        pet.awareness.interact()
        pet.awareness.audio.stop()
        self.epoch += 1
        epoch = self.epoch
        self.cancel = threading.Event()
        self.active, self.paused = True, False
        self.elapsed = 0.0
        self.duration = int(seconds or self.minutes * 60)
        self.preset = preset
        self.timer.start()
        pet.bubble.show_text(f'{PRESETS[preset]} · 无语音\n右键 ASMR 可暂停、调音量或停止。', autohide_s=5)
        self.thread = threading.Thread(target=self._play, args=(epoch, self.cancel), daemon=True, name='pet-asmr')
        self.thread.start()

    def _play(self, epoch, cancel):
        error = ''
        try:
            import numpy as np
            import sounddevice as sd
            library = TextureLibrary(project_root() / 'data/asmr/library')
            blocks = library.blocks(self.preset, self.duration, seed=secrets.randbits(32))
            last = np.zeros((4410, 2), dtype=np.float32)
            gain = 0.0
            with sd.OutputStream(samplerate=RATE, channels=2, dtype='float32', blocksize=4410) as stream:
                for block in blocks:
                    while self.paused and not cancel.is_set():
                        stream.write(np.zeros_like(last))
                    if cancel.is_set():
                        # A short exit ramp avoids an abrupt cutoff; only this stream is stopped.
                        stream.write(last * np.linspace(1, 0, len(last), dtype=np.float32)[:, None])
                        break
                    target = max(0, min(.8, self.volume))
                    ramp = np.linspace(gain, target, len(block), dtype=np.float32)
                    last = block * ramp[:, None]
                    stream.write(last)
                    gain = target
                    self.elapsed += len(block) / RATE
        except Exception as exc:
            error = str(exc)
        finally:
            try:
                self.finished.emit(epoch, error)
            except RuntimeError:
                pass  # Qt may already have destroyed the controller during application exit.

    def _finished(self, epoch, error):
        if epoch != self.epoch:
            return
        self.active = False
        self.timer.stop()
        self.pet.awareness.interact()
        self.pet.update()
        if error:
            self.pet.bubble.show_text('ASMR 播放失败：' + error[:180], autohide_s=8)

    def stop(self):
        self.cancel.set()
        self.active, self.paused = False, False
        self.timer.stop()
        self.pet.update()

    def toggle_pause(self):
        if self.active:
            self.paused = not self.paused
            self.pet.update()

    def handle_command(self, text):
        command = parse_command(text)
        if not command:
            return False
        if command['action'] == 'start':
            self.start(command['preset'], command['seconds'])
        elif command['action'] == 'stop':
            self.stop()
            self.pet.bubble.show_text('ASMR 已停止。', autohide_s=3)
        elif self.active:
            self.paused = command['action'] == 'pause'
            self.pet.update()
        else:
            self.pet.bubble.show_text('现在没有正在播放的 ASMR。', autohide_s=3)
        return True

    def set_volume(self, value):
        self.volume = max(.05, min(.8, float(value)))
        self.pet.settings.set('asmr_volume', self.volume)

    def set_minutes(self, value):
        self.minutes = int(value)
        self.pet.settings.set('asmr_minutes', self.minutes)

    def add_menu(self, parent):
        menu = parent.addMenu('ASMR（无语音）')
        for key, name in PRESETS.items():
            menu.addAction(name).triggered.connect(lambda checked=False, value=key: self.start(value))
        menu.addAction('试听 45 秒').triggered.connect(lambda: self.start('mixed', 45))
        duration = menu.addMenu('时长（下次生效）')
        for minutes in (5, 15, 30):
            a = duration.addAction(f'{minutes} 分钟')
            a.setCheckable(True)
            a.setChecked(self.minutes == minutes)
            a.triggered.connect(lambda checked=False, value=minutes: self.set_minutes(value))
        volume = menu.addMenu('音量')
        for level in (.05, .15, .3, .5, .8):
            a = volume.addAction(f'{level:.0%}')
            a.setCheckable(True)
            a.setChecked(abs(self.volume - level) < .001)
            a.triggered.connect(lambda checked=False, value=level: self.set_volume(value))
        pause = menu.addAction('继续播放' if self.paused else '暂停')
        pause.setEnabled(self.active)
        pause.triggered.connect(self.toggle_pause)
        stop = menu.addAction('停止 ASMR')
        stop.setEnabled(self.active)
        stop.triggered.connect(self.stop)
        hint = menu.addAction('本地音效库；神经样本需试听后收录')
        hint.setEnabled(False)
        return menu
