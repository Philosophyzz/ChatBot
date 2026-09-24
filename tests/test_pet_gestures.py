import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QPixmap, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication


def test_smooth_wheel_and_direct_touch_areas(tmp_path, monkeypatch):
    from pet.client import BackendClient
    from pet.settings import PetSettings
    from pet.window import PetWindow
    app = QApplication.instance() or QApplication([])
    settings = PetSettings(tmp_path / 'pet.json')
    monkeypatch.setattr('pet.window.get_settings', lambda: settings)
    monkeypatch.setattr(PetWindow, 'refresh_personas', lambda self: None)
    monkeypatch.setattr(PetWindow, 'refresh_models', lambda self: None)
    window = PetWindow(BackendClient('http://127.0.0.1:1'), persona_id='sakura_cat', with_tray=False)
    window.state = 'idle'; window.muted = True
    try:
        window.set_scale(1)
        position = QPointF(105, 80)
        screen_position = QPointF(window.mapToGlobal(position.toPoint()))
        event = QWheelEvent(position, screen_position, QPoint(), QPoint(0, 120),
                            Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
        window.wheelEvent(event)
        assert window.scale_factor == 1, 'Wheel changes target; visual size should animate.'
        window._animate_scale()
        assert 1 < window.scale_factor < window._scale_target < 1.1
        for _ in range(30): window._animate_scale()
        assert window.scale_factor == window._scale_target
        QTest.qWait(400)
        assert settings.get('scale') == window._scale_target
        window.set_scale(-1); assert window.scale_factor == .35
        window.set_scale(9); assert window.scale_factor == 3
        hits = []
        monkeypatch.setattr(window, 'touch', lambda kind: hits.append(kind))
        for scale in (.5, 1, 2.5):
            window.set_scale(scale)
            frame = QPixmap(window.size()); window.render(frame)
            width, height = window._sprite_size
            def mapped(x, y):
                return window._sprite_transform.map(QPointF((x-.5)*width, (y-1)*height)).toPoint()
            for point, expected in [((.52,.20),'head'),((.59,.36),'face'),((.52,.38),'feed'),((.52,.64),'body')]:
                QTest.mouseClick(window, Qt.LeftButton, pos=mapped(*point))
                QTest.qWait(QApplication.doubleClickInterval()+80)
                assert hits[-1] == expected
            assert window._touch_at(QPointF(0,0)) is None
        QTest.mousePress(window, Qt.LeftButton, pos=mapped(.52,.64))
        QTest.qWait(750)
        QTest.mouseRelease(window, Qt.LeftButton, pos=mapped(.52,.64))
        assert hits[-1] == 'hold'
        before = len(hits)
        QTest.mousePress(window, Qt.LeftButton, pos=mapped(.52,.64))
        QTest.mouseMove(window, mapped(.52,.64)+QPoint(35,20))
        QTest.mouseRelease(window, Qt.LeftButton, pos=mapped(.52,.64)+QPoint(35,20))
        QTest.qWait(500)
        assert len(hits) == before, 'Dragging must not also trigger a touch.'
        chats = []
        monkeypatch.setattr(window.bubble, 'ask', lambda: chats.append(True))
        # QtTest treats the null QPoint(0, 0) as the widget centre.
        QTest.mouseDClick(window, Qt.LeftButton, pos=QPoint(1, 1))
        assert len(hits) == before and not chats, 'Transparent pixels must ignore double clicks too.'
        QTest.mouseDClick(window, Qt.LeftButton, pos=mapped(.52, .64))
        QTest.mouseRelease(window, Qt.LeftButton, pos=mapped(.52, .64))
        assert hits[-1] == 'double' and chats == [True]

        # The offline menu must change the identity, not put another character's art on it.
        monkeypatch.setattr('pet.single_instance.acquire', lambda key: True)
        monkeypatch.setattr('pet.single_instance.release', lambda key: None)
        monkeypatch.setattr(window, 'refresh_emotion', lambda: None)
        window.personas = []
        window.pet_mood = 'caring'
        window.choose_builtin_skin('mint_bunny')
        assert window.persona_id == 'mint_bunny' and window.persona_name == '薄荷'
        assert window.pet_mood == 'calm'
        assert settings.get('skins')['mint_bunny'] == 'builtin:mint_bunny'
        menu = window._menu(from_pet=True)
        labels = [action.text() for action in menu.actions()]
        assert '更换形象' in labels and '切换人设' not in labels and '互动' not in labels
        menu.deleteLater()
    finally:
        window._anim.stop(); window._scale_timer.stop(); window._scale_save_timer.stop()
        window.bubble.close(); window.close(); window.client.close()
