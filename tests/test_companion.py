import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from core.companion import CompanionDirector, InterventionGate, bounded_parameters
from pet.live2d import ActionTimeline
from llm.vllm_supervisor import wsl_path


def observation(**values):
    return dict(enabled=True, scene="normal", idle_s=20, pet_idle_s=100, **values)


def test_gate_cooldown_hourly_and_no_intervention_while_busy():
    gate = InterventionGate()
    assert not gate.check(observation(), now=0)
    gate.reserve(now=0)
    assert gate.check(observation(), now=20)
    assert gate.check(observation(busy=True), now=100)
    gate.record(now=100)
    assert gate.check(observation(), now=650)
    assert not gate.check(observation(), now=701)
    for stamp in (800, 1500, 2200):
        gate.record(now=stamp)
    assert gate.check(observation(), now=3000) == "本小时主动发言已达上限"
    assert not gate.check(observation(), now=3800)


@pytest.mark.parametrize("change", [{"enabled": False}, {"scene": "quiet"}, {"protected": True}, {"idle_s": 2}, {"idle_s": 1000}, {"pet_idle_s": 3}])
def test_local_gate_suppresses_unwanted_llm_calls(change):
    obs = observation()
    obs.update(change)
    assert InterventionGate().check(obs, now=100)


def test_action_limits_and_return_to_neutral():
    caps = {"ParamAngleX": {"min": -30, "max": 30, "default": 0}, "ParamMouthOpenY": {"min": 0, "max": 1, "default": 0}}
    assert bounded_parameters({"ParamAngleX": 800, "ParamMouthOpenY": 1, "Unknown": 5}, caps) == {"ParamAngleX": 30}
    assert bounded_parameters({"ParamAngleX": math.nan}, caps) == {}
    timeline = ActionTimeline(caps)
    timeline.set({"frames": [{"duration_ms": 1000, "parameters": {"ParamAngleX": 30}}]}, now=0)
    assert timeline.values(.5)["ParamAngleX"] == pytest.approx(15)
    assert timeline.values(1)["ParamAngleX"] == 30
    assert timeline.values(3) == {}


def test_remote_observation_never_sends_image_to_llm():
    app = SimpleNamespace(config=SimpleNamespace(llm=SimpleNamespace(mode="api", base_url="https://example.com/v1")), llm=SimpleNamespace(complete=AsyncMock()))
    director = CompanionDirector(app)
    response = asyncio.run(director.observe(observation(), "private-image", "hiyori", {}))
    assert not response["speak"]
    app.llm.complete.assert_not_called()


def test_wsl_path_does_not_interpolate_shell_commands():
    assert wsl_path("D:/Models/My Model") == "/mnt/d/Models/My Model"
    assert wsl_path("D:\\Harness\\ChatBot") == "/mnt/d/Harness/ChatBot"


def test_vllm_pending_restart_cannot_stop_or_start_model(tmp_path, monkeypatch):
    from llm.vllm_supervisor import VllmSupervisor, ModelSwitchError
    manager = VllmSupervisor(tmp_path)
    monkeypatch.setattr(manager, "specification", lambda: {"models": [{"id": "mimo", "path": str(tmp_path)}]})
    monkeypatch.setattr(manager, "installed", lambda: {"state": "reboot_required", "detail": "needs reboot"})
    calls = []
    monkeypatch.setattr(manager, "run", lambda *args: calls.append(args))
    with pytest.raises(ModelSwitchError, match="needs reboot"):
        manager.switch("mimo")
    assert manager.stop() == {"stopped": False}
    assert calls == []
