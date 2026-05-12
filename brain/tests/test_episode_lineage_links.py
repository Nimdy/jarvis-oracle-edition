from memory.episodes import EpisodicMemory


def test_episode_turns_capture_trace_and_ledger_linkage(tmp_path) -> None:
    persist_path = tmp_path / "episodes.json"
    episodes = EpisodicMemory(persist_path=str(persist_path))

    episodes.add_user_turn(
        "hello jarvis",
        speaker="alice",
        emotion="neutral",
        conversation_id="conv_test",
        trace_id="trc_test",
        request_id="req_test_1",
        conversation_entry_id="led_user_1",
        root_entry_id="led_user_1",
    )
    episodes.add_assistant_turn(
        "hello alice",
        conversation_id="conv_test",
        trace_id="trc_test",
        request_id="req_test_1",
        output_id="out_test_1",
        conversation_entry_id="led_user_1",
        response_entry_id="led_resp_1",
        root_entry_id="led_user_1",
    )

    ep = episodes.get_active_episode()
    assert ep is not None
    assert len(ep.turns) == 2

    user_turn = ep.turns[0]
    assert user_turn.conversation_id == "conv_test"
    assert user_turn.trace_id == "trc_test"
    assert user_turn.request_id == "req_test_1"
    assert user_turn.conversation_entry_id == "led_user_1"
    assert user_turn.root_entry_id == "led_user_1"

    assistant_turn = ep.turns[1]
    assert assistant_turn.conversation_id == "conv_test"
    assert assistant_turn.trace_id == "trc_test"
    assert assistant_turn.request_id == "req_test_1"
    assert assistant_turn.output_id == "out_test_1"
    assert assistant_turn.conversation_entry_id == "led_user_1"
    assert assistant_turn.response_entry_id == "led_resp_1"
    assert assistant_turn.root_entry_id == "led_user_1"


def test_episode_lineage_fields_persist_and_reload(tmp_path) -> None:
    persist_path = tmp_path / "episodes.json"
    episodes = EpisodicMemory(persist_path=str(persist_path))
    episodes.add_user_turn(
        "start",
        conversation_id="conv_reload",
        trace_id="trc_reload",
        request_id="req_reload",
        conversation_entry_id="led_user_reload",
        root_entry_id="led_user_reload",
    )
    episodes.add_assistant_turn(
        "ack",
        conversation_id="conv_reload",
        trace_id="trc_reload",
        request_id="req_reload",
        output_id="out_reload",
        conversation_entry_id="led_user_reload",
        response_entry_id="led_resp_reload",
        root_entry_id="led_user_reload",
    )
    episodes.save()

    reloaded = EpisodicMemory(persist_path=str(persist_path))
    recent = reloaded.get_recent_episodes(count=1)
    assert len(recent) == 1
    turns = recent[0].turns
    assert len(turns) == 2
    assert turns[0].conversation_entry_id == "led_user_reload"
    assert turns[0].root_entry_id == "led_user_reload"
    assert turns[1].output_id == "out_reload"
    assert turns[1].response_entry_id == "led_resp_reload"
