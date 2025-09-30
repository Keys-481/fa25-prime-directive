def test_import():
    import potato_mission.mission_node as m
    assert hasattr(m, "MissionNode")
