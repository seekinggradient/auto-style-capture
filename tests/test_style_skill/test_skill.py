from auto_style_capture.style_skill.skill import StyleSkill


def test_skill_save_and_load(tmp_path):
    skill = StyleSkill(author_name="Hemingway", content="# Style Skill: Hemingway\n\nWrite short sentences.")
    path = skill.save(str(tmp_path))
    assert path.exists()
    assert "v0" in path.name

    loaded = StyleSkill.load(path)
    assert loaded.author_name == "Hemingway"
    assert "short sentences" in loaded.content


def test_skill_update():
    skill = StyleSkill(author_name="Hemingway", content="v0 content", version=0)
    updated = skill.update("v1 content")
    assert updated.version == 1
    assert updated.content == "v1 content"
    assert len(updated.history) == 1
    assert updated.history[0] == "v0 content"


def test_skill_to_prompt():
    skill = StyleSkill(author_name="Poe", content="Use dark imagery.")
    prompt = skill.to_prompt()
    assert "Poe" in prompt
    assert "dark imagery" in prompt
