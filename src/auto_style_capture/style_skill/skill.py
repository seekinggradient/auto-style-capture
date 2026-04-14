from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StyleSkill:
    author_name: str
    content: str
    version: int = 0
    history: list[str] = field(default_factory=list)

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"style_skill_v{self.version}.md"
        path.write_text(self.content, encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> StyleSkill:
        path = Path(path)
        content = path.read_text(encoding="utf-8")
        # Try to extract version from filename
        version = 0
        if "_v" in path.stem:
            try:
                version = int(path.stem.split("_v")[-1])
            except ValueError:
                pass
        # Extract author name from first heading
        author_name = "Unknown Author"
        for line in content.split("\n"):
            if line.startswith("# Style Skill:"):
                author_name = line.replace("# Style Skill:", "").strip()
                break
        return cls(author_name=author_name, content=content, version=version)

    def update(self, new_content: str) -> StyleSkill:
        """Create a new version of the skill with updated content.

        If new_content is empty, keeps the current content to avoid regression.
        """
        if not new_content or not new_content.strip():
            return StyleSkill(
                author_name=self.author_name,
                content=self.content,
                version=self.version + 1,
                history=[*self.history, self.content],
            )
        return StyleSkill(
            author_name=self.author_name,
            content=new_content,
            version=self.version + 1,
            history=[*self.history, self.content],
        )

    def to_prompt(self) -> str:
        return (
            f"You are writing in the style of {self.author_name}. "
            f"Follow these style instructions precisely:\n\n{self.content}"
        )
