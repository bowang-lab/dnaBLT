from pydantic import BaseModel, ConfigDict


class BltExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    text: str
    tokens: list[int] | None
    entropies: list[float] | None
    patch_lengths: list[int] | None
    mask: list[bool] | None