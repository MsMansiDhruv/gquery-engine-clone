"""Columnar schema and type definitions modeled after Apache Arrow."""

from dataclasses import dataclass
from typing import List, Optional


class DataType:
    """Base class for all columnar types."""
    def __repr__(self) -> str:
        return self.__class__.__name__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)


class Int64(DataType):
    pass


class Float64(DataType):
    pass


class Utf8(DataType):
    pass


class Boolean(DataType):
    pass


@dataclass(frozen=True)
class Field:
    name: str
    data_type: DataType
    nullable: bool = True

    def __repr__(self) -> str:
        null_str = "nullable" if self.nullable else "not null"
        return f"{self.name}: {self.data_type} ({null_str})"


class Schema:
    """Represents the schema of a RecordBatch or Table."""

    def __init__(self, fields: List[Field]):
        self.fields = list(fields)
        self._name_to_index = {f.name: idx for idx, f in enumerate(self.fields)}

    def field_by_name(self, name: str) -> Optional[Field]:
        idx = self._name_to_index.get(name)
        return self.fields[idx] if idx is not None else None

    def index_of(self, name: str) -> int:
        if name not in self._name_to_index:
            raise KeyError(f"Field '{name}' not found in schema {[f.name for f in self.fields]}")
        return self._name_to_index[name]

    def project(self, column_names: List[str]) -> "Schema":
        return Schema([self.fields[self.index_of(name)] for name in column_names])

    def __repr__(self) -> str:
        return f"Schema({', '.join(repr(f) for f in self.fields)})"

    def __len__(self) -> int:
        return len(self.fields)
