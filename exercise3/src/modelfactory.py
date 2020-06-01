from typing import *

from models import ModelPicker, AbstractModel


class ModelFactory:
    _picker_string_dictionary = dict((enum_value.name, enum_value.value) for enum_value in ModelPicker)

    @classmethod
    def create(cls, picker: Union[ModelPicker, str], *args, **kwargs) -> AbstractModel:
        if isinstance(picker, ModelPicker):
            return picker.value(args, kwargs)

        return ModelFactory._picker_string_dictionary[picker](args, kwargs)
