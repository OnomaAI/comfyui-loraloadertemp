from .loader import CLASS_MAPPINGS as LoadMapping, CLASS_NAMES as LoadNames
from .illustrious_generator import CLASS_MAPPINGS as GenerateMapping, CLASS_NAMES as GenerateNames

NODE_CLASS_MAPPINGS = {
}

NODE_CLASS_MAPPINGS.update(LoadMapping)
NODE_CLASS_MAPPINGS.update(GenerateMapping)


NODE_DISPLAY_NAME_MAPPINGS = {

}

NODE_DISPLAY_NAME_MAPPINGS.update(LoadNames)
NODE_DISPLAY_NAME_MAPPINGS.update(GenerateNames)
