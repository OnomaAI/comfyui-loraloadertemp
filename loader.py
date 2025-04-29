import folder_paths
import comfy
from .autonode import node_wrapper, get_node_names_mappings, validate


fundamental_classes = []
fundamental_node = node_wrapper(fundamental_classes)

@fundamental_node
class LoraLoaderTemp:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "input_string": ("STRING", {"tooltip": "select lora name from input string"}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }
    
    FUNCTION = "load_lora_temp"
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    CATEGORY = "loaders"
    custom_name = "load lora temp"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    def load_lora_temp(self, model, clip, input_string, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
        if "aos0" in input_string:
            lora_name = 'Aosora_style-000012.safetensors'
            trigger_words = "AOS0_STY, "
        elif "amelicart" in input_string:
            lora_name = 'Amelicart_style_Illustrious_Xl.safetensors'
            trigger_words = "0MEL1C0RT, CHIBI ONLY, "
        elif "celtictat" in input_string:
            lora_name = 'Celtic_world_illustrious.safetensors'
            trigger_words = "CELTICTAT, "
        elif "chrome" in input_string or "metallic" in input_string:
            lora_name = 'ChromeNoob_byKonan.safetensors'
            trigger_words = "CHROME, SHINY SKIN, CHROME CLOTHING, CHROME SKIN, "
        elif "CARTOON" in input_string or "COMIC" in input_string:
            lora_name = 'ANavarroCabreraXL_style-12.safetensors'
            trigger_words = "NAVARROCABRERAXL,"
        elif "marvel" in input_string:
            lora_name = 'Marvel_Ilu.safetensors'
            trigger_words = "MARVELRIVALS, "
        elif "figure" in input_string:
            lora_name = 'nendoroid_xl_illustrious01_v1.0.safetensors'
            trigger_words = "NENDOROID, CHIBI, FULL BODY, "
        elif "neon" in input_string or "cyber" in input_string or "cyberpunk" in input_string:
            lora_name = 'neon_cyber_fantasy_illustriousXL.safetensors'
            trigger_words = "0MEL1C0RT, CHIBI ONLY, "
        elif "3D" in input_string or "real" in input_string:
            lora_name = 'Retro3DstyleP1-000078.safetensors'
            trigger_words = "RETRO3D, 3D, "
        elif "retro" in input_string:
            lora_name = 'rurouni_kenshin_1996_illustriousXL.safetensors'
            trigger_words = "RUROUNI_KENSHIN_1996_STYLE, "
        elif "summertime saga" in input_string:
            lora_name = 'Summertime Saga Style [Illustrious].safetensors'
            trigger_words = "SMMRTMSGA, "
        elif "doujinshi" in input_string or "doujin" in input_string or "manga" in input_string:
            lora_name = 'Takeratsu Style - NatMontero.safetensors'
            trigger_words = " "
        elif "twistedscarlet" in input_string:
            lora_name = 'TwistedScarlett-Illustrious-000017.safetensors'
            trigger_words = "SC4RL, "
        else:
            lora_name = "spo-ep34.safetensors"
            trigger_words = " "

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora, trigger_words)

CLASS_MAPPINGS, CLASS_NAMES = get_node_names_mappings(fundamental_classes)
validate(fundamental_classes)