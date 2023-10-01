import re
import torch

from modules.processing import StableDiffusionProcessing

from scripts.animatediff_logger import logger_animatediff as logger


class AnimateDiffPromptSchedule:

    def __init__(self):
        self.prompt_map = None


    def parse_prompt(self, p: StableDiffusionProcessing):
        if type(p.prompt) is not str:
            logger.warn("prompt is not str, cannot support prompt map")
            return

        lines = prompt.strip().split('\n')
        data = {
            'head_prompts': [],
            'mapp_prompts': {},
            'tail_prompts': []
        }

        mode = 'head'
        for line in lines:
            if mode == 'head':
                if re.match(r'^\d+:', line):
                    mode = 'mapp'
                else:
                    data['head_prompts'].append(line)
                    
            if mode == 'mapp':
                match = re.match(r'^(\d+): (.+)$', line)
                if match:
                    frame, prompt = match.groups()
                    data['mapp_prompts'][int(frame)] = prompt
                else:
                    mode = 'tail'
                    
            if mode == 'tail':
                data['tail_prompts'].append(line)
        
        if data['mapp_prompts']:
            logger.info("You are using prompt travel.")
            self.prompt_map = {}
            prompt_list = []
            last_frame = 0
            current_prompt = ''
            for frame, prompt in data['mapp_prompts'].items():
                current_prompt = f"{', '.join(data['head_prompts'])}, {prompt}, {', '.join(data['tail_prompts'])}"
                self.prompt_map[frame] = current_prompt
                prompt_list += [current_prompt for _ in range(last_frame, frame)]
                last_frame = frame
            prompt_list += [current_prompt for _ in range(last_frame, p.batch_size)]
            assert len(prompt_list) == p.batch_size, f"prompt_list length {len(prompt_list)} != batch_size {p.batch_size}"
            p.prompt = prompt_list * p.n_iter


    def get_current_prompt_embeds_from_text(
        self,
        center_frame = None,
        video_length : int = 0,
        prompt_embeds_map : dict = None,
        ):

        key_prev = list(self.prompt_map.keys())[0]
        key_next = list(self.prompt_map.keys())[-1]

        for p in self.prompt_map.keys():
            if p > center_frame:
                key_next = p
                break
            key_prev = p

        dist_prev = center_frame - key_prev
        if dist_prev < 0:
            dist_prev += video_length
        dist_next = key_next - center_frame
        if dist_next < 0:
            dist_next += video_length

        if key_prev == key_next or dist_prev + dist_next == 0:
            return prompt_embeds_map[key_prev]

        rate = dist_prev / (dist_prev + dist_next)

        return AnimateDiffPromptSchedule.slerp( prompt_embeds_map[key_prev], prompt_embeds_map[key_next], rate )


    @staticmethod
    def slerp(
        v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995
    ) -> torch.Tensor:
        u0 = v0 / v0.norm()
        u1 = v1 / v1.norm()
        dot = (u0 * u1).sum()
        if dot.abs() > DOT_THRESHOLD:
            return (1.0 - t) * v0 + t * v1
        omega = dot.acos()
        return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()
