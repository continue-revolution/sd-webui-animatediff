import re

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
