"""Interface reserved by the implementation plan."""



def generate(prompt, output, *, artifacts, device, dtype, steps, cfg_scale, lat_h, lat_w, seed, system_prompt, save_tokens, low_memory):
    raise NotImplementedError



def ask(input_path, question, *, artifacts, device, dtype, lat_h, lat_w, low_memory):
    raise NotImplementedError



def text(prompt, *, artifacts, device, dtype, system_prompt):
    raise NotImplementedError
