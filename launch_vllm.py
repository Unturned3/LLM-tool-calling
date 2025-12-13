import random
import subprocess


def main():
    use_random_port = False

    model_name = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
    # model_name = "Qwen/Qwen3-4B-Thinking-2507"
    port = str(random.randint(1024, 65535)) if use_random_port else "8000"

    print(f"Using port: {port}")

    # fmt: off
    vllm_cmd = [
        "vllm", "serve", model_name,
        "--served-model-name", model_name.split("/")[-1],
        "--host", "0.0.0.0",
        "--port", port,
        "--max-model-len", "100k",  # We are limited by the 48GB VRAM
        "--gpu-memory-utilization", "0.95",

        # --enable-reasoning was deprecated in v0.9.0
        # qwen3 reasoning parser doesn't work if <think> is given in the prompt and not generated
        # by the model itself, which is what Qwen3-*-Thinking-* models do. We use the deepseek_r1
        # parser, which does handle this case.
        "--reasoning-parser", "deepseek_r1",

        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_xml",
    ]
    # fmt: on

    # Start the vllm process
    vllm_process = subprocess.Popen(vllm_cmd, stdout=None, stderr=None)

    vllm_process.wait()


if __name__ == "__main__":
    main()
