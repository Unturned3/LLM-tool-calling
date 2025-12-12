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
        "--host", "0.0.0.0",
        "--port", port,
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.9",

        # --enable-reasoning was deprecated in v0.9.0
        #"--reasoning-parser", "qwen3",
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
