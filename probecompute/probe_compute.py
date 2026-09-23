import torch

def probe_compute():

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print("GPU Pool Active - Batch Embedding/Whisper Ready")
        print(f"Device: {device_name}")
        device = torch.device("cuda")
    else:
        print("CPU-Only Fallback Active")
        device = torch.device("cpu")




if __name__ == "__main__":
    target_device = probe_compute();