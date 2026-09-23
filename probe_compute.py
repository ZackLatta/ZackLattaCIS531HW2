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

def create_faiss_index(dimension: int, device: torch.device):
    """
    Creates a FAISS index, automatically routing to GPU if available,
    otherwise falling back to CPU.
    """
    
    cpu_index = faiss.IndexFlatL2(dimension)
    
    if device.type == "cuda":
        try:
            
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            print(f"FAISS Index: Transferred to GPU 0 (Dimension: {dimension})")
            return gpu_index
        except Exception as e:
            print(f"FAISS GPU transfer failed: {e}. Falling back to CPU index.")
            return cpu_index
    else:
        print(f"FAISS Index: Operating on CPU (Dimension: {dimension})")
        return cpu_index


if __name__ == "__main__":
    target_device = probe_compute();

    embedding_dim = 768
    index = create_faiss_index(embedding_dim, device)
    
    # verifies
    print(f"Pipeline ready. Total vectors in index: {index.ntotal}")