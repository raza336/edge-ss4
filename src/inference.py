"""
inference.py
============
Q1.6 / Deliverable 9  -  ONE unified inference pipeline for all trained models:

    teacher | <student>_baseline | <student>_kd   (students: mobilenetv2, squeezenet, efficientnet_lite0)
    nas_best                                      (CIFAR-10 model found by the NAS)

Two back-ends: PyTorch (checkpoints/*.pth) and ONNX Runtime (onnx_models/*.onnx, CPU).

Examples
  python src/inference.py --model mobilenetv2_kd --num-images 8
  python src/inference.py --model nas_best --image my_photo.png --topk 3
  python src/inference.py --model all_students --num-images 5
  python src/inference.py --model efficientnet_lite0_kd --backend onnx --num-images 8

`Predictor` is imported by the benchmark scripts, so benchmarking and deployment share one code path.
"""
import argparse
import os

import numpy as np
import torch

from common import (ALL_MODELS, STUDENTS, ckpt_path, dataset_of, load_model_by_name, load_test_arrays,
                    normalize_u8, onnx_path, DATA_DIR)


class Predictor:
    """predictor(uint8 images [N,3,32,32]) -> float32 logits [N, num_classes]."""

    def __init__(self, name, backend="torch", threads=None):
        self.name, self.backend, self.dataset = name, backend, dataset_of(name)
        if backend == "torch":
            if threads:
                torch.set_num_threads(threads)
            self.path = ckpt_path(name)
            self.model = load_model_by_name(name).eval()
        elif backend == "onnx":
            import onnxruntime as ort
            self.path = onnx_path(name)
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])
        else:
            raise ValueError(f"unknown backend {backend}")

    @torch.no_grad()
    def __call__(self, x_u8):
        x = normalize_u8(x_u8, self.dataset)
        if self.backend == "torch":
            return self.model(x).float().numpy()
        return self.sess.run(None, {"input": x.numpy()})[0]

    def predict_batches(self, x_u8, batch_size=256):
        return np.concatenate([self(x_u8[i:i + batch_size]) for i in range(0, len(x_u8), batch_size)])


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def load_image(path):
    """Any image file -> uint8 tensor [1,3,32,32]."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((32, 32))
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).unsqueeze(0).contiguous()


def resolve_models(arg, backend):
    path_of = ckpt_path if backend == "torch" else onnx_path
    if arg == "all":
        names = ALL_MODELS
    elif arg == "all_students":
        names = [f"{s}_{k}" for s in STUDENTS for k in ("baseline", "kd")]
    else:
        names = [arg]
    present = [n for n in names if os.path.exists(path_of(n))]
    for n in set(names) - set(present):
        print(f"[skip] {n}: file not found ({path_of(n)})")
    return present


def main():
    ap = argparse.ArgumentParser(description="Inference with any trained model (PyTorch or ONNX Runtime)")
    ap.add_argument("--model", default="mobilenetv2_kd", help="model name | all_students | all")
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--image", help="path of an image file (otherwise CIFAR test images are used)")
    ap.add_argument("--num-images", type=int, default=8)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--data-dir", default=DATA_DIR)
    a = ap.parse_args()

    cache = {}
    for name in resolve_models(a.model, a.backend):
        pred = Predictor(name, a.backend)
        ds = pred.dataset
        if ds not in cache:
            x, y, classes = load_test_arrays(ds, a.data_dir)
            cache[ds] = (x, y, classes)
        x, y, classes = cache[ds]
        if a.image:
            xs, ys = load_image(a.image), None
        else:
            xs, ys = x[:a.num_images], y[:a.num_images]
        prob = softmax(pred(xs))
        order = np.argsort(-prob, axis=1)[:, :a.topk]
        print(f"\n=== model: {name} | backend: {a.backend} | dataset: {ds} | file: {os.path.basename(pred.path)}")
        for i in range(len(xs)):
            guess = ", ".join(f"{classes[j]} ({prob[i, j]:.1%})" for j in order[i])
            truth = f" | truth: {classes[ys[i]]}  {'OK' if order[i, 0] == ys[i].item() else 'WRONG'}" if ys is not None else ""
            print(f"  [{i}] {guess}{truth}")
        if ys is not None:
            acc = 100 * (order[:, 0] == ys.numpy()).mean()
            print(f"  top-1 accuracy on these {len(xs)} images: {acc:.1f}%")


if __name__ == "__main__":
    main()
