import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import librosa
import librosa.display
from IPython.display import Audio, display
from matplotlib.colors import TwoSlopeNorm

class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]
    
class ClassifierOutputSoftmaxTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return torch.softmax(model_output, dim=-1)[self.category]
        return torch.softmax(model_output, dim=-1)[:, self.category]

class ActivationsAndGradients:
    """ 모델의 활성화와 그래디언트를 추적하는 클래스 """
    def __init__(self, model, target_layers):
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations.append(output.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        def _store_grad(grad):
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

class AudioGradCAM:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        target_class: int = 0, # target_class 추가
        compute_input_gradient: bool = False,
        uses_gradients: bool = True,
    ):
        self.model = model
        self.target_layers = target_layers
        self.target_class = target_class # target_class 추가
        self.device = next(model.parameters()).device
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers)
        self.waveform = None
        self.sr = None

    def get_cam_weights(self, input_tensor: torch.Tensor, target_layer: torch.nn.Module,
                        targets: List[torch.nn.Module], activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        print("grads.shape",grads.shape) # 4초 기준 (1,199,128)
        print("grad : ", grads)
        return np.mean(grads, axis=1) # 4초 기준 (1,1,128) linear
        # return np.mean(grads, axis=2) #CNN layer

    def get_cam_audio(self, input_tensor: torch.Tensor, target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module], activations: torch.Tensor,
                      grads: torch.Tensor) -> np.ndarray:
        print("target_layer : ", target_layer)
        print("activations.shape",activations.shape) # 4초 기준 (1,199,128)
        print("activations : ", activations)
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        print("weights.shape befor weights[:, None, :]",weights.shape) # 4초 기준 (1,128)
        weights = weights[:, None, :] # linear layer
        # weights = weights[:,:,None] # CNN layer 
        print("weights.shape",weights.shape) # 4초 기준 (1,1,128)
        print("weights : ", weights)
        weighted_activations = activations * weights
        print("weighted_activations.shape",weighted_activations.shape) # 4초 기준 (1,199,128)
        print("weighted_activations : ", weighted_activations)
        cam = weighted_activations.sum(axis=2) # linear layer
        # cam = weighted_activations.sum(axis=1) # CNN layer
        print("cam.shape",cam.shape) # 4초 기준 (1,199)
        print("cam : ", cam)
        return cam

    def compute_cam_per_layer(self, input_tensor: torch.Tensor,
                              targets: List[torch.nn.Module]) -> np.ndarray:
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        
        cam_per_target_layer = []
        for target_layer, layer_activations, layer_grads in zip(self.target_layers, activations_list, grads_list):
            cam = self.get_cam_audio(input_tensor, target_layer, targets, layer_activations, layer_grads)
            # cam = np.maximum(cam, 0) # Relu
            self.visualize_cam(cam)
            cam_per_target_layer.append(cam)
        return np.array(cam_per_target_layer).mean(axis=0)

    def forward(self, input_tensor: torch.Tensor,
                targets: Optional[List[torch.nn.Module]] = None) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)
        self.waveform = input_tensor.cpu().numpy().squeeze()
        self.sr = 16000
        if self.compute_input_gradient:
            input_tensor.requires_grad_()
            
        outputs = self.activations_and_grads(input_tensor)
        # if targets is None:
        #     target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
        #     print("target_categories : ", target_categories)
        #     targets = [ClassifierOutputSoftmaxTarget(category) for category in target_categories]
        #     print("targets : ", targets)
        
        # target_class 추가
        # 항상 지정된 타겟 클래스(spoof)를 사용
        target_categories = [self.target_class]
        print("target_categories : ", target_categories)
        targets = [ClassifierOutputSoftmaxTarget(category) for category in target_categories]
        print("targets : ", targets)
        
        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # cam_per_layer = self.compute_cam_per_layer(input_tensor, targets)
        # 마지막 타겟 레이어의 활성화와 그래디언트를 가져옵니다.
        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()

        # get_cam_audio를 직접 호출하여 CAM을 계산합니다.
        cam = self.get_cam_audio(input_tensor, self.target_layers[-1], targets, activations, grads)

        # ReLU를 적용하여 음수 값을 제거합니다 (선택적).
        # cam = np.maximum(cam, 0)
        # print("cam_relu : ", cam)
        # CAM을 시각화합니다 (선택적).

        normalized_cam = self.normalize_cam(cam)
        self.visualize_cam(cam, normalized_cam)
        
        return normalized_cam
    def normalize_cam(self, cam):
        positive_mask = cam > 0
        negative_mask = cam < 0
        
        normalized_cam = np.zeros_like(cam)
        
        if np.any(positive_mask):
            pos_max = cam[positive_mask].max()
            normalized_cam[positive_mask] = cam[positive_mask] / pos_max
        
        if np.any(negative_mask):
            neg_min = cam[negative_mask].min()
            normalized_cam[negative_mask] = -cam[negative_mask] / neg_min
        
        return normalized_cam
    # def normalize_cam(self, cam):
        
    #     cam_min, cam_max = cam.min(), cam.max()
        
    #     # 음수 값 처리
    #     if cam_max <= 0:
    #         normalized_cam = (cam + abs(cam_min)) / (abs(cam_min) - cam_min)

    #     else:
    #     # 양수 값이 존재하는 경우
    #         positive_mask = cam > 0
    #         negative_mask = cam <= 0
            
    #         normalized_cam = np.zeros_like(cam)
            
    #         # 양수 값 정규화 (min-max 정규화)
    #         if np.any(positive_mask):
    #             positive_min, positive_max = cam[positive_mask].min(), cam[positive_mask].max()
    #             normalized_cam[positive_mask] = (cam[positive_mask] - positive_min) / (positive_max - positive_min)
            
    #         # 음수 값 정규화 (첫 번째 방식과 유사한 정규화)
    #         if np.any(negative_mask):
    #             negative_min = cam[negative_mask].min()
    #             normalized_cam[negative_mask] = (cam[negative_mask] + abs(negative_min)) / (abs(negative_min) - negative_min)
    #     return normalized_cam
    
    # 3번째 시각화
    # def visualize_cam(self, cam, normalized_cam):
    #     if self.waveform is None or self.sr is None:
    #         raise ValueError("Waveform and sample rate are not set. Please run forward() first.")

    #     # Get Grad-CAM values
    #     grad_cam_values = cam.squeeze()
    #     normalized_grad_cam_values = normalized_cam.squeeze()

    #     # Calculate frame times
    #     num_frames = len(grad_cam_values)
    #     duration = len(self.waveform) / self.sr
    #     frame_times = np.linspace(0, duration, num_frames)

    #     # Create the figure
    #     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

    #     # Plot the waveform
    #     librosa.display.waveshow(self.waveform, sr=self.sr, ax=ax1)
    #     ax1.set_title('Waveform')
    #     ax1.set_xlabel('Time (s)')
    #     ax1.set_ylabel('Amplitude')

    #     # Plot Grad-CAM heatmap
    #     vmax = np.max(np.abs(grad_cam_values))
    #     norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    #     im = ax2.imshow(grad_cam_values[np.newaxis, :], cmap='RdBu_r', norm=norm, 
    #                     aspect='auto', extent=[0, duration, 0, 1])
    #     ax2.set_title('Grad-CAM Heatmap')
    #     ax2.set_xlabel('Time (s)')
    #     ax2.set_ylabel('Channels')
    #     plt.colorbar(im, ax=ax2, orientation='vertical', label='Grad-CAM Score')

    #     # Plot normalized Grad-CAM scores as a bar chart with flipped negative values
    #     colors = ['blue' if v < 0 else 'red' for v in grad_cam_values]
    #     flipped_normalized_values = np.where(grad_cam_values < 0, -normalized_grad_cam_values, normalized_grad_cam_values)
    #     ax3.bar(frame_times, flipped_normalized_values, width=duration/num_frames, align='edge', color=colors, alpha=0.7)
    #     ax3.set_title('Normalized Grad-CAM Scores (Negative Values Flipped)')
    #     ax3.set_xlabel('Time (s)')
    #     ax3.set_ylabel('Score')
    #     ax3.set_ylim(0, 1)  # y축 범위를 -1에서 1로 설정
    #     ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5)  # 0 기준선 추가

    #     # Adjust layout and display
    #     plt.tight_layout()
    #     plt.show()

    #     # Display audio player
    #     audio_widget = Audio(self.waveform, rate=self.sr)
    #     display(audio_widget)
    
    def visualize_cam(self, cam, normalized_cam):
        if self.waveform is None or self.sr is None:
            raise ValueError("Waveform and sample rate are not set. Please run forward() first.")

        # Get Grad-CAM values
        grad_cam_values = cam.squeeze()
        normalized_grad_cam_values = normalized_cam.squeeze()

        # Calculate frame times
        num_frames = len(grad_cam_values)
        duration = len(self.waveform) / self.sr
        frame_times = np.linspace(0, duration, num_frames)

        # Create the figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

        # Plot the waveform
        librosa.display.waveshow(self.waveform, sr=self.sr, ax=ax1)
        ax1.set_title('Waveform')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')

        # Plot Grad-CAM heatmap
        vmax = np.max(np.abs(grad_cam_values))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax2.imshow(grad_cam_values[np.newaxis, :], cmap='RdBu_r', norm=norm, 
                        aspect='auto', extent=[0, duration, 0, 1])
        ax2.set_title('Grad-CAM Heatmap')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Channels')
        plt.colorbar(im, ax=ax2, orientation='vertical', label='Grad-CAM Score')

        # Plot normalized Grad-CAM scores as a bar chart with colors based on original values
        colors = ['blue' if v < 0 else 'red' for v in cam.squeeze()]
        ax3.bar(frame_times, normalized_cam.squeeze(), width=duration/num_frames, align='edge', color=colors, alpha=0.7)
        ax3.set_title('Normalized Grad-CAM Scores')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Score')
        ax3.set_ylim(-1, 1)
        ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5)  # 0 기준선 추가
        # colors = ['blue' if v < 0 else 'red' for v in grad_cam_values]
        # ax3.bar(frame_times, normalized_grad_cam_values, width=duration/num_frames, align='edge', color=colors, alpha=0.7)
        # ax3.set_title('Normalized Grad-CAM Scores')
        # ax3.set_xlabel('Time (s)')
        # ax3.set_ylabel('Score')
        # ax3.set_ylim(0, 1)

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

        # Display audio player
        audio_widget = Audio(self.waveform, rate=self.sr)
        display(audio_widget)
    # def visualize_cam(self, cam):
    #     plt.figure(figsize=(10, 2))
        
    #     # 데이터의 절대값 최대치를 찾아 대칭적인 색상 스케일 생성
    #     vmax = np.max(np.abs(cam))
    #     norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    #     # 히트맵 그리기
    #     im = plt.imshow(cam, cmap='RdBu_r', norm=norm, interpolation='nearest', aspect='auto')
        
    #     # 컬러바 추가
    #     cbar = plt.colorbar(im)
    #     cbar.set_label('Grad-CAM Score')
        
    #     plt.title('Grad-CAM Heatmap')
    #     plt.xlabel('Time Steps')
    #     plt.ylabel('Channels')
    #     plt.tight_layout()
    #     plt.show()

    def __call__(self, input_tensor: torch.Tensor,
                 targets: Optional[List[torch.nn.Module]] = None) -> np.ndarray:
        # return self.forward(input_tensor, targets, target_class)
        return self.forward(input_tensor, targets)


# def visualize_cam(cam, audio_path):
#     # Load the audio file
#     waveform, sr = librosa.load(audio_path, sr=None)

#     # Get Grad-CAM values
#     grad_cam_values = cam.squeeze()  # Remove any extra dimensions

#     # Calculate frame times
#     num_frames = len(grad_cam_values)
#     duration = len(waveform) / sr
#     frame_times = np.linspace(0, duration, num_frames)

#     # Create the figure
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

#     # Plot the waveform
#     librosa.display.waveshow(waveform, sr=sr, ax=ax1)
#     ax1.set_title('Waveform')
#     ax1.set_xlabel('Time (s)')
#     ax1.set_ylabel('Amplitude')

#     # Plot Grad-CAM scores as a bar chart
#     ax2.bar(frame_times, grad_cam_values, width=duration/num_frames, align='edge', color='purple', alpha=0.7)
#     ax2.set_title('Grad-CAM Scores')
#     ax2.set_xlabel('Time (s)')
#     ax2.set_ylabel('Score')
#     ax2.set_ylim(0, grad_cam_values.max())  # Set y-axis limit to max Grad-CAM score

#     # Adjust layout and display
#     plt.tight_layout()
#     plt.show()

#     # Display audio player
#     audio_widget = Audio(waveform, rate=sr)
#     display(audio_widget)
def visualize_cam(cam, audio_path):
    # Load the audio file
    waveform, sr = librosa.load(audio_path, sr=None)

    # Get Grad-CAM values
    grad_cam_values = cam.squeeze()  # Remove any extra dimensions

    # Calculate frame times
    num_frames = len(grad_cam_values)
    duration = len(waveform) / sr
    frame_times = np.linspace(0, duration, num_frames)

    # Create the figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

    # Plot the waveform
    librosa.display.waveshow(waveform, sr=sr, ax=ax1)
    ax1.set_title('Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')

    # Plot Grad-CAM heatmap
    vmax = np.max(np.abs(grad_cam_values))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax2.imshow(grad_cam_values[np.newaxis, :], cmap='RdBu_r', norm=norm, 
                    aspect='auto', extent=[0, duration, 0, 1])
    ax2.set_title('Grad-CAM Heatmap')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('')
    plt.colorbar(im, ax=ax2, orientation='vertical', label='Grad-CAM Score')

    # Plot Grad-CAM scores as a bar chart
    ax3.bar(frame_times, grad_cam_values, width=duration/num_frames, align='edge', color='purple', alpha=0.7)
    ax3.set_title('Grad-CAM Scores')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Score')
    ax3.set_ylim(grad_cam_values.min(), grad_cam_values.max())  # Set y-axis limit to min and max Grad-CAM score

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Display audio player
    audio_widget = Audio(waveform, rate=sr)
    display(audio_widget)