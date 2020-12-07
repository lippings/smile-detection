from time import time

from collections import deque
from numpy import mean
from numpy import round as np_round
from torchvision.transforms import Normalize, Compose, ToTensor, Lambda
from torch.cuda import is_available as cuda_is_available
from torch import load as load_state
from torch import no_grad, round
import cv2

from tools import read_yaml
from method_helpers import get_model_path, get_model


CLASS_DICT = {
    0: 'No smile',
    1: 'Smile'
}

CLASS_CLR = {
    0: (0, 0, 255),
    1: (0, 255, 0)
}


if __name__ == '__main__':
    config = read_yaml('config/main_settings.yaml')
    device = 'cuda' if cuda_is_available() and config['training']['device'] == 'gpu' else 'cpu'

    pre_process = Compose([
        Lambda(lambda im: cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model = get_model(config).float()

    model.load_state_dict(
        load_state(
            get_model_path(config)
        )
    )
    model.eval()
    model = model.to(device)

    sec_per_frame = 1. / config['demo']['max_fps']
    proc_times = deque(maxlen=config['demo']['fps_avg_nb'])

    # Init video capture
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()  # Get web cam dimensions
    h = frame.shape[0]
    w = frame.shape[1]

    with no_grad():
        while True:
            start_time = time()

            # Read frame
            ret, frame = cap.read()

            # Resize etc
            x = pre_process(frame).unsqueeze(0)
            x = x.to(device)

            # Make prediction
            pred_idx = int(round(model(x)).item())
            pred_class = CLASS_DICT[pred_idx]

            # Store elapsed time, evaluate fps
            proc_times.append(1. / (time()-start_time))
            fps_str = f'fps: {int(np_round(mean(proc_times)))}'

            # Display image with prediction
            im_to_show = cv2.putText(frame, pred_class, (20, h), cv2.FONT_HERSHEY_PLAIN, 5, CLASS_CLR[pred_idx], 2)
            im_to_show = cv2.putText(im_to_show, fps_str, (0, 12), cv2.FONT_HERSHEY_PLAIN,
                                     1, (0, 255, 0), 1)
            cv2.imshow(
                'Press Q to quit',
                im_to_show
            )

            time_left = sec_per_frame - (time() - start_time)
            time_left_ms = max(int(time_left * 1e-3), 1)  # Minimum of 1 ms wait time (0 means hold image)
            if cv2.waitKey(time_left_ms) & 0xFF == ord('q'):
                break

    # Close video capture
    cap.release()
    cv2.destroyAllWindows()
