import av, gc
import paddle
import warnings
import numpy as np


_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 20


# remove warnings
av.logging.set_level(av.logging.ERROR)


class VideoReader:
    """
    Paddle adaptation of a simple wrapper around PyAV for video reading.
    """
    def __init__(self, video, num_frames=float("inf"), decode_lossy=False, audio_resample_rate=None, bi_frame=False):
        self.container = av.open(video)
        self.num_frames = num_frames
        self.bi_frame = bi_frame

        self.resampler = None
        if audio_resample_rate is not None:
            self.resampler = av.AudioResampler(rate=audio_resample_rate)

        if self.container.streams.video:
            if decode_lossy:
                warnings.warn('PaddleVideoReader| thread_type==AUTO can yield potential frame dropping!', RuntimeWarning)
                self.container.streams.video[0].thread_type = 'AUTO'
            self.video_stream = self.container.streams.video[0]
        else:
            self.video_stream = None

        self.fps = self._get_video_frame_rate()

    def seek(self, pts, backward=True, any_frame=False):
        stream = self.video_stream
        self.container.seek(pts, any_frame=any_frame, backward=backward, stream=stream)

    def _occasional_gc(self):
        global _CALLED_TIMES, _GC_COLLECTION_INTERVAL
        _CALLED_TIMES += 1
        if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == _GC_COLLECTION_INTERVAL - 1:
            gc.collect()

    def _read_video(self, offset):
        self._occasional_gc()

        pts = self.container.duration * offset
        time_ = pts / float(av.time_base)
        self.container.seek(int(pts))

        video_frames = []
        count = 0
        for _, frame in enumerate(self._iter_frames()):
            if frame.pts * frame.time_base >= time_:
                video_frames.append(frame)
                if count >= self.num_frames - 1:
                    break
                count += 1
        return video_frames

    def _iter_frames(self):
        for packet in self.container.demux(self.video_stream):
            for frame in packet.decode():
                yield frame

    def _compute_video_stats(self):
        if self.video_stream is None or self.container is None:
            return 0
        num_of_frames = self.container.streams.video[0].frames
        if num_of_frames == 0:
            num_of_frames = self.fps * float(self.container.streams.video[0].duration * self.video_stream.time_base)
        self.seek(0, backward=False)
        count = 0
        time_base = 512
        for p in self.container.decode(video=0):
            count += 1
            if count == 1:
                start_pts = p.pts
            elif count == 2:
                time_base = p.pts - start_pts
                break
        return start_pts, time_base, num_of_frames

    def _get_video_frame_rate(self):
        return float(self.container.streams.video[0].guessed_rate)

    def sample(self, debug=False):
        if self.container is None:
            raise RuntimeError('video stream not found')
        sample = dict()
        _, _, total_num_frames = self._compute_video_stats()
        offset = np.random.randint(max(1, total_num_frames - self.num_frames - 1))
        video_frames = self._read_video(offset / total_num_frames)
        video_frames = np.array([np.uint8(f.to_rgb().to_ndarray()) for f in video_frames])
        sample["frames"] = video_frames
        sample["frame_idx"] = [offset]

        if self.bi_frame:
            frames = [np.random.beta(2, 1, size=1), np.random.beta(1, 2, size=1)]
            frames = [int(frames[0] * self.num_frames), int(frames[1] * self.num_frames)]
            frames.sort()
            video_frames = np.array([video_frames[min(frames)], video_frames[max(frames)]])
            Ts = [min(frames) / (self.num_frames - 1), max(frames) / (self.num_frames - 1)]
            sample["frames"] = video_frames
            sample["real_t"] = paddle.to_tensor(Ts, dtype=paddle.float32)
            sample["frame_idx"] = [offset + min(frames), offset + max(frames)]
            return sample

        return sample

    def read_frames(self, frame_indices):
        self.num_frames = frame_indices[1] - frame_indices[0]
        video_frames = self._read_video(frame_indices[0] / self.get_num_frames())
        video_frames = np.array([
            np.uint8(video_frames[0].to_rgb().to_ndarray()),
            np.uint8(video_frames[-1].to_rgb().to_ndarray())
        ])
        return video_frames

    def read(self):
        video_frames = self._read_video(0)
        video_frames = np.array([np.uint8(f.to_rgb().to_ndarray()) for f in video_frames])
        return video_frames

    def get_num_frames(self):
        _, _, total_num_frames = self._compute_video_stats()
        return total_num_frames