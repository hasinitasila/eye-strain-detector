class BlinkDetector:
    def __init__(self, threshold=0.22, drop_threshold=0.03, min_frames=2, max_frames=6):
        self.threshold = threshold
        self.drop_threshold = drop_threshold
        self.min_frames = min_frames
        self.max_frames = max_frames

        self.prev_ear = None
        self.blink_count = 0

        self.closed_frames = 0
        self.in_blink = False

    def update(self, ear):
        if self.prev_ear is None:
            self.prev_ear = ear
            return self.blink_count

        ear_drop = self.prev_ear - ear

        # Start blink
        if ear < self.threshold and ear_drop > self.drop_threshold and not self.in_blink:
            self.in_blink = True
            self.closed_frames = 1

        elif self.in_blink:
            if ear < self.threshold:
                self.closed_frames += 1
            else:
                # Check valid blink duration
                if self.min_frames <= self.closed_frames <= self.max_frames:
                    self.blink_count += 1

                self.in_blink = False
                self.closed_frames = 0

        self.prev_ear = ear
        return self.blink_count