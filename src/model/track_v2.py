class Track:
    def __init__(self, track_view:int) -> None:
        self.track_map = []

        track_position = 0
        curve_velocity = 0
        curve_acceleration = 0.01  # -0.01 for left curve and 0.01 for right curve
        curve_angle_increment = 0 # 0.0001
        for _ in range(track_view):
            curve_velocity += curve_acceleration
            track_position += curve_velocity
            curve_acceleration += curve_angle_increment
            self.track_map.append(track_position)
            # print(f"track position: {track_position}")

    def __getitem__(self, key) -> float:
        return self.track_map[key]

    def __len__(self) -> int:
        return len(self.track_map)
    
    def __min__(self) -> float:
        return min(self.track_map)
    
    def __max__(self) -> float:
        return max(self.track_map)
