import numpy as np
class Kalman():
    def __init__(self, initial_uncertainty, process_variance, house_error=0, initial_state=None):
        self.state = initial_state
        self.uncertainty = initial_uncertainty #P0
        self.Q = process_variance #Q
        self.house_error = house_error #Allows to add an extra variance term for poll noise, to estimate variance beyond pure samplign variance

        self.t = 0



    def update(self, measurement, t, weight, n=1000):
        if np.isnan(measurement):
            return self.state #With NA input do not update, just return the old state
        assert measurement >= 0 and measurement <= 1, f"Value {measurement} is not valid"

        poll_noise = (measurement * (1 - measurement))/n + self.house_error ** 2#R

        poll_noise = poll_noise/(weight ** 2) #add weighing of pollsters

        time_passed = t - self.t
        self.t = t

        #Prediction step
        if self.state is not None:
            state_prediction = self.state #predicted state is old state
        else:
            state_prediction = measurement
        predicted_variance = self.uncertainty + self.Q * time_passed #scale the predicted variance by how much time has passed

        #Update step
        innovation = measurement - state_prediction
        innovation_variance = predicted_variance + poll_noise
        kalman_gain = predicted_variance / innovation_variance


        self.state = state_prediction + kalman_gain * innovation
        self.state = min(1, max(0, self.state))

        self.uncertainty = (1.0 - kalman_gain) * predicted_variance



        return self.state
