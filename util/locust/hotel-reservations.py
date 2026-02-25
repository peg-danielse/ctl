import os
import glob
import math
import time

import datetime

import numpy as np
import matplotlib.pyplot as plt 

from scipy.stats import weibull_min

from random import randint, choice

from locust import HttpUser, task, between, LoadTestShape, tag
from locust import events
from locust.runners import MasterRunner

from weibull import get_n_weibull_variables, compute_weibull_scale

# Tag names used by Locust --tags (e.g. from main.py loadtest()). Only tasks with these tags run when --tags is passed.
TASK_TAGS = ("search_hotel", "recommend", "reserve", "user_login")

log_file=""

experiment_label = "unknown"

@events.test_stop.add_listener
def on_test_stop(environment, **_kwargs):
    if not isinstance(environment.runner, MasterRunner):
        return

    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(seconds=environment.parsed_options.run_time)

    print("start", start, "end", end)

    label = environment.parsed_options.csv_prefix
    
    # Consolidate response time logs
    os.makedirs(f"./output/{label}/data", exist_ok=True)
    total_file = f"./output/{label}/data/locust_responce_log.csv"
    csv_log_files = glob.glob(f"./output/tmp/.response_times_*.csv")
    
    with open(total_file, "w") as f:
        f.write("request_type,name,response_time,response_length,status_code\n")
        for log_name in csv_log_files:
            with open(log_name, 'r') as in_f:
                f.write(in_f.read())
    return

@events.quitting.add_listener
def on_quit(environment):
    if isinstance(environment.runner, MasterRunner):
        return
    time.sleep(10)

    global log_file
    os.remove(log_file)

@events.test_start.add_listener
def on_start(**_kwargs):
    global log_file
    worker_id = os.getpid()  # or use uuid.uuid4() for uniqueness
    
    os.makedirs(f"./output/tmp", exist_ok=True)
    log_file = f"./output/tmp/.response_times_{worker_id}.csv"
    print(f"log_file: {log_file}")

@events.request.add_listener
def log_request(request_type, name, response_time, response_length, response, **kwargs):
    global log_file
    try:
        with open(log_file, "a") as f:
            f.write(f"{request_type},{name},{response_time},{response_length}, {response.status_code}\n")
    except Exception:
        pass  # avoid request handler exceptions blocking the runner

class HotelUser(HttpUser):
    wait_time = between(2,4)

    @tag("search_hotel")
    @task(60)
    def search_hotel(self):
        in_date = randint(9, 23)
        out_date = randint(in_date + 1, 30) 
        lat, lon = HotelUser.get_lat_lon()
        
        self.client.get(f"/hotels?inDate=2015-04-{in_date:02}&outDate=2015-04-{out_date:02}&lat={lat}&lon={lon}", 
                        name="/hotels", timeout=10) # ,headers={"Content-Type":"application/x-www-form-urlencoded"})

    @tag("recommend")
    @task(38)
    def recommend(self):
        require = choice(["dis", "rate", "price"])
        lat, lon = HotelUser.get_lat_lon()
        self.client.get(f"/recommendations?require={require}&lat={lat}&lon={lon}", 
                        name="/recommendations", timeout=10) # ,headers={"Content-Type":"application/x-www-form-urlencoded"})

    @tag("reserve")
    @task(1)
    def reserve(self):
        lat, lon = HotelUser.get_lat_lon()
        in_date = randint(9, 23)
        out_date = in_date + randint(1, 5)
        hotel_id = str(randint(1, 80))
        username, password = HotelUser.get_user()
        num_room = "1"


        self.client.post(f"/reservation?inDate=2015-04-{in_date:02}&outDate=2015-04-{out_date:02}&lat={lat}&lon={lon}&hotelId={hotel_id}&customerName={username}&username={username}&password={password}&number={num_room}",
                        name="/reservation", timeout=10)

    @tag("user_login")
    @task(1)
    def user_login(self):
        username, password = HotelUser.get_user()
        self.client.post(f"/user?username={username}&password={password}",
                        name="/user", timeout=10)

    @staticmethod
    def get_lat_lon():
        return 38.0235 + float(randint(0, 481) - 240.5)/1000.0, -122.095 + float(randint(0, 325) - 157.0)/1000.0

    @staticmethod
    def get_user():
        id = randint(0,500)
        return f"Cornell_{id}", ''.join(str(i) for i in range(0,9))


@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--seed", type=int, is_required=False, default=42)
    parser.add_argument("--w-shape", type=int,is_required=False, default=1)
    parser.add_argument("--w-mean", type=int, is_required=False, default=150)
    parser.add_argument("--w-user-min", type=int, is_required=False, default=100)
    parser.add_argument("--w-user-max", type=int, is_required=False, default=1000)
    parser.add_argument("--w-dt", type=int, is_required=False, default=20)
    parser.add_argument("--w-ls-y", type=int, is_required=False, default=500)
    parser.add_argument("--w-dry-run", type=int, is_required=False, default=False)    
    
class WeibullShape(LoadTestShape):
    stages = []
    use_common_options = True

    def plot(self, tmin, tmax, shape_k, scale_lambda, N, T, stages, offset, seed, label):
        # Plot histogram of samples
        plt.figure(figsize=(10, 6))
        bins = np.linspace(tmin, tmax, 100)
        plt.hist([int(e["load"]) - o for e, o in zip(stages, offset)], bins=bins, density=True, alpha=0.6, color='lightgreen', edgecolor='black', label="Truncated Weibull Samples")

        # Plot analytical truncated PDF
        x_vals = np.linspace(tmin, tmax, 500)
        pdf_vals = weibull_min.pdf(x_vals, c=shape_k, scale=scale_lambda)
        cdf_min = weibull_min.cdf(tmin, c=shape_k, scale=scale_lambda)
        cdf_max = weibull_min.cdf(tmax, c=shape_k, scale=scale_lambda)
        truncated_pdf = pdf_vals / (cdf_max - cdf_min)

        plt.plot(x_vals, truncated_pdf, 'r--', lw=2, label="Truncated Weibull PDF")
        plt.title("Truncated Weibull Distribution Samples")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)

        os.makedirs(f"./output/{label}/plots", exist_ok=True)
        plt.savefig(f"./output/{label}/plots/user_count_distribution_{seed}_{label}.png")
        plt.clf()

        # plot users per second
        print(0, len(stages), [int(e["load"]) for e in stages])

        load = [int(e["load"]) for e in stages]
        print(load)

        x = np.linspace(0, T,len(stages))
        plt.plot(x, load)
        plt.title("Truncated Weibull Load generation in Users per Second.")
        plt.xlabel("time [S]")
        plt.ylabel("users [#]")
        plt.grid(True)

        plt.savefig(f"./output/{label}/plots/users_over_time_{seed}_{label}.png")
        plt.clf()
   

    def tick(self):
        # build stages on the first tick.
        if self.stages == []:
            label = self.runner.environment.parsed_options.csv_prefix
            seed = self.runner.environment.parsed_options.seed
            
            w_shape  = self.runner.environment.parsed_options.w_shape
            w_mean  = self.runner.environment.parsed_options.w_mean
            U_min  = self.runner.environment.parsed_options.w_user_min
            U_max  = self.runner.environment.parsed_options.w_user_max
            t  = self.runner.environment.parsed_options.run_time
            dt  = self.runner.environment.parsed_options.w_dt

            ls_y  = self.runner.environment.parsed_options.w_ls_y
            

            ## magic-kit
            _lambda = compute_weibull_scale(w_mean, w_shape)
            N = int(t/dt)
            L = get_n_weibull_variables(w_shape, _lambda, U_min, U_max, N, seed)

            l_prev = 0

            # or logspace()
            offset = np.linspace(U_min, U_min + ls_y, N)

            for s, l, o in zip(range(dt,t+dt,dt), L, offset):
                self.stages.append({"start": s, "load": int(l + o), "rate": int(math.ceil(abs((l_prev - l)/dt)))})
                l_prev = l
            
            print(self.stages)

            # self.plot(U_min, U_max, w_shape, _lambda, N, t, self.stages, offset, seed,  label)
         
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["start"]:
                return (stage["load"], stage["rate"])

        # Past all stages: keep last stage's load until --run-time ends (avoid returning None and stalling)
        if self.stages:
            last = self.stages[-1]
            return (last["load"], last["rate"])
        return None
