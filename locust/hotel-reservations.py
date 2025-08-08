import os
import io
import glob
import math
import time
import json
import re

import requests
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from scipy.stats import weibull_min

from random import randint, choice

from locust import HttpUser, task, between, LoadTestShape
from locust import events
from locust.runners import MasterRunner

from weibull import get_n_weibull_variables, compute_weibull_scale

JAEGER_ENDPOINT_FSTRING = "http://145.100.135.11:30550/api/traces?limit={limit}&lookback={lookback}&service={service}&start={start}"
PROMETHEUS_BASE_URL = "http://145.100.135.11:31207"

PATH = "./locust/"
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
    total_file = f"./{label}_responce_log.csv"
    csv_log_files = glob.glob("./.response_times_*.csv")
    
    with open(total_file, "w") as f:
        f.write("request_type,name,response_time,response_length,status_code\n")
        for log_name in csv_log_files:
            with open(log_name, 'r') as in_f:
                f.write(in_f.read())


    url = PROMETHEUS_BASE_URL + '/api/v1/label/configuration_name/values'
    response = requests.get(url)


    data = response.json()
    print(data)
    
    total_m_df = None
    for c in data['data']:
        url = PROMETHEUS_BASE_URL + '/api/v1/label/revision_name/values'
        
        # get the names
        params = {'match[]': f'autoscaler_desired_pods{{namespace_name="default",configuration_name="{c}"}}'}

        response = requests.get(url, params=params)

        revision = response.json()
        print("Revision name:", revision['data'])

        url = PROMETHEUS_BASE_URL + '/api/v1/query_range'

        # Autoscaler metrics
        query = ['sum(autoscaler_requested_pods{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
                'sum(autoscaler_terminating_pods{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
                'sum(autoscaler_actual_pods{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
                'sum(activator_request_concurrency{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
                'sum(autoscaler_desired_pods{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
                'sum(autoscaler_stable_request_concurrency{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
                'sum(autoscaler_panic_request_concurrency{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
                'sum(autoscaler_target_concurrency_per_pod{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
                'sum(autoscaler_excess_burst_capacity{{namespace_name="default", configuration_name="{config}", revision_name="{revision}"}})',
                'sum(rate(container_cpu_usage_seconds_total{{namespace="default", pod=~"{revision}.*", container != "POD", container != ""}}[1m])) by (container)']
    
        metric_df = None
        for q in query:
            fvalues = {"config": c, "revision": revision['data'][-1]}
            params = {'query': q.format_map(fvalues),
                    'start': start.isoformat() + 'Z',
                    'end': end.isoformat() + 'Z',
                    'step': '5s' } # str(math.ceil(int(end.timestamp()) - int(start.timestamp())) / 1000) } # from ceil((end - start) / 1000) 

            response = requests.get(url, params=params)
            result = response.json()

            print(result)

            match = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\{\{', q)
            metric_name = q
            
            if match:
                metric_name = match.group(1)
            
            # if the data collection gets no values for a revision.
            try:
                result_data = {"index":[], f"{c}_{metric_name}":[]}

                print(result)

                for e in result['data']['result'][0]["values"]:
                    result_data['index'].append(int(e[0]))
                    result_data[f"{c}_{metric_name}"].append(float(e[1]))
            except Exception as e:
                print(e)
                continue

            result_df = pd.DataFrame(result_data)
            
            if total_m_df is None:
                metric_df = result_df
                total_m_df = result_df
                continue

            if metric_df is None:
                metric_df = result_df
                continue 

            metric_df = pd.merge(metric_df, result_df, on='index', how='outer')
            total_m_df = pd.merge(metric_df, result_df, on='index', how='outer')
        
        metric_df.to_csv(f"{label}_{c}_metrics.csv")
    total_m_df.to_csv(f"{label}_total_metrics.csv")


    # get the traces from jaeger save and rename file.
    url = JAEGER_ENDPOINT_FSTRING.format(limit=str(4000), 
                                            lookback=str(environment.parsed_options.run_time) , 
                                            service="frontend", 
                                            start=int((time.mktime(time.localtime()) - environment.parsed_options.run_time) * 1_000_000))

    print(url)

    data = requests.get(url).json()

    with io.open(f'./{label}_traces.json', 'w', encoding='utf8') as outfile:
        json.dump(data, outfile,
                indent=4, sort_keys=True,
                separators=(',', ': '), ensure_ascii=False)

    
    return

@events.quitting.add_listener
def on_quit(environment):
    if isinstance(environment.runner, MasterRunner):
        return
    
    global log_file
    os.remove(log_file)


@events.test_start.add_listener
def on_test_start(environment, **_kwargs):
    global experiment_label
    experiment_label = environment.parsed_options.csv_prefix

    global log_file
    worker_id = os.getpid()  # or use uuid.uuid4() for uniqueness
    log_file = f"./.response_times_{worker_id}.csv"

@events.request.add_listener
def log_request(request_type, name, response_time, response_length, response, **kwargs):
    global log_file

    with open(log_file, "a") as f:
        f.write(f"{request_type},{name},{response_time},{response_length}, {response.status_code}\n")


class HotelUser(HttpUser):
    wait_time = between(3,5)

    @task(60)
    def search_hotel(self):
        in_date = randint(9, 23)
        out_date = randint(in_date + 1, 30) 
        lat, lon = HotelUser.get_lat_lon()
        
        self.client.get(f"/hotels?inDate=2015-04-{in_date:02}&outDate=2015-04-{out_date:02}&lat={lat}&lon={lon}", 
                        name="/hotels") # ,headers={"Content-Type":"application/x-www-form-urlencoded"})

    @task(38)
    def recommend(self):
        require = choice(["dis", "rate", "price"])
        lat, lon = HotelUser.get_lat_lon()
        self.client.get(f"/recommendations?require={require}&lat={lat}&lon={lon}", 
                        name="/recommendations") # ,headers={"Content-Type":"application/x-www-form-urlencoded"})

    @task(1)
    def reserve(self):
        lat, lon = HotelUser.get_lat_lon()
        in_date = randint(9, 23)
        out_date = in_date + randint(1, 5)
        hotel_id = str(randint(1, 80))
        username, password = HotelUser.get_user()
        num_room = "1"


        self.client.post(f"/reservation?inDate=2015-04-{in_date:02}&outDate=2015-04-{out_date:02}&lat={lat}&lon={lon}&hotelId={hotel_id}&customerName={username}&username={username}&password={password}&number={num_room}",
                        name="/reservation")

    @task(1)
    def user_login(self):
        username, password = HotelUser.get_user()
        self.client.post(f"/user?username={username}&password={password}",
                        name="/user")

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
    
    
class WeibullShape(LoadTestShape):
    stages = []
    use_common_options = True

    def plot(self, tmin, tmax, shape_k, scale_lambda, N, T, stages, offset, seed, label):
        # Plot histogram of samples
        plt.figure(figsize=(10, 6))
        bins = np.linspace(tmin, tmax, 100)
        print
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
        
        plt.savefig(PATH + f"user_count_distribution_{seed}_{label}.png")
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

        plt.savefig(PATH + f"users_over_time_{seed}_{label}.png")
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
            T  = self.runner.environment.parsed_options.run_time
            dt  = self.runner.environment.parsed_options.w_dt

            ls_y  = self.runner.environment.parsed_options.w_ls_y
            

            ## magic-kit
            _lambda = compute_weibull_scale(w_mean, w_shape)
            N = int(T/dt)
            L = get_n_weibull_variables(w_shape, _lambda, U_min, U_max, N, seed)

            l_prev = 0


            # or logspace()
            offset = np.linspace(U_min, ls_y, N)

            for s, l, o in zip(range(dt,T+dt,dt), L, offset):
                self.stages.append({"start": s, "load": int(l + o), "rate": int(math.ceil(abs((l_prev - l)/dt)))})
                l_prev = l
            
            print(self.stages)

            self.plot(U_min, U_max, w_shape, _lambda, N, T, self.stages, offset, label, seed)
         
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["start"]:
                return (stage["load"], stage["rate"])

        return None
