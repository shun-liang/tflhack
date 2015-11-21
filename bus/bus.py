import pandas as pd
import numpy as np
import datetime as dt

def filter_nan_entries(df):
    return df.iloc[np.all(df.notnull(), axis=1)]

def get_journeys(df):
    journeys = np.split(df, np.where(np.diff(df["STOPSEQUENCE"])<= 0)[0]+1)

    return journeys


def get_all_stop_transitions(df):
    journeys = get_journeys(df)
    transitions = set()
    for journey in journeys:
        transitions.update(list(zip(journey.iloc[:-1]["BUS_STOP_CODE"], journey.iloc[1:]["BUS_STOP_CODE"])))

    return transitions


def collade_transition_times(df, dt1, dt2, transition):
    start = transition[0]
    stop = transition[1]
    locs = np.where(np.all(np.asarray([df.iloc[:-1]["BUS_STOP_CODE"] == start,
                                  np.asarray(df.iloc[1:]["BUS_STOP_CODE"] == stop)]), axis=0))
    filtered_dt1 = dt1[locs]
    filtered_dt2 = dt2[locs]
    filtered_dept_time = df.iloc[locs]["OBSERVED_DEPARTURE_TIME"]
    filtered_dist = df.iloc[locs]["SCHEDULEDDISTANCE"]


    times = (filtered_dt1 - filtered_dt2).astype("timedelta64[ms]").astype(int) / 1000.0


    return list(zip(filtered_dept_time, filtered_dist / times, filtered_dist, times))

def compute_travel_times(df):
    dt1 =  pd.to_datetime(df['OBSERVED_DEPARTURE_TIME']).dt.to_pydatetime()
    dt2 =  pd.to_datetime(df['OBSERVED_ARRIVAL_TIME']).dt.to_pydatetime()

    return dt1, dt2


def calculate_stop_transition_time(df):
    df = filter_nan_entries(df)
    transitions = get_all_stop_transitions(df)

    dt1, dt2 = compute_travel_times(df)

    times = {}
    for transition in transitions:
        times[transition] = collade_transition_times(df, dt1, dt2, transition)

    return times


def get_bus_stops(df):
    return set(df["BUS_STOP_CODE"])


def get_bus_stop_occupation_matrix(bs):
    return np.zeros((len(bs), (60 * 24 * 31)))


def split_date(date_str):
    days, time = date_str.split(" ")
    d, mth, y = [int(x) for x in days.split("/")]

    h, m, s = [int(x) for x in time.split(":")]

    return d, mth, y, h, m, s



def extract_day_from_timestamp(date_str):
    days, time = date_str.split(" ")

    return days


def get_time_map(df):
    d, mth, y, h, m, s = split_date(df.iloc[0]["OBSERVED_DEPARTURE_TIME"])
    start_day = pd.Timestamp("{}-{}-{} 00:00:00".format(y, mth, d, h, m))
    d, mth, y, h, m, s = split_date(df.iloc[-1]["OBSERVED_DEPARTURE_TIME"])
    stop_day = pd.Timestamp("{}-{}-{} 23:59:00".format(y, mth, d, h, m))


    time_map = pd.date_range(start_day,
                             stop_day, freq='min').tolist()


    return dict(zip(time_map, range(len(time_map))))

# def matrix_idx(d, h, m, s):
#     idx = (60 * 24) * (d-1) + 60 * h + m

#     if np.isnan(idx):
#         1/0

#     return idx

def matrix_idx(time_map, date_str):
    d, mth, y, h, m, s = split_date(date_str)
    floored_date_time = pd.Timestamp("{}-{}-{} {}:{}:00".format(y, mth, d, h, m))
    return time_map[pd.Timestamp(floored_date_time)]



def calculate_bus_stop_occupation(df):
    journeys = get_journeys(df)
    bs_1 = journeys[0]["BUS_STOP_CODE"].tolist()
    bs_2 = journeys[-1]["BUS_STOP_CODE"].tolist()

    df = filter_nan_entries(df)
    time_map = get_time_map(df)

    mat = get_bus_stop_occupation_matrix(bs_1 + bs_2)

#     starts = np.asarray(df.apply(lambda x: matrix_idx(*split_date(x["OBSERVED_ARRIVAL_TIME"])), axis=1))
#     stops = np.asarray(df.apply(lambda x: matrix_idx(*split_date(x["OBSERVED_DEPARTURE_TIME"])), axis=1))

    starts = np.asarray(df.apply(lambda x: matrix_idx(time_map, x["OBSERVED_ARRIVAL_TIME"]), axis=1))
    stops = np.asarray(df.apply(lambda x: matrix_idx(time_map, x["OBSERVED_DEPARTURE_TIME"]), axis=1))

    bs_direction = df["DIRECTION"] == 1

    for i, bus_stop in enumerate(bs_1):
        df_idc = np.where((df["BUS_STOP_CODE"] == bus_stop) & bs_direction)[0]
        start = starts[df_idc]
        stop = stops[df_idc]

        for srt, stp in zip(start, stop):
            mat[i, slice(srt, stp)] += 1

    for i, bus_stop in enumerate(bs_2):
        df_idc = np.where((df["BUS_STOP_CODE"] == bus_stop) & ~bs_direction)[0]
        start = starts[df_idc]
        stop = stops[df_idc]

        for srt, stp in zip(start, stop):
            mat[i + len(bs_1), slice(srt, stp)] += 1

    return mat, time_map



def get_index_slice(time_map, arrival_time, dept_time):
    start_idx = pd.Timestamp(arrival_time)
    stop_idx = pd.Timestamp(dept_time)
    return slice(time_map[start_idx], time_map[stop_idx])


def get_heat_map(time_map, mat, start_dt, stop_dt):
    idx_slice = get_index_slice(time_map, start_dt, stop_dt)
    return np.sum(mat[:, idx_slice], axis=1)
