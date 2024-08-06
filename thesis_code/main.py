import numpy as np
from fields import ITEM_FIELDS
from order import Order, global_order_list
from batch import Batch, init_item_orientation
from sim_ann import simulated_annealing_order_items, calculate_route_makespan
import random, math
from typing import List
import logging
from config import parser
from util import read_coordinates_file, read_orders_file
import time
import pandas as pd

origin_distance = []
distance_matrix = []
batches = list[Batch]() # batches = [] 

def batchToString(batch: Batch):
    s = f"Batch({batch.id}): 0, "
    for i, order_id in enumerate(batch.orders):
      r = batch.is_order_reversed[i]
      s = s + f" o{order_id}({global_order_list[order_id].route}){"'" if batch.is_order_reversed[i] else ""}[{calculate_route_makespan(global_order_list[order_id].route, distance_matrix)}] ,"
    s = s + " 0"
    s = s + f", Makespan: [{calculate_batch_makespan(batch.orders, batch.is_order_reversed)}]"
    return s

def calculate_batch_makespan(batch_orders, reverse=None):
    route = []
    if reverse is None:
       reverse = len(batch_orders) * [False]
    for i, order_id in enumerate(batch_orders):
        r = global_order_list[order_id].route
        if reverse[i]:
           r = r.copy()
           r.reverse()
        route.extend(r)
    return origin_distance[route[0]] + calculate_route_makespan(route, distance_matrix) + origin_distance[route[-1]]

def simulated_annealing_batch_orders(batch: Batch, initial_temperature_batch_orders, cooling_rate_batch_orders, L_batch_orders, K2_batch_orders, epsilon2_batch_orders):
    
    n = len(batch.orders)
    current_reverse = n * [False]
    
    if n == 1:
      return batch.orders
    current_batch_route = batch.orders.copy()
    current_batch_distance = calculate_batch_makespan(current_batch_route, current_reverse)
    
    best_batch_route = current_batch_route.copy()
    best_batch_distance = current_batch_distance
    best_batch_reverse = current_reverse.copy()

    temperature = initial_temperature_batch_orders
    k = 1  # Plateau counter
    terminate = False
    no_improvement_count = 0
    accepted_count = 0

    while not terminate:
        new_batch_route = current_batch_route.copy()
        new_reverse = current_reverse.copy()

        swap_or_reverse = np.random.rand() > 0.5
        if swap_or_reverse:
           # Swap
          i, j = random.sample(range(n), 2)
          logging.debug(f"Swapping {i}, {j}")
          new_batch_route[i], new_batch_route[j] = new_batch_route[j], new_batch_route[i]
          new_reverse[i], new_reverse[j] = new_reverse[j], new_reverse[i]
          new_batch_distance = calculate_batch_makespan(new_batch_route, new_reverse)
        else:
          # Reverse
          i = random.sample(range(n), 1)[0]
          logging.debug(f"Reversing {i}")
          new_reverse[i] = not new_reverse[i]
          new_batch_distance = calculate_batch_makespan(new_batch_route, new_reverse)

        logging.debug(f"Current: {current_batch_route}, [{calculate_batch_makespan(current_batch_route, current_reverse)}]")
        logging.debug(f"New: {new_batch_route}, [{calculate_batch_makespan(new_batch_route, new_reverse)}]")
        logging.debug(f"Best: {best_batch_route}, [{calculate_batch_makespan(best_batch_route, best_batch_reverse)}]")
        s = ""
        for i, order_id in enumerate(best_batch_route):
          s = s + f" o{order_id}({global_order_list[order_id].route}){"'" if best_batch_reverse[i] else ""} ,"
        logging.debug(s)
        logging.debug("")

        if new_batch_distance < current_batch_distance or random.random() < math.exp((current_batch_distance - new_batch_distance) / temperature):
            current_batch_route = new_batch_route
            current_batch_distance = new_batch_distance
            current_reverse = new_reverse
            accepted_count += 1


        if new_batch_distance < best_batch_distance:
            best_batch_route = new_batch_route
            best_batch_distance = new_batch_distance
            best_batch_reverse = new_reverse
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Evaluate stopping conditions
        if k < L_batch_orders:
            k += 1
            terminate = False
        elif no_improvement_count >= K2_batch_orders and (accepted_count / L_batch_orders) < epsilon2_batch_orders:
            terminate = True
        else:
            temperature *= cooling_rate_batch_orders  # Decrease temperature
            k += 1
            terminate = False

    batch.is_order_reversed = best_batch_reverse.copy()
    batch.orders = best_batch_route

def calculate_makespan(num_pickers: int, batch_list: List[Batch]):
  job_worker_pairs = [(f"Batch {b.id}", float(calculate_batch_makespan(b.orders, b.is_order_reversed))) for b in batch_list]

  # İş-işçi çiftlerini iş sürelerine göre azalan sırayla sıralama
  sorted_job_worker_pairs = sorted(job_worker_pairs, key=lambda x: x[1], reverse=True)

  # Makineler listesi (her biri başlangıçta 0 süreye sahip)
  pickers = [0] * num_pickers

  # Her iş için en az yük altındaki makineyi bulma ve iş-işçi çiftini o makineye atama
  job_allocation = [[] for _ in range(num_pickers)]

  for job_worker in sorted_job_worker_pairs:
      _, job_duration = job_worker
      # En az yük altındaki makineyi bul
      min_machine_index = pickers.index(min(pickers))
      # O makineye iş-işçi çiftini ata (süresini ekle)
      pickers[min_machine_index] += job_duration
      job_allocation[min_machine_index].append(job_worker)

  # Sonuçları yazdırma
  for i, machine_jobs in enumerate(job_allocation):
      machine_jobs_str = ', '.join([f"{worker} ({duration})" for worker, duration in machine_jobs])
      logging.debug(f"Makine {i+1}: {machine_jobs_str}, Toplam süre: {sum(duration for _, duration in machine_jobs)}")

  # Makespan'i yazdırma
  logging.debug(f"Toplam tamamlanma süresi (Makespan): {max(pickers)}")

  return job_allocation, max(pickers)

def assign_orders_to_batches(order_list, batch_list: List[Batch]):
    for order_id in order_list:
      assigned = False

      for b in batch_list:
        assigned = b.assign(global_order_list[order_id].items)
        if assigned:
            break

      if not assigned:
        b_new = Batch()
        b_new.assign(global_order_list[order_id].items)
        batch_list.append(b_new)


def run_sim_ann_foreach_batch(batch_list, initial_temperature_batch_orders, cooling_rate_batch_orders, L, K2, epsilon2):
    logging.debug("=== Before simulated annealing ===")
    logging.debug("=== Batches ===")
    for b in batch_list:
      logging.debug(batchToString(b))

    for b in batch_list:
      simulated_annealing_batch_orders(b, initial_temperature_batch_orders, cooling_rate_batch_orders, L, K2, epsilon2)

    logging.debug("=== After simulated annealing ===")
    logging.debug("=== Batches ===")
    for b in batch_list:
      logging.debug(batchToString(b))

def main(args):
  global batches, distance_matrix, origin_distance

  start_time = time.time()

  np.random.seed(seed=args.seed)

  orders_items_list = read_orders_file(args.orders_file)
  num_orders = max([order_id[ITEM_FIELDS.ORDER_ID] for order_id in orders_items_list])+1
  init_item_orientation(len(orders_items_list))

  distance_matrix, origin_distance = read_coordinates_file(args.coordinates_file)

  # Calculate savings
  savings = []
  for i in range(len(origin_distance)):
      for j in range(i+1, len(origin_distance)):
        saving = origin_distance[i] + origin_distance[j] - distance_matrix[i][j]
        savings.append((saving, i, j))

  savings.sort(reverse=True)

  # Initialize orders

  for order_id in range(num_orders):
    o = Order([i for i in orders_items_list if i[ITEM_FIELDS.ORDER_ID] == order_id], savings)

    o.route = simulated_annealing_order_items(
       o.route, 
       distance_matrix, 
       args.initial_temperature_order_items, 
       args.cooling_rate_order_items,
       args.l_order_items,
       args.k2_order_items,
       args.epsilon2_order_items
       )

    global_order_list.append(o)
    print(f"order{order_id}: {o.items}" )

  iterations = []

  current_order_list = list(range(0, len(global_order_list)))
  assign_orders_to_batches(current_order_list, batches)
  run_sim_ann_foreach_batch(
     batches, 
     args.initial_temperature_batch_orders, 
     args.cooling_rate_batch_orders,
     args.l_batch_orders,
     args.k2_batch_orders,
     args.epsilon2_batch_orders
     )
  current_makespan = calculate_makespan(args.num_pickers, batches)

  iterations.append({
    "order_list": current_order_list,
    "batches": batches,
    "makespan": current_makespan
  })
  
  best_iteration = 0

  temperature = args.initial_temperature_orders
  k = 1  # Plateau counter
  terminate = False
  no_improvement_count = 0
  accepted_count = 0
  L = args.l_orders
  K2 = args.k2_orders
  epsilon2 = args.epsilon2_orders
  temperature_list = []
  makespan_list = []
  best_makespan_list = []

  while not terminate:

    Batch.reset_num_batches()
    batches = []

    current_iteration = iterations[-1]

    temperature_list.append(temperature)
    makespan_list.append(current_iteration["makespan"][1])
    best_makespan_list.append(iterations[best_iteration]["makespan"][1])

    # Swap two items in the new_order_list
    new_order_list = current_iteration["order_list"].copy()
    
    i, j = random.sample(range(len(new_order_list)), 2)
    new_order_list[i], new_order_list[j] = new_order_list[j], new_order_list[i]
    
    assign_orders_to_batches(new_order_list, batches)
    run_sim_ann_foreach_batch(
       batches, 
       args.initial_temperature_batch_orders, 
       args.cooling_rate_batch_orders,
       args.l_batch_orders,
       args.k2_batch_orders,
       args.epsilon2_batch_orders)
    new_makespan = calculate_makespan(args.num_pickers, batches)

    if new_makespan[1] < current_iteration["makespan"][1] or random.random() < math.exp((current_iteration["makespan"][1] - new_makespan[1]) / temperature):
      iterations.append({
        "order_list": new_order_list,
        "batches": batches,
        "makespan": new_makespan
      })
      accepted_count += 1

    if new_makespan[1] < iterations[best_iteration]["makespan"][1]:
      best_iteration = len(iterations) - 1
      no_improvement_count = 0
    else:
      no_improvement_count += 1
    
    print({
      "k": k,
      "L": L,
      "no_improvement_count": no_improvement_count,
      "accepted_count": accepted_count,
      "K2": K2,
      "epsilon2": epsilon2,
      "temp": temperature
    })    
    


    # Evaluate stopping conditions
    if k < L:
      k += 1
      terminate = False
    elif no_improvement_count >= K2 and (accepted_count / L) < epsilon2:
      terminate = True
    else:
      temperature *= args.cooling_rate_orders  # Decrease temperature
      print(f"temperature : {temperature}")
      k += 1
      terminate = False
    
    print(iterations[best_iteration])
  

  temperature_list.append(temperature)
  makespan_list.append(iterations[-1]["makespan"][1])
  best_makespan_list.append(iterations[best_iteration]["makespan"][1])
  
  end_time = time.time()
  duration = end_time - start_time

  if args.out_file:
     out_file = args.out_file
  else:
     out_file = f"{args.orders_file}.output.csv"

  np.savetxt(
     out_file,
     np.stack([range(k+1), makespan_list, best_makespan_list, temperature_list], axis=1), 
     fmt="%d", 
     delimiter=",", 
     header="k, Makespan, BestMakespan, Temperatures", 
     comments=""
  )

  args.start_time = start_time
  args.end_time = end_time
  args.duration = duration

  dictionary_to_df = pd.DataFrame(args.__dict__, index=[0])
  with open(f"{args.orders_file}.test_runs.txt", 'a') as f:
        dictionary_to_df.to_csv(f, mode = "a",index=False, header=not f.tell())


if __name__ == "__main__":
  args = parser.parse_args()
  print(f"Args: {args}")
  main(args)
