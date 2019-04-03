python tripadvisor.py --aspect 0 >&1 | tee trip_asp_0
python tripadvisor.py --aspect 1 >&1 | tee trip_asp_1
python tripadvisor.py --aspect 2 >&1 | tee trip_asp_2
python tripadvisor.py --aspect 3 >&1 | tee trip_asp_3
python tripadvisor.py --aspect 4 >&1 | tee trip_asp_4
python tripadvisor.py --aspect 5 >&1 | tee trip_asp_5
python tripadvisor.py --aspect 6 >&1 | tee trip_asp_6
python beer.py --aspect 0 >&1 | tee beer_asp_0
python beer.py --aspect 1 >&1 | tee beer_asp_1
python beer.py --aspect 2 >&1 | tee beer_asp_2
python beer.py --aspect 3 >&1 | tee beer_asp_3
python avg_acc.py --dataset trip
rm trip_asp_[0-6]
python avg_acc.py --dataset beer
rm beer_asp_[0-4]
