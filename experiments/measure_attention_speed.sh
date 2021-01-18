python3 scripts/measure_attention_speed.py  --vector-size 128 --nb-waves 64 --gaussian --gpu-id 0 --filename 'data/times/gaussian64.txt'
python3 scripts/plot_attention_speed.py --filename data/times/gaussian64.txt
