## **Token sales from Coinbase**

Run in poetry shell
```
python scripts/risk_pipeline/data_prep/coinbase_data_prep.py --pair ETH-USD --pull_periodicity 300 --final_periodicity 600 --start_time 2022-01-01-00-00 --end_time 2022-03-01-00-00
```

## **NFT Floors from Reservoir.tools**
Run in poetry shell
```
python scripts/risk_pipeline/data_prep/reservoir_floors_data_prep.py --collection_addr 0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D --collection_name BAYC --start_time 2023-03-01-00-00 --end_time 2023-03-10-00-00 --final_periodicity 7200
```