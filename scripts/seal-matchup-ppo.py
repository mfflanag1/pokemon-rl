from pokemon_matchup.seal import *
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--review",type=Path,required=True)
    a=p.parse_args();print(seal(a.run.resolve(),a.review.resolve()),flush=True)
