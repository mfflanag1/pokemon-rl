"""V2 admits valid refreshed-request feedback without filtering sampled decisions."""
import hashlib
import numpy as np
from pokemon_transfer.common import (read,initialize,select,specs,probe_specs,tactical,summarize,verified_inventory)
from pokemon_transfer.common import predecessor as original_predecessor
from pokemon_full_train.storage import ROOT,canonical,digest,atomic_json
from pokemon_full_learning.campaign import check_games,evaluation_specs
from pokemon_fit.data import require


def predecessor(plan):
    batch,inventory=original_predecessor(plan)
    path=ROOT/plan['failed_attempt_receipt'];r=read(path);body=dict(r);h=body.pop('content_hash')
    require(h==plan['failed_attempt_content_hash']==hashlib.sha256(canonical(body)).hexdigest(),'Failed attempt receipt binding')
    require(path.read_bytes()==(ROOT/'artifacts/manifests'/f'f9-full-transfer-failure-{h}'/'receipt.json').read_bytes(),'Failure archive mismatch')
    for rel,sha in r['source_files'].items():require(digest(ROOT/rel)==sha and digest(ROOT/r['source_snapshot_root']/rel)==sha,f'Failed attempt source drift: {rel}')
    for rel,sha in r['evidence_files'].items():require(digest(ROOT/rel)==sha,f'Failed attempt evidence drift: {rel}')
    return batch,inventory


def check_collection(collection,training):
    for rows,game in collection:
        n=game['unavailable_choices']
        require(type(n) is int and 0<=n<=game['decisions'],'Invalid unavailable-choice counter')
        require(len(rows)==game['decisions'] if training else not rows,'Wrong learner-row role')


def collect_update(pool,p,plan,seed,index,cursor):
    collection=[];profiles=[];first=cursor
    while sum(len(rows) for rows,_ in collection)<plan['selected_rows_per_update']:
        require(cursor-first<plan['maximum_games_per_update'],'Fixed collection cap exhausted')
        chunk,profile=pool.collect(p,specs(plan,seed,cursor));games=[g for _,g in chunk]
        check_games(games,split='train',first=cursor,count=plan['training_chunk_games']);check_collection(chunk,True)
        collection.extend(chunk);profiles.append(profile);cursor+=plan['training_chunk_games']
    batch,selection=select(collection,seed,index,plan['selected_rows_per_update'])
    return batch,{'first_episode':first,'next_episode':cursor,'games':[g for _,g in collection],'selection':selection},profiles


def evaluation(pool,p,plan,seed,evidence=None):
    result={};profiles=[]
    for opponent in plan['opponents']:
        games=[]
        for first in range(0,plan['evaluation_pairs_per_opponent'],plan['evaluation_chunk_pairs']):
            jobs=evaluation_specs(plan,seed,opponent,first,plan['evaluation_chunk_pairs'])
            chunk,profile=pool.collect(p,jobs)
            # Write the completed chunk before any new coordinator admission can fail.
            if evidence is not None:atomic_json(evidence/f'{opponent}-{first:04d}.json',{'specs':jobs,'games':[g for _,g in chunk],'row_counts':[len(rows) for rows,g in chunk]})
            check_collection(chunk,False);games.extend(g for _,g in chunk);profiles.append(profile)
        check_games(games,split='development',first=0,count=plan['evaluation_pairs_per_opponent'],opponent=opponent)
        result[opponent]=games
    return result,profiles
