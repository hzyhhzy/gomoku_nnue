import argparse
import os
import ctypes
import lz4.frame
from enum import Enum

import numpy as np

# ------------------------------------------------


class Result(Enum):
    LOSS = 0
    DRAW = 1
    WIN = 2

    def opposite(r):
        return Result(2 - r.value)


class Rule(Enum):
    FREESTYLE = 0
    STANDARD = 1
    RENJU = 4


class Move():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def value(self):
        return (self.x << 5) | self.y

    def __repr__(self):
        return f'({self.x},{self.y})'

# ------------------------------------------------


class Entry(ctypes.Structure):
    _fields_ = [
        ('result', ctypes.c_uint16, 2),
        ('ply', ctypes.c_uint16, 9),
        ('boardsize', ctypes.c_uint16, 5),
        ('rule', ctypes.c_uint16, 3),
        ('move', ctypes.c_uint16, 13),
        ('position', ctypes.c_uint16 * 1024)
    ]


class EntryHead(ctypes.Structure):
    _fields_ = [
        ('result', ctypes.c_uint16, 2),
        ('ply', ctypes.c_uint16, 9),
        ('boardsize', ctypes.c_uint16, 5),
        ('rule', ctypes.c_uint16, 3),
        ('move', ctypes.c_uint16, 13)
    ]


def write_entry(f, result: Result, boardsize: int, rule: Rule, move: Move, position: list):
    entry = Entry()
    entry.result = result.value
    entry.ply = len(position)
    entry.boardsize = boardsize
    entry.rule = rule.value
    entry.move = move.value()
    for i, m in enumerate(position):
        entry.position[i] = m.value()

    f.write(bytearray(entry)[: 4 + 2 * len(position)])


def read_entry(f):
    ehead = EntryHead()
    f.readinto(ehead)

    result = Result(ehead.result)
    ply = int(ehead.ply)
    boardsize = int(ehead.boardsize)
    rule = Rule(ehead.rule)
    move = Move((ehead.move >> 5) & 31, ehead.move & 31)

    pos_array = (ctypes.c_uint16 * ehead.ply)()
    f.readinto(pos_array)
    position = [Move((m >> 5) & 31, m & 31) for m in pos_array]

    return result, ply, boardsize, rule, move, position


# ------------------------------------------------

def process_bin(index, result, ply, boardsize, rule, move, position,usefulColor):
    #print('-' * 50)
    if(index%10000==0):
        print(f'index: {index}')
    #print(f'result: {result}')
    #print(f'ply: {ply}')
    #print(f'boardsize: {boardsize}')
    #print(f'rule: {rule}')
    #print(f'move: {move}')
    #print(f'position: {position}')



    nextPlayer=1 if (len(position)%2==0) else 2
    if(usefulColor!=0 and nextPlayer!=usefulColor):
        return False,None,None,None,None


    # Add process logic here......
    bf=np.zeros((2,boardsize,boardsize),dtype=np.int8)
    gf=np.zeros((1),dtype=np.float32)
    pt=np.zeros((boardsize,boardsize),dtype=np.int8)
    vt=np.zeros((3),dtype=np.float32)

    #bf
    color=0
    for loc in position:
        bf[color,loc.y,loc.x]=1
        color=1-color
    if(nextPlayer==2):
        bf=np.flip(bf,axis=0)
    #print(bf)

    pt[move.y,move.x]=1

    if (result == Result.WIN):
        vt[0] = 1
    elif(result==Result.LOSS):
        vt[1]=1
    else:
        vt[2]=1
    #print(vt)

    return True,bf,gf,pt,vt




def read_bin():
    parser = argparse.ArgumentParser(
        description="SGF to BIN training format converter")
    parser.add_argument('input',
                        type=str,
                        help="Input binary file name")
    parser.add_argument('output',
                        type=str,
                        help="Output npz file name")
    parser.add_argument('-side',
                        type=str,default='',
                        help="Which side of data to use.b is black, w is white. If empty, use both")
    #parser.add_argument('--compress', action='store_true',
    #                    help="Input is compressed with lz4")
    args = parser.parse_args()

    usefulColor=0
    if(args.side=='b' or args.side=='B'):
        usefulColor=1
    elif(args.side=='w' or args.side=='W'):
        usefulColor=2

    def open_file(fn, mode):
        #if args.compress:
            return lz4.frame.open(fn, mode, compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC)
        #else:
        #    return open(fn, mode)

    total = 0
    bfs=[]
    gfs=[]
    pts=[]
    vts=[]
    with open_file(args.input, 'rb') as input_f:
        try:
            while input_f.peek() != b'':
                data = read_entry(input_f)
                useful,bf,gf,pt,vt=process_bin(total, *data,usefulColor=usefulColor)
                if(useful):
                    bfs.append(bf)
                    gfs.append(gf)
                    pts.append(pt)
                    vts.append(vt)
                    total += 1

            print(f'Finished, total read: {total}')
        except Exception:
            print("Some error occured")
            print(f'total read: {total}')
            pass

    bfs=np.stack(bfs,axis=0)
    gfs=np.stack(gfs,axis=0)
    pts=np.stack(pts,axis=0)
    vts=np.stack(vts,axis=0)
    np.savez_compressed(args.output,bf=bfs,gf=gfs,vt=vts,pt=pts)


if __name__ == "__main__":
    read_bin()
