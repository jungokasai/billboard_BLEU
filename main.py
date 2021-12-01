import argparse, json
import sacrebleu

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--src', type=str, metavar='N',
                    help='source file')
parser.add_argument('--hyp', type=str, metavar='N',
                    help='hypothesis file')
parser.add_argument('--refs', type=str, metavar='N',
                    help='reference file')
parser.add_argument('--outfile', type=str, metavar='N',
                    help='output file')
args = parser.parse_args()

def read_jsonl(infile, extract_key=None):
    f = open(infile, 'r')
    if extract_key is None:
        out = [json.loads(line.strip()) for line in f]
    else:
        out = [json.loads(line.strip())[extract_key] for line in f]
    f.close()
    return out

def score(src, hyp, refs, outfile):
    src = read_jsonl(src, 'src')
    hyp = read_jsonl(hyp, 'hyp')
    refs = read_jsonl(refs, 'refs')
    scores = []
    for hyp_s, refs_s in zip(hyp, refs):
        scores.append(sacrebleu.sentence_bleu(hyp_s, refs_s).score)
    with open(outfile, 'wt') as fout:
        for score in scores:
            fout.write(str(score))
            fout.write('\n')


if __name__ == '__main__':
    score(args.src, args.hyp, args.refs, args.outfile)
