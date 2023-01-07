'''
Biaffine Dependency parser from AllenNLP
'''
import argparse
import json
import os
from allennlp.predictors.predictor import Predictor
from lxml import etree
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm
import copy

model_path = '/home/cq/RGAT-ABSA/premodel/biaffine-dependency-parser-ptb-2020.04.06.tar.gz'


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--model_path', type=str, default=model_path,
                        help='Path to biaffine dependency parser.')
    parser.add_argument('--data_path', type=str, default='/home/cq/RGAT-ABSA/data/semeval14',
                        help='Directory of where semeval14 or twiiter data held.')
    return parser.parse_args()


def xml2txt(file_path):
    '''
    Read the original xml file of semeval data and extract the text that have aspect terms.
    Store them in txt file.
    '''
    output = file_path.replace('.xml', '_text.txt')
    sent_list = []
    with open(file_path, 'rb') as f:
        raw = f.read()
        root = etree.fromstring(raw)
        for sentence in root:
            sent = sentence.find('text').text
            terms = sentence.find('aspectTerms')
            if terms is None:
                continue
            if terms:
                sent_list.append(sent)
    with open(output, 'w') as f:
        for s in sent_list:
            f.write(s + '\n')
    print('processed', len(sent_list), 'of', file_path)


def text2docs(file_path, predictor):
    '''
    Annotate the sentences from extracted txt file using AllenNLP's predictor.
    '''
    with open(file_path, 'r') as f:
        sentences = f.readlines()
    docs = []
    print('Predicting dependency information...')
    for i in tqdm(range(len(sentences))):
        docs.append(predictor.predict(sentence=sentences[i]))

    return docs


def dependencies2format(doc):  # doc.sentences[i]
    '''
    Format annotation: sentence of keys
                                - tokens
                                - tags
                                - predicted_dependencies
                                - predicted_heads
                                - dependencies
    '''
    sentence = {}
    sentence['tokens'] = doc['words']
    sentence['tags'] = doc['pos']
    # sentence['energy'] = doc['energy']
    predicted_dependencies = doc['predicted_dependencies']
    predicted_heads = doc['predicted_heads']
    sentence['predicted_dependencies'] = doc['predicted_dependencies']
    sentence['predicted_heads'] = doc['predicted_heads']
    sentence['dependencies'] = []
    for idx, item in enumerate(predicted_dependencies):
        dep_tag = item
        frm = predicted_heads[idx]
        to = idx + 1
        sentence['dependencies'].append([dep_tag, frm, to])

    return sentence


def get_dependencies(file_path, predictor):
    docs = text2docs(file_path, predictor)
    sentences = [dependencies2format(doc) for doc in docs]
    return sentences


def syntaxInfo2json(sentences, origin_file):
    json_data = []
    tk = TreebankWordTokenizer()
    mismatch_counter = 0
    idx = 0
    with open(origin_file, 'rb') as fopen:
        raw = fopen.read()
        root = etree.fromstring(raw)
        for sentence in root:
            example = dict()
            example["sentence"] = sentence.find('text').text

            # for RAN
            terms = sentence.find('aspectTerms')
            if terms is None:
                continue

            example['tokens'] = sentences[idx]['tokens']
            example['tags'] = sentences[idx]['tags']
            example['predicted_dependencies'] = sentences[idx]['predicted_dependencies']
            example['predicted_heads'] = sentences[idx]['predicted_heads']
            example['dependencies'] = sentences[idx]['dependencies']
            # example['energy'] = sentences[idx]['energy']

            example["aspect_sentiment"] = []
            example['from_to'] = []  # left and right offset of the target word

            for c in terms:
                if c.attrib['polarity'] == 'conflict':
                    continue
                target = c.attrib['term']
                example["aspect_sentiment"].append((target, c.attrib['polarity']))

                # index in strings, we want index in tokens
                left_index = int(c.attrib['from'])
                right_index = int(c.attrib['to'])

                left_word_offset = len(tk.tokenize(example['sentence'][:left_index]))
                to_word_offset = len(tk.tokenize(example['sentence'][:right_index]))

                example['from_to'].append((left_word_offset, to_word_offset))
            if len(example['aspect_sentiment']) == 0:
                idx += 1
                continue
            json_data.append(example)
            idx += 1
    extended_filename = origin_file.replace('.xml', '_biaffine_depparsed.json')
    with open(extended_filename, 'w') as f:
        json.dump(json_data, f)
    print('done', len(json_data))
    print(idx)


def main():
    args = parse_args()
    predictor = Predictor.from_path(args.model_path)

    data = [('Restaurants_Train_v2.xml', 'Restaurants_Test_Gold.xml'),
            ('Laptop_Train_v2.xml', 'Laptops_Test_Gold.xml')]
    for train_file, test_file in data:
        # xml -> txt
        xml2txt(os.path.join(args.data_path, train_file))
        xml2txt(os.path.join(args.data_path, test_file))

        # txt -> json
        train_sentences = get_dependencies(
            os.path.join(args.data_path, train_file.replace('.xml', '_text.txt')), predictor)
        test_sentences = get_dependencies(os.path.join(
            args.data_path, test_file.replace('.xml', '_text.txt')), predictor)

        print(len(train_sentences), len(test_sentences))

        syntaxInfo2json(train_sentences, os.path.join(args.data_path, train_file))
        syntaxInfo2json(test_sentences, os.path.join(args.data_path, test_file))


def add_ner_token(file_path, tgt_path, check_error=False):
    final_data = []
    final_data_change_froms = []
    # O:1,B-ASP:2,I-ASP:3 3种token,padding 用0，计算交叉熵时忽略

    with open(file_path, 'r') as f:
        data = json.load(f)
    cnt = 0
    for line in data:
        from_tos = line['from_to']
        ner_token = ['O' for _ in range(len(line['tokens']))]
        new_from_tos = []
        for i, from_to in enumerate(from_tos):
            real_token = line['aspect_sentiment'][i][0]
            tokens = line['tokens']
            token = line['tokens'][from_to[0]:from_to[1]]
            if check_error and real_token != ' '.join(token):
                try:
                    fir = real_token.split(' ')[0]
                    idx_fir = tokens.index(fir)
                    if ' '.join(tokens[idx_fir:idx_fir + len(real_token.split(' '))]) == real_token:
                        from_to = [idx_fir, idx_fir + len(real_token.split(' '))]
                        cnt += 1
                    else:
                        correct_error(tokens, real_token, token, from_to[0], from_to[1])
                        cnt += 1
                except:
                    correct_error(tokens, real_token, token, from_to[0], from_to[1])

            ner_token[from_to[0]] = 'B-ASP'
            for j in range(from_to[0] + 1, from_to[1]):
                ner_token[j] = 'I-ASP'
                # ner_labels[i] = 3
            # 修改错误的from to
            new_from_tos.append(from_to)
        line['ner_tokens'] = ner_token
        final_data.append(line)
        new_line = copy.deepcopy(line)
        new_line['from_to'] = new_from_tos
        final_data_change_froms.append(new_line)
    with open(tgt_path, 'w') as f:
        json.dump(final_data, f)
    # with open(tgt_path + 'change_froms', 'w') as f:
    #     json.dump(final_data_change_froms, f)


def correct_error(tokens, real_token, token, from_, to_):
    print(real_token)
    print(token)
    print(tokens)
    if 0 <= from_ < len(tokens):
        print('from:{}  from tok:{}'.format(from_, tokens[from_]))
    if 0 <= to_ < len(tokens):
        print('to:{}   to tok:{}'.format(to_, tokens[to_]))
    new_from = int(input('input from:'))
    new_to = int(input('input to:'))
    print()
    return new_from, new_to


if __name__ == "__main__":
    # main()
    # add_ner_token('/home/cq/RGAT-ABSA/data/semeval14/Restaurants_Train_v2_biaffine_depparsed_with_energy.json',
    #               '/home/cq/RGAT-ABSA/data/semeval14/Restaurants_Train_add_ner_remove_error.json')
    # add_ner_token('/home/cq/RGAT-ABSA/data/semeval14/Restaurants_Test_Gold_biaffine_depparsed_with_energy.json',
    #               '/home/cq/RGAT-ABSA/data/semeval14/Restaurants_Test_add_ner_remove_error.json')
    # add_ner_token('/home/cq/RGAT-ABSA/data/semeval14/Laptop_Train_v2_biaffine_depparsed.json',
    #               '/home/cq/RGAT-ABSA/data/semeval14/Laptop_Train_v2_add_ner.json')
    # add_ner_token('/home/cq/RGAT-ABSA/data/semeval14/Laptops_Test_Gold_biaffine_depparsed.json',
    #               '/home/cq/RGAT-ABSA/data/semeval14/Laptops_Test_Gold_add_ner.json')
    # add_ner_token('/home/cq/RGAT-ABSA/data/twitter/train_biaffine.json',
    #               '/home/cq/RGAT-ABSA/data/twitter/train_biaffine_add_ner.json')
    # add_ner_token('/home/cq/RGAT-ABSA/data/twitter/test_biaffine.json',
    #               '/home/cq/RGAT-ABSA/data/twitter/test_biaffine_add_ner.json')

    # add_ner_token('/home/cq/RGAT-ABSA/data/semeval14/Restaurants_Train_v2_biaffine_depparsed_with_energy.json',
    #               '/home/cq/RGAT-ABSA/data/semeval14/Restaurants_Train_add_ner_remove_error.json', True)
    # add_ner_token('/home/cq/RGAT-ABSA/data/semeval14/Restaurants_Test_Gold_biaffine_depparsed_with_energy.json',
    #               '/home/cq/RGAT-ABSA/data/semeval14/Restaurants_Test_add_ner_remove_error.json', True)
    # add_ner_token('/home/cq/RoBERTaABSA/Perturbed-Masking/rgat/robertaLaptoptrainedFT0/11/Laptop/Laptop_Train.json',
    #               '/home/cq/RGAT-ABSA/data/semeval14/Laptop_Train_parser.json')
    # add_ner_token('/home/cq/RoBERTaABSA/Perturbed-Masking/rgat/robertaLaptoptrainedFT0/11/Laptop/Laptop_Test.json',
    #               '/home/cq/RGAT-ABSA/data/semeval14/Laptop_Test_parser.json')
    # add_ner_token('/home/cq/RoBERTaABSA/Perturbed-Masking/rgat/robertaRestaurantstrainedFT0/11/Restaurants/Restaurants_Train.json',
    #               '/home/cq/RGAT-ABSA/data/semeval14/Restaurants_Train_parser.json')
    # add_ner_token('/home/cq/RoBERTaABSA/Perturbed-Masking/rgat/robertaRestaurantstrainedFT0/11/Restaurants/Restaurants_Test.json',
    #               '/home/cq/RGAT-ABSA/data/semeval14/Restaurants_Test_parser.json')
    add_ner_token(
        '/home/cq/RoBERTaABSA/Perturbed-Masking/rgat/robertaTweetstrainedFT0/11/Tweets/Tweets_Train.json',
        '/home/cq/RGAT-ABSA/data/semeval14/Twitter_Train_parser.json')
    add_ner_token(
        '/home/cq/RoBERTaABSA/Perturbed-Masking/rgat/robertaTweetstrainedFT0/11/Tweets/Tweets_Test.json',
        '/home/cq/RGAT-ABSA/data/semeval14/Twitter_Test_parser.json')
