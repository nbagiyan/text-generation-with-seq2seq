import pandas as pd
import argparse
import logging

if __name__ == '__main__':
    logger = logging.getLogger('prepare data logger')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser(
        description="Configs for preparing data")
    parser.add_argument('--clickbait-data')
    parser.add_argument('--nonclickbait-data')
    parser.add_argument('--file-save-path')
    args = vars(parser.parse_args())

    logger.info('Reading data')
    clickbait = pd.read_csv(args['clickbait_data'])
    logger.info('Finished')
    logger.info('Preprocessing data')
    clickbait.drop(['publish_date'], axis=1, inplace=True)
    clickbait.dropna(inplace=True)
    clickbait.columns = ['headline']
    clickbait['headline'] = clickbait['headline'].apply(lambda x: x.lower())
    clickbait['target'] = 1
    logger.info('Finished')
    logger.info('Reading data')
    non_clickbait = pd.read_csv(args['nonclickbait_data'])
    logger.info('Finished')
    logger.info('Preprocessing data')
    non_clickbait.dropna(inplace=True)
    non_clickbait.drop(['publish_date'], axis=1, inplace=True)
    non_clickbait.columns = ['headline']
    non_clickbait['target'] = 0
    logger.info('Finished')
    logger.info('Saving data')
    df_all = pd.concat([clickbait, non_clickbait], ignore_index=True)
    df_all.to_csv(args['file_save_path'], index=False)
    logger.info('Finished')
