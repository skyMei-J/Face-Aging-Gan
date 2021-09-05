require version: torch 1.3.1, python3

    PREPROCESSING:

        move 'train_label.txt','test_label.txt','test_desired_age.txt' to 0610172_src/

        use jupyter notebook

        open 'dataset_for_dataloader.ipynb'
            set path = {thumbnails128x128_DATASET}
            set output_path = {dataset_for_dataloader_PATH} (classify data by train/test)
            run 'dataset_for_dataloader.ipynb' to prepare dataset for dataloader
            
    
    TRAINING:
        open 'train.py'
            set ITERATION = 900000
            set BATCH = 4 
            set SAVING_POINT = 10000 #every 10000 iteration save one checkpoint
            set label_PATH = 'train_label.txt'
            set load_data_path='{dataset_for_dataloader_PATH}'+'train/'
            run train.py to output checkpoint, please iterate more than 400000
            NOTICE: please don't use the lastest checkpoint, and use the second to last.
            
    TESTING:
		use pretrain weight :unzip 'average.zip', and put 'average' in 0610172_src/
        open 'classify_data.ipynb'
            set path = {thumbnails128x128_DATASET}
            set output_path = {CLASSIFIED_DATASET_PATH} (classify data by age and store in this directory)
            restart kernel and re-run the whole notebook to classify data
			If you want to know How to train 'average':
			{
				open 'find_age_latent.py'
					set ckpt = {CHECKPOINT_PATH}
					set step = 1000
					set train_img_path = {CLASSIFIED_DATASET_PATH}
					set REPRESENTING_LATENT_NUM = 500 #larger REPRESENTING_LATENT_NUM is better but take more time
					run 'find_age_latent.py' to save all the age latents in {CLASSIFIED_DATASET_PATH}/{age}/{this_age_latent.pt}

				open 'average.ipynb'
					set path = {CLASSIFIED_DATASET_PATH}
					run 'average.ipynb' to average all the latents with same age to represent certain age latent
			}
        open 'test.py'
            set step = 1000
            set CHECKPOINT ={CHECKPOINT_PATH}
            set DATASET = {thumbnails128x128_DATASET}
            set aging_answer_path = {output_img_path} default = '../0610172_img/'
            run test.py to save {}_aged.png & {}_rec.png to aging_answer_path

