import argparse
from barcode_predict.model import BarcodeModel

def main():
    parser = argparse.ArgumentParser(description='RLScan offline analysis')
    parser.add_argument('--model', type=str,
                        default='simple')  # 'simple' or 'unet'
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--image_comb', type=str, default="time")
    parser.add_argument('--num_images', type=int, default=4)
    parser.add_argument('--color_mode', type=str, default="gray")
    args = parser.parse_args()
    model = BarcodeModel(
        epochs=args.epochs,
        datadir=args.datadir,
        gpu=args.gpu,
    )
    # model.model.load_weights("saved_models/barcode_prediction_model_2021-05-13-12-04-02.pkl")
    model.train()

if __name__ == "__main__":
    main()