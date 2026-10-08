from .base_options import BaseOptions


class InferenceOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        parser.add_argument('--data_path', type=str,
                          default='/data0/yuanyz/301/DKD/StainingTest/PAS_whole_images_cropped/2019-860')
        parser.add_argument('--save_path', type=str,
                          default='/data0/yuanyz/301/DKD/StainingTest/PAS_whole_images_inference/2019-860')
        parser.add_argument('--num_test', type=int, default=1000000000)
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
