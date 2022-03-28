from .detector3d_template import Detector3DTemplate


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict['dataset_cfg'] = self.dataset.dataset_cfg

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            if 'mimic' not in batch_dict.keys():
                return ret_dict, tb_dict, disp_dict
            else:
                return ret_dict, tb_dict, disp_dict, batch_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if 'mimic' not in batch_dict.keys():
                return pred_dicts, recall_dicts
            else:
                return pred_dicts, recall_dicts, batch_dict

        

            

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
