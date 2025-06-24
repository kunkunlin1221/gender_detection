import chameleon as cl


class GenderModel(cl.PowerModule):
    def __init__(self):
        super().__init__()
        self.model = cl.build_backbone("timm_lcnet_050", num_classes=2, pretrained=True)

    def forward(self, x):
        return self.model(x)
