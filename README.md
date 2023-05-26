调用自己的ResNet34模型进行数据增广
——————————————————————————————————————————
-------------------------------------------------------------------------------------------------------------------------------------------------------------

🔸设置自己的ResNet34模型
---------------------------------------------------------------------------------------------------------------------------------------------------------
遇见的问题：

![捕获](https://github.com/2433792809/calling-self-ResNet34/assets/128702185/e8e3f73a-1fe1-4ed9-975e-0d2918a1a651)
  
错误原因：

这个错误是由于矩阵乘法操作中的维度不匹配引起的。根据错误信息，mat1 的形状是 (4x8192) 而 mat2 的形状是 (4608x2)，无法进行矩阵乘法操作。

解决方法：

在 ResNet34 模型中，将最后一层的线性层 (self.classifier) 的输入维度修改为 8192：

      self.classifier = nn.Linear(8192, self.num_classes)

---------------------------------------------------------------------------------------------------------------------------------------------------------

🔸 搭建测试脚本test.py
---------------------------------------------------------------------------------------------------------------------------------------------------------

📌 根据以前的train脚本可以很轻松的搭建test文件，但是我们需要调用自己的ResNet34模型进行测试

需要改动的有三点


1、在你的代码中导入修改好的ResNet34模型类。假设你将修改后的模型定义在ResNet34.py文件中。

    from ResNet34 import ResNet34
    
    
2、指定模型的名称为ResNet34。   
    
     args = parser.parse_args()
    args.model='ResNet34'
    args.directory = 'dictionary/%s/Hi%s/' % (args.model, args.time_exp_start)
    log_file = os.path.join(args.directory, 'log.json')
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    with open(log_file, 'w') as log:
        json.dump(vars(args), log, indent=4)   


3、创建ResNet34模型实例

        model = ResNet34(num_classes=args.num_classes)
        self.model = model.to(self.device)
        
        
改动完成后进行测试

![捕获2](https://github.com/2433792809/calling-self-ResNet34/assets/128702185/c3231a57-843c-43c1-b650-47a54f8e4752)

