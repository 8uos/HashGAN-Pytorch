## Discussion

### Network Structure

논문에서는 사용한 네트워크 구조에 대해서 명확하게 설명하고 있지 않기 때문에, 논문에서 언급한 Improved GAN 구조와 DCGAN 구조에 대해 모두 실험을 진행했다. 논문에서는 Improved GAN 구조와 유사하게 사용했다고 언급하고있으나, DCGAN 구조로 학습했을 때 학습이 더 안정적이고 나은 성능을 보였기 때문에 DCGAN 구조를 채택했다. 사용했던 Improved GAN 구조는 models_improvedGAN.py 에 포함되어 있다.


### Performance

해당 코드로 실험했을 때, precision@1000 기준으로 MNIST에서 0.36-0.44, CIFAR10에서 0.22-0.30 의 성능을 보였다. 이는 논문에서 제시하는 결과보다 많이 떨어지는 수치이다. 이는 Encoder network가 잘 학습되지 못했기 때문이라고 의심된다. 학습이 진행되었을 때 Generator가 생성한 이미지의 질도 충분히 좋은 편이며, Discriminator의 마지막 hidden layer의 feature도 좋은 특성을 가진다. 생성 이미지와 hidden layer feature의 그림을 함께 첨부하였다. (imgs/generated_{dataset}.png, imgs/D_feat_{dataset}.png)
따라서 Generator와 Discriminator는 충분히 학습되었다고 볼 수 있다. 

Encoder 성능을 올리기 위해 1)Loss weight 조절, 2)Minimum entropy loss, Uniform frequency loss form 변경 을 시도해 보았으나, 둘 다 성능이 크게 향상되지는 않았다. 다만, 논문에서는 weight 조절을 하지 않았다고 하였으나 weight 조절에 따른 성능 변화는 무시할 수 없는 수준이었다(~5%). 이러한 문제를 해결하기 위해 저자에게 연락을 시도하였으나 답변을 받지는 못했다.

### Training

논문에서는 학습 epoch 수, batch size 등은 제시하고 있지 않으나, 학습 epoch이 너무 많아지면 Generator가 mode collapse에 빠지는 문제가 발생하였다. Batch size는 큰 영향을 주지 않았고, epoch 수는 50이 가장 적절했다.