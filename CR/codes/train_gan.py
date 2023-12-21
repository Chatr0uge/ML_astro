plt.ion() # to be able to plot at every epoch

batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for n, (real_samples, useless_labels) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples = real_samples
        real_samples_labels = torch.ones((batch_size, 1))
        latent_space_samples = torch.randn((batch_size, 100))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1))
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))
        
        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()
        
        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 100))
        
        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()
    
    # Plotting generated data
    latent_space_samples = torch.randn((16, 100))
    generated_samples = generator(latent_space_samples).cpu().detach()
    
        # to see if it is stuck
    for i in range(16): 
        ax = plt.subplot(4, 4, i+1)
        plt.imshow(generated_samples[i].reshape(64, 64))
        plt.xticks([])
        plt.yticks([])
    
        # to compare with dataset
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(real_samples[2].reshape(64,64))
    ax1.set_title('Training set')
    ax2.imshow(generated_samples[2].reshape(64,64))
    ax2.set_title('Generated image')
    plt.show()
    
    print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
    print(f"Epoch: {epoch} Loss G.: {loss_generator}")   
    
plt.ioff()