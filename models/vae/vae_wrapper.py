from models.vae.vae_core import VAE

def vae_wrapper(sensor,
                sensor_test,
                latent_dim,
                list_hidden,
                lr=1e-3,
                epoch_num=30,
                batch_size=32,
                optimizer_name='adam',
                verbose=1,
                optimizer_params={'weight_decay': 1e-5},
                beta=1.0,
                capacity=0.0,
                hidden_activation_name='relu',
                output_activation_name='sigmoid',
                batch_norm=False,
                dropout_rate=0.2,
                is_csv=False,
                seed=42,
                ckpt_dir='./',
                **kwargs):

    latent_dim = sensor.shape[0] if latent_dim is None else latent_dim  # number of latent components
    x = sensor.T
    vae = VAE(lr=lr,
            epoch_num=epoch_num,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            device=None,
            random_state=seed,
            preprocess=not is_csv,
            use_compile=False,
            compile_mode='default',
            verbose=verbose,
            optimizer_params=optimizer_params,
            beta=beta, capacity=capacity,
            encoder_neuron_list=list_hidden,
            decoder_neuron_list=list_hidden[::-1],
            latent_dim=latent_dim,
            hidden_activation_name=hidden_activation_name,
            output_activation_name=output_activation_name,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            n=kwargs.get('n', 1),
            l=kwargs.get('l', 1))
    history = vae.fit(X=x, x_test=sensor_test)
    vae.save(ckpt_dir)
    latent, scores = vae.evaluate(x)
    return latent
