from .utils_vad import OnnxWrapper

def load_silero_vad(session_opts, providers, provider_options):
    model_name = 'silero_vad.onnx'
    package_path = "silero_vad.data"
    
    try:
        import importlib_resources as impresources
        model_file_path = str(impresources.files(package_path).joinpath(model_name))
    except:
        from importlib import resources as impresources
        try:
            with impresources.path(package_path, model_name) as f:
                model_file_path = f
        except:
            model_file_path = str(impresources.files(package_path).joinpath(model_name))

    model = OnnxWrapper(model_file_path, session_opts, providers, provider_options)
    
    return model
