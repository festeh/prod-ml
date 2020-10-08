from tensorflow_transform import TFTransformOutput

from training.model import get_model


def input_fn(*args, **kwargs):
    return None


def run_fn(args):
    tf_transform_output = TFTransformOutput(args.transform_output_dir)
    train_dataset = input_fn(args.train_files, tf_transform_output)
    val_dataset = input_fn(args.val_files, tf_transform_output)

    model = get_model()
    model.fit(train_dataset,
              steps_per_epoch=args.train_steps,
              validation_data=val_dataset,
              validation_steps=args.val_steps)

    model.save()