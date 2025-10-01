from inference_common import (
    create_inference_parser, 
    setup_device_and_dtype, 
    create_pipeline, 
    generate_image, 
    setup_logging
)

def main():
    parser = create_inference_parser()
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    device, dtype = setup_device_and_dtype()
    pipeline = create_pipeline(args.model_name, device, dtype, logger)
    image = generate_image(pipeline, args, logger)
    
    image.save(args.output)
    print(f"Generated image saved to: {args.output}")

if __name__ == "__main__":
    main()