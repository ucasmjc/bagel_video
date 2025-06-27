def configure_dataloaders(self, **kwargs):
        self.use_offline_emb = self.config.data.get("offline_emb", False)

        # todo: remove this once offline embs are ready
        kwargs.update(
            latent_channels=self.config.vae.model.latent_channels,
            txt_in_dim=self.config.dit.model.txt_in_dim,
        )
        # Load downsample factors.
        vt = self.config.vae.model.get("temporal_downsample_factor", 4)
        vs = self.config.vae.model.get("spatial_downsample_factor", 8)
        pt, ph, pw = _triple(self.config.dit.model.patch_size)
        # Create dataset.
        self.video_dataset = None
        self.image_dataset = None

        process_func = kwargs.pop("process_func", None)

        def _process_datasets(datasets, process_func=None):
            datasets = process_func(datasets) if process_func else datasets
            if isinstance(datasets, ListConfig):
                return DictConfig({f"dataset_{i}": dataset for i, dataset in enumerate(datasets)})
            if not isinstance(datasets, DictConfig):
                raise ValueError("The input datasets must be of type DictConfig or ListConfig.")
            return datasets

        if self.config.data.get("video"):
            datasets = _process_datasets(
                datasets=self.config.data.video.datasets, process_func=process_func
            )
            self.video_dataset = CollectionDataset.from_list(
                datasets=[
                    create_dataset(
                        path=dataset.path,
                        seed=shift_seed(self.seed, i),
                        parquet_sampler=create_parquet_sampler(dataset.get("parquet_sampler")),
                        text_sampler=create_text_sampler(dataset.text_sampler),
                        system_prompt_sampler=create_text_sampler(
                            dataset.get("system_prompt_sampler")
                        ),
                        video_frame_sampler=create_frame_sampler(dataset.frame_sampler),
                        video_transform=Compose(
                            [
                                NaResize(
                                    resolution=dataset.resolution,
                                    mode=dataset.get("resize", "area"),
                                    downsample_only=True,
                                    aspect_ratios=dataset.get("aspect_ratios"),
                                    factor=(vs * ph, vs * pw),
                                ),
                                DivisibleCrop((vs * ph, vs * pw)),
                                Normalize(0.5, 0.5),
                                Rearrange("t c h w -> c t h w"),
                            ]
                        ),
                        use_offline_emb=self.use_offline_emb,
                        system_prompt_prob=dataset.get("system_prompt_prob", 0.0),
                        key=key,
                        caption_type=dataset.get("caption_type", "vfm"),
                        caption_prefix=dataset.get("caption_prefix"),
                        **kwargs,
                    )
                    for i, (key, dataset) in enumerate(datasets.items())
                ],
                weights=[dataset.weight for dataset in datasets.values()],
                seed=self.seed,
            )

        if self.config.data.get("image"):
            datasets = _process_datasets(
                datasets=self.config.data.image.datasets, process_func=process_func
            )
            self.image_dataset = CollectionDataset.from_list(
                datasets=[
                    create_dataset(
                        path=dataset.path,
                        seed=shift_seed(self.seed, i),
                        parquet_sampler=create_parquet_sampler(dataset.get("parquet_sampler")),
                        text_sampler=create_text_sampler(dataset.text_sampler),
                        image_transform=Compose(
                            [
                                NaResize(
                                    resolution=dataset.resolution,
                                    mode=dataset.get("resize", "area"),
                                    downsample_only=True,
                                ),
                                DivisibleCrop((vs * ph, vs * pw)),
                                Normalize(0.5, 0.5),
                            ]
                        ),
                        use_offline_emb=self.use_offline_emb,
                        key=key,
                        **kwargs,
                    )
                    for i, (key, dataset) in enumerate(datasets.items())
                ],
                weights=[dataset.weight for dataset in datasets.values()],
                seed=self.seed,
            )

        # Create video dataloader.
        if self.video_dataset is None:
            self.video_dataloader = itertools.cycle([])
        elif self.config.data.video.get("use_mariana_dataloader", False):
            # We do not need the torch Dataloader wrapper when offline mariana dataloader is used.
            self.video_dataloader = self.video_dataset
        else:
            self.video_dataloader = DataLoader(
                dataset=self.video_dataset,
                batch_size=None,
                num_workers=self.config.data.video.dataloader.num_workers,
                prefetch_factor=self.config.data.video.dataloader.prefetch_factor,
                pin_memory=True,
                pin_memory_device=str(get_device()),
            )

        # Create image dataloader
        if self.image_dataset is None:
            self.image_dataloader = itertools.cycle([])
        elif self.config.data.image.get("use_mariana_dataloader", False):
            # We do not need the torch Dataloader wrapper when offline mariana dataloader is used.
            self.image_dataloader = self.image_dataset
        else:
            self.image_dataloader = DataLoader(
                dataset=self.image_dataset,
                batch_size=None,
                num_workers=self.config.data.image.dataloader.num_workers,
                prefetch_factor=self.config.data.image.dataloader.prefetch_factor,
                pin_memory=True,
                pin_memory_device=str(get_device()),
            )

        # Restore dataloader checkpoint(only when video and image are both offline
        # Mariana Dataloader)
        if self.resume and self.config.data.get("resume", True):
            # Load dataloader states.
            if "video" in self.resume.dataloaders.keys():
                self.video_dataloader.__setstate__(
                    self.resume.dataloaders["video"].states.load(device="cpu", distributed=False)
                )
                self.logger.info("Successfully restored state for video dataloader")

            if "image" in self.resume.dataloaders.keys():
                self.image_dataloader.__setstate__(
                    self.resume.dataloaders["image"].states.load(device="cpu", distributed=False)
                )
                self.logger.info("Successfully restored state for image dataloader")

        # Create data mixer.
        self.datamixer_schedule = create_schedule(
            config=self.config.data.mixer.video_ratio_schedule,
        )

        def attn_seq_shape_compute_fn(t, h, w):
            if self.use_offline_emb:
                return math.ceil(t / pt), h // ph, w // pw
            else:
                return math.ceil(((t - 1) // vt + 1) / pt), h // vs // ph, w // vs // pw

        # Do `iter(dataloader)` to initiate data prefetching.
        self.datamixer = NaMixBatchify(
            video_dataset=iter(self.video_dataloader),
            image_dataset=iter(self.image_dataloader),
            attn_seq_max_length=self.config.data.mixer.attn_seq_max_length,
            video_ratio=self.datamixer_schedule[self.resume.step if self.resume else 0],
            attn_seq_shape_compute_fn=attn_seq_shape_compute_fn,
            use_offline_emb=self.use_offline_emb,
        )

        if self.config.data.get("prefetch_batches", 1) > 1:
            self.datamixer = NaLoadBalance(
                datamixer=self.datamixer,
                balance_group_size=self.config.data.get("balance_group_size", get_world_size()),
                prefetch_batches=self.config.data.get("prefetch_batches", 1),
                hid_dim=self.config.dit.model.vid_dim,
                debug=self.config.data.get("debug", False),
                balance_mode=self.config.data.get("balance_mode", None),
                efficiency_info_pkl=self.config.data.get("efficiency_info_pkl", None),
                temperal_window_size=self.config.dit.model.get("temporal_window_size", None),
                temporal_shifted=self.config.dit.model.get("temporal_shifted", None),
            )

def dumping_file(self, source_file, target_file, is_last_file=False):

        torch.cuda.empty_cache()
        self.logger.info(f"Dumping offline emb of source_file: {source_file}")
        latent_list = []
        latent_last_list = []
        vision_emb_list = []
        text_emb_list = []
        uttid_list = []
        text_list = []
        meta_list = []
        frame_info_list = []
        is_valid_list = []
        caption_type_list = []

        # Set up filter function.
        process_func = partial(
            filter_datasets,
            key=self.config.data.get("key"),
            dump_config=self.config.data.get("dump", {}),
        )

        mock_data = self.config.data.get("mock_data", False)
        is_pd_sr = self.config.data.get("is_pd_sr", False)

        # Get supp_data
        supp_data = None
        if self.config.data.get("supp_data"):
            assert self.use_text_model, "only works when using text model!"
            supp_data = self.config.data.supp_data
            if isinstance(supp_data, (DictConfig, ListConfig)):
                supp_data = OmegaConf.to_object(supp_data)
        elif self.supp_data_dict:
            supp_data = self.supp_data_dict.get(os.path.dirname(source_file))

        # Set up vision processor func
        # Configure dataloader for each parquet file.
        self.configure_dataloaders(
            data_path=source_file,
            process_func=process_func,
            supp_data=supp_data,
            mock_data=mock_data,
            is_pd_sr=is_pd_sr,
            **self.config.data.parquet,
        )

        # Set up progress bar.
        pbar = tqdm(initial=0, dynamic_ncols=True, postfix={"RANK": get_global_rank()})

        # Set up data iterator.
        datamixer_iter = iter(self.datamixer)

        # Loop over the dataloader.
        while True:
            # if self.log_mfu:
            #     self.mfu_start.record()
            try:
                video_ratio = self.datamixer_schedule[0]
                assert video_ratio in [0, 1], "video_ratio must be either 0 or 1"
                self.datamixer.set_video_ratio(video_ratio)
                video_batch, image_batch = next(datamixer_iter)
            except StopIteration as ex:
                self.logger.info(f"Meet last sample of source_file {source_file}, {ex}")
                break

            data_type, data_batch = (
                ("video_dict", video_batch) if video_ratio == 1 else ("image", image_batch)
            )

            with enable_flops_accumulate():
                (
                    latent_pkls,
                    latent_last_pkls,
                    vision_emb_pkls,
                    text_emb_pkls,
                    uttids,
                    texts,
                    metas,
                    frame_infos,
                    is_valid,
                    caption_types,
                ) = self.dumping_step(data_batch, data_type)

            # torch.cuda.empty_cache()
            # gc.collect()

            # if self.log_mfu:
            #     self.mfu_end.record()
            #     torch.cuda.synchronize()
            latent_list.extend(latent_pkls)
            latent_last_list.extend(latent_last_pkls)
            vision_emb_list.extend(vision_emb_pkls)
            text_emb_list.extend(text_emb_pkls)
            uttid_list.extend(uttids)
            text_list.extend(texts)
            meta_list.extend(metas)
            frame_info_list.extend(frame_infos)
            is_valid_list.extend(is_valid)
            caption_type_list.extend(caption_types)
            pbar.set_postfix(
                {
                    "RANK": get_global_rank(),
                }
            )
            pbar.update()

        gc.collect()
        # Get the row_group_size and original 'uttid' order
        uttid_order_dict, row_group_size = get_parquet_order_info(source_file)[0]
        # with open_parquet_files(source_file) as (parquet_file, _):
        #     table = parquet_file.read(columns=["uttid"])
        #     uttid_order = table.to_pandas()["uttid"].tolist()
        #     uttid_order_dict = {value: index for index, value in enumerate(uttid_order)}
        #     row_group_size = parquet_file.metadata.row_group(0).num_rows

        sorted_items = sorted(zip(uttid_list, text_list, meta_list), key=lambda x: uttid_order_dict[x[0]])

        sorted_uttid, sorted_text, sorted_meta = zip(*sorted_items)
        data_dict = {"uttid": list(sorted_uttid), "meta": list(sorted_meta)}

        if self.save_mode == "text_emb_sys":
            data_dict["text_sys"] = list(sorted_text)
        else:
            data_dict["text"] = list(sorted_text)

        if latent_list:
            assert len(latent_list) == len(uttid_list)
            _, sorted_latent = zip(*sorted(zip(uttid_list, latent_list), key=lambda x: uttid_order_dict[x[0]]))
            data_dict["latent"] = list(sorted_latent)

        if latent_last_list:
            assert len(latent_last_list) == len(uttid_list)
            _, sorted_latent_last = zip(
                *sorted(zip(uttid_list, latent_last_list), key=lambda x: uttid_order_dict[x[0]])
            )
            data_dict["latent_last"] = list(sorted_latent_last)

        if frame_info_list:
            assert len(frame_info_list) == len(uttid_list)
            _, sorted_frame_info = zip(
                *sorted(zip(uttid_list, frame_info_list), key=lambda x: uttid_order_dict[x[0]])
            )
            data_dict["frame_info"] = list(sorted_frame_info)

        if text_emb_list:
            assert len(text_emb_list) == len(uttid_list)
            _, sorted_text_emb = zip(*sorted(zip(uttid_list, text_emb_list), key=lambda x: uttid_order_dict[x[0]]))

            if self.save_mode == "text_emb_sys":
                data_dict["text_sys_emb"] = sorted_text_emb
            else:
                data_dict["text_emb"] = sorted_text_emb

        if vision_emb_list:
            assert len(vision_emb_list) == len(uttid_list)
            _, sorted_vision_emb = zip(
                *sorted(zip(uttid_list, vision_emb_list), key=lambda x: uttid_order_dict[x[0]])
            )
            data_dict["vision_emb"] = list(sorted_vision_emb)

        if is_valid_list:
            assert len(is_valid_list) == len(uttid_list)
            _, sorted_is_valid = zip(
                *sorted(zip(uttid_list, is_valid_list), key=lambda x: uttid_order_dict[x[0]])
            )
            data_dict["is_valid"] = list(sorted_is_valid)

        if caption_type_list:
            assert len(caption_type_list) == len(uttid_list)
            _, sorted_caption_type = zip(
                *sorted(zip(uttid_list, caption_type_list), key=lambda x: uttid_order_dict[x[0]])
            )
            data_dict["caption_type"] = list(sorted_caption_type)

        # parquet_meta = get_parquet_metadata(source_file)[0]
        # num_row_groups = parquet_meta.num_row_groups
        # row_group_size = math.ceil(len(uttid_list) / num_row_groups)

        assert row_group_size > 0
        self.dumping_manager.save_parquet(
            data_dict=data_dict,
            file_name=target_file,
            row_group_size=row_group_size,
            is_last_file=is_last_file,
        )

def get_filesystem(path: str) -> Union[LocalFileSystem, HadoopFileSystem]:
    """
    Get filesystem based on the path.
    """
    if path.startswith("hdfs://"):
        return HadoopFileSystem.from_uri(path)
    else:
        return LocalFileSystem()


def read_metadata(
    path: str,
):
    fs = get_filesystem(path)
    with ParquetFile(path, filesystem=fs) as file:
        metadata = file.metadata
    return metadata

def read_order_info(
    path: str,
):
    fs = get_filesystem(path)
    with ParquetFile(path, filesystem=fs) as file:
        try:
            table = file.read(columns=["uttid"])
            uttid_order = table.to_pandas()["uttid"].tolist()
        except Exception as e:
            table = file.read(columns=["uid"])
            uttid_order = table.to_pandas()["uid"].tolist()
        uttid_order_dict = {value: index for index, value in enumerate(uttid_order)}
        row_group_size = file.metadata.row_group(0).num_rows
    return (uttid_order_dict, row_group_size)


def get_parquet_metadata(
    file_path: Union[str, List[str]],
    num_processes: int = 1,
):
    paths = [file_path] if isinstance(file_path, str) else file_path
    with Pool(num_processes) as pool:
        metadata = pool.map(read_metadata, paths)
    return metadata

def get_parquet_order_info(
    file_path: Union[str, List[str]],
    num_processes: int = 1,
):
    paths = [file_path] if isinstance(file_path, str) else file_path
    with Pool(num_processes) as pool:
        metadata = pool.map(read_order_info, paths)
    return metadata