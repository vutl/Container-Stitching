from pydantic import BaseModel
from sources.main_app.api_controller import APIController, APIData, SplitData, APIVideoData
from sources.config import HOST, PORT, TIMEOUT_KEEP_ALIVE, WORKERS
from sources.config import HOST, PORT, TIMEOUT_KEEP_ALIVE, WORKERS
from fastapi import FastAPI
import time
import signal
import uvicorn
import multiprocessing
import torch

# /mnt/atin/huyquang/container-main/stitch-storage


class ResDataSide(BaseModel):
    imagePath1: str  # save image path
    imagePath2: str  # save image path
    status: int  # status stitching


class ResDataTop(BaseModel):
    imagePath1: str  # save image path
    imagePath2: str  # save image path
    status: int  # status stitching
    

class ResDataSplit(BaseModel):
    status: int  # status stitching


app = FastAPI(title="STITCHING API", description="Stitching image")

api_controller = APIController()
api_controller.setup_controller()


@app.post('/api/stitch/side', description="STIT", response_model=ResDataSide, tags=["API"])
async def api_stit_side(data: APIData):
    t = time.time()
    res = await api_controller.stitching_side(data)
    print("Total time stitching side: ", time.time() - t)
    t1 = time.time()
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    print(time.time() - t1, "time clear memory")
    return res


@app.post('/api/stitch/top', description="STIT", response_model=ResDataTop, tags=["API"])
async def api_stit_top(data: APIData):
    t = time.time()
    res = await api_controller.stitching_top(data)
    print("Total time stitching top: ", time.time() - t)
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    return res


@app.post('/api/stitch/side/video', description="STIT", response_model=ResDataSide, tags=["API"])
async def api_stit_side_video(data: APIVideoData):
    t = time.time()
    res = await api_controller.stitching_side_with_video(data)
    print("Total time stitching side with video: ", time.time() - t)
    return res


@app.post('/api/split/container/side', description="SPLIT CONTAINER", response_model=ResDataSplit, tags=["API"])
async def api_split_container(data: SplitData):
    res = await api_controller.image_split_container_side(data)
    return res


@app.post('/api/split/container/top', description="SPLIT CONTAINER", response_model=ResDataSplit, tags=["API"])
async def api_split_container(data: SplitData):
    res = await api_controller.image_split_container_top(data)
    return res


# --hidden-import=main_fast_api
if __name__ == '__main__':
    multiprocessing.freeze_support()
    # print(f'{ROOT}\\{PATH}')
    # Nếu muốn sử dụng worker cần đổi: app -> "name:app" ,name là tên file main cần chạy, hoặc đường dẫn đến main
    print(f"NUMBER WORKERS: {WORKERS}")
    uvicorn.run("main:app", host=HOST, port=PORT,
                reload=False)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
