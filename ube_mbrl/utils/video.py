"""
Copyright (c) 2023 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
"""
Small modification from https://github.com/denisyarats/pytorch_sac/blob/master/video.py
Copyright (c) 2019 Denis Yarats
cf. thirdparty_licenses.md file in the root directory of this source tree.
"""

import os

import imageio
from pyvirtualdisplay import Display


display = Display(visible=0, size=(512, 512))
display.start()


def close_virtual_display():
    display.stop()


class VideoRecorder(object):
    def __init__(self, save_dir, height=256, width=256, camera_id=0, fps=30):
        self.save_dir = save_dir
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode="rgb_array",
                height=self.height,
                width=self.width,
                # camera_id=self.camera_id,
            )
            self.frames.append(frame)

    def record_default(self, env):
        if self.enabled:
            frame = env.render(
                mode="rgb_array",
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps, macro_block_size=None)
