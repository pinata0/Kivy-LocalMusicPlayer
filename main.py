from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.lang.builder import Builder
from kivy.clock import Clock
from ffpyplayer.player import MediaPlayer
from mutagen.mp3 import MP3
from mutagen.id3 import ID3
from PIL import Image as PILImage, ImageFilter, ImageEnhance
from kivy.core.window import Window
from kivy.uix.label import Label
import os

from algorithm import algorithm

Builder.load_file("chocopopplayer.kv")
Window.set_icon("assets/Icons/favicon.ico")
Window.size = (600, 600)

def resize_to_square(image_path, size=(700, 700)):
    """Resize an image to a square of the specified size."""
    try:
        with PILImage.open(image_path) as img:
            img = img.resize(size, PILImage.Resampling.LANCZOS)
            return img
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")
        return None

class ImageButton(ButtonBehavior, Image):
    """Custom button with image support."""
    pass

class ChocopopPlayer(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.player = None
        self.is_playing = False
        self.current_time = 0
        self.total_duration = 1
        self.is_dragging = False
        self.is_showLyrics = False
        self.songs_folder = "data/songs"
        self.covers_folder = "data/covers"
        self.lyrics_folder = "data/lyrics"
        self.default_cover = "default_cover.png"
        self.default_resized_cover = "default_cover_resized.png"
        self.blurred_image_path = "temp/temp_blurred_background.jpg"

        # 기본 커버 이미지를 정사각형으로 리사이즈
        resize_to_square(self.default_cover, self.default_resized_cover)

        # 초기 커버와 배경 설정
        self.update_image_sources(self.default_resized_cover)

        # 데이터 폴더 준비
        self.refresh_song_list()

        # 30FPS로 진행 업데이트
        Clock.schedule_interval(self.update_progress, 1 / 30)

    def refresh_song_list(self):
        """Load songs and update song list."""

        # 곡 파일 이름만 추출
        self.song_list = algorithm(self.songs_folder, self.lyrics_folder)
        self.update_text_layout()

    def load_song(self, file_name, should_play = False):
        """
        Load the selected song and update the cover image and background.
        """
        print('Loading!!!!')
        # 기존 곡 재생 종료
        if self.player:
            self.player.close_player()
            self.is_playing = False
            self.ids.play_button.source = "assets/Icons/play-white.png"

        # 초기화: 진행 바, 동그라미, 텍스트
        self.current_time = 0
        self.ids.progress_bar.value = 0
        self.ids.current_time.text = "0:00"
        self.update_circle_position()

        # 새로운 곡 로드
        file_path = os.path.join(self.songs_folder, file_name)
        self.ids.track_label.text = os.path.basename(file_path)
        self.player = MediaPlayer(file_path, ff_opts={'paused': True})

        # 곡 길이 계산
        self.total_duration = self.get_total_duration(file_path)
        self.ids.progress_bar.max = self.total_duration
        self.ids.total_time.text = self.format_time(self.total_duration)

        # 커버 이미지와 배경 화면 업데이트
        cover_file = os.path.splitext(file_name)[0] + ".webp"
        cover_path = os.path.join(self.covers_folder, cover_file)

        if os.path.exists(cover_path):
            self.update_image_sources(cover_path)
        else:
            self.update_image_sources(self.default_resized_cover)
        self.select_song(file_name)

        # 곡 상태 복원
        if should_play:
            print("DEBUG: Resuming playback for the next song.")
            self.toggle_play()  # 재생 상태로 전환


    def select_song(self, selected_song):
        """Update the song selection and change text color."""
        # Reset all songs to default color
        for item in self.ids.text_layout.data:
            item['color'] = (1, 1, 1, 1)  # 기본 흰색

        # Change the selected song's color to yellow
        for item in self.ids.text_layout.data:
            if item['text'] == selected_song:
                item['color'] = (1, 1, 0, 1)  # 노란색

        # Refresh the RecycleView
        self.ids.text_layout.refresh_from_data()
        print(f"Selected song: {selected_song}")


    def get_total_duration(self, file_path):
        """Get the total duration of the MP3 file using mutagen."""
        try:
            audio = MP3(file_path)
            return audio.info.length
        except Exception as e:
            print(f"Error getting duration: {e}")
            return 1

    def get_total_duration(self, file_path):
        """Get the total duration of the MP3 file using mutagen."""
        try:
            audio = MP3(file_path)
            return audio.info.length
        except Exception as e:
            print(f"Error getting duration: {e}")
            return 1

    def toggle_play(self):
        """Toggle play/pause for the song."""
        if self.player:
            if self.is_playing:
                # 일시 정지
                self.player.toggle_pause()
                self.is_playing = False
                self.ids.play_button.source = "assets/Icons/play-white.png"
            else:
                # 곡 재생 시작
                if self.current_time == 0:
                    self.player.seek(0, relative=False)  # 곡의 시작 위치로 이동
                else:
                    self.player.seek(self.current_time, relative=False)  # 드래그된 위치에서 시작

                self.player.toggle_pause()  # 재생 시작
                self.is_playing = True
                self.ids.play_button.source = "assets/Icons/stop-white.png"

    def update_progress(self, dt):
        """Update the progress bar and related elements."""
        if not self.is_dragging and self.is_playing and self.player:
            # 현재 재생 시간 업데이트
            self.current_time = self.player.get_pts()  # 현재 재생 위치 가져오기
            self.ids.progress_bar.value = self.current_time

            # 현재 시간 텍스트 업데이트
            self.ids.current_time.text = self.format_time(self.current_time)

            # 동그라미 위치 업데이트
            self.update_circle_position()

            # 곡이 끝났을 경우 처리
            if round(self.current_time) >= self.total_duration-1:
                # self.is_playing = False
                self.ids.progress_bar.value = self.total_duration  # 진행 바 끝으로 설정
                self.ids.current_time.text = self.format_time(self.total_duration)  # 마지막 시간 표시
                self.ids.play_button.source = "assets/Icons/play-white.png"
                self.next_song()

    def update_circle_position(self):
        """Update the position of the progress circle."""
        progress_bar = self.ids.progress_bar
        circle = self.ids.progress_circle
        progress_ratio = self.current_time / self.total_duration if self.total_duration > 0 else 0
        circle_x = progress_bar.x + (progress_ratio * progress_bar.width) - circle.width / 2
        circle_y = progress_bar.center_y - circle.height / 2
        circle.pos = (circle_x, circle_y)

    def on_touch_down(self, touch):
        """Handle touch down events for dragging."""
        circle = self.ids.progress_circle
        progress_bar = self.ids.progress_bar
        if circle.collide_point(*touch.pos) or progress_bar.collide_point(*touch.pos):
            self.is_dragging = True
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        """Handle touch move events for dragging."""
        if self.is_dragging:
            progress_bar = self.ids.progress_bar
            bar_min_x = progress_bar.x
            bar_max_x = progress_bar.right

            # 터치 위치를 진행 바의 값으로 변환
            clamped_x = max(bar_min_x, min(touch.x, bar_max_x))
            new_value = (clamped_x - bar_min_x) / progress_bar.width * progress_bar.max

            # 진행 바 값 및 곡 재생 위치 업데이트
            progress_bar.value = new_value
            self.current_time = new_value
            self.update_circle_position()

            # 현재 시간 텍스트 업데이트
            self.ids.current_time.text = self.format_time(self.current_time)

            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        """Handle touch up events for dragging."""
        if self.is_dragging:
            self.is_dragging = False
            if self.player:
                self.player.seek(self.current_time, relative=False)  # 드래그된 위치로 이동
            return True
        return super().on_touch_up(touch)

    def format_time(self, seconds):
        """Format seconds into MM:SS format."""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}:{seconds:02}"
    
    def find_current_song_index(self):
        """Find the index of the current song in the text layout."""
        current_song = self.ids.track_label.text.strip()

        for i, item in enumerate(self.ids.text_layout.data):
            if item["text"] == current_song:
                print(f"Found song at index: {i}")
                return i
        print("Current song not found in list.")
        return -1

    def previous_song(self):
        """Play the previous song."""
        if self.is_showLyrics:
            self.on_cover_image_click()

        current_index = self.find_current_song_index()
        if current_index > 0:  # 이전 곡이 있는 경우
            previous_song = self.ids.text_layout.data[current_index - 1]["text"]
            was_playing = self.is_playing  # 현재 재생 상태 저장
            self.load_song(previous_song, should_play=was_playing)
            self.select_song(previous_song)  # 선택 상태 업데이트
            self.song_finished = False  # 플래그 초기화
        else:
            next_song = self.ids.text_layout.data[-1]["text"]
            was_playing = self.is_playing  # 현재 재생 상태 저장
            self.load_song(next_song, should_play=was_playing)
            self.select_song(next_song)  # 선택 상태 업데이트
            self.song_finished = False  # 플래그 초기화

    def next_song(self):
        """Play the next song."""
        if self.is_showLyrics:
            self.on_cover_image_click()

        current_index = self.find_current_song_index()
        if current_index < len(self.ids.text_layout.data) - 1:  # 다음 곡이 있는 경우
            next_song = self.ids.text_layout.data[current_index + 1]["text"]
            was_playing = self.is_playing  # 현재 재생 상태 저장
            self.load_song(next_song, should_play=was_playing)
            self.select_song(next_song)  # 선택 상태 업데이트
            self.song_finished = False  # 플래그 초기화
        else:
            next_song = self.ids.text_layout.data[0]["text"]
            was_playing = self.is_playing  # 현재 재생 상태 저장
            self.load_song(next_song, should_play=was_playing)
            self.select_song(next_song)  # 선택 상태 업데이트
            self.song_finished = False  # 플래그 초기화
    
    def apply_blur_to_background(self, image_path):
        """
        Apply a blur effect to the given image, make it darker, save it as a file, and update the background.
        """
        os.makedirs("temp", exist_ok=True)  # Ensure the temp directory exists

        try:
            with PILImage.open(image_path) as img:
                img = img.resize((1920, 1080), PILImage.Resampling.LANCZOS)  # Resize to fit screen
                blurred_img = img.filter(ImageFilter.GaussianBlur(20))  # Apply Gaussian blur

                # Make the image darker
                enhancer = ImageEnhance.Brightness(blurred_img)
                darkened_img = enhancer.enhance(0.5)  # Adjust brightness (0.5 for 50% brightness)

                # Save darkened and blurred image to file
                darkened_img.save(self.blurred_image_path)

            # Update the background image
            self.ids.background_image.source = self.blurred_image_path
            self.ids.background_image.reload()  # Force Kivy to reload the image
            # print(f"Background successfully updated to: {self.blurred_image_path}")


        except Exception as e:
            print(f"Error applying blur to background: {e}")

    def update_image_sources(self, image_path):
        """
        Update both the cover image and background image.
        """
        self.ids.cover_image.source = image_path
        self.apply_blur_to_background(image_path)

    def on_cover_image_click(self):
        """Toggle between song list and lyrics view."""
        self.is_showLyrics = not self.is_showLyrics
        self.update_text_layout()

    def update_text_layout(self):
        """Update the text_layout with songs or lyrics."""
        if self.is_showLyrics:
            # 현재 곡의 가사를 표시
            current_song = self.ids.track_label.text.strip()  # 곡 이름에서 불필요한 공백 제거
            lyrics_path = os.path.join(self.lyrics_folder, f"{current_song[:-4]}.txt")
            if os.path.exists(lyrics_path):
                with open(lyrics_path, "r", encoding="utf-8") as file:
                    lyrics = file.read().splitlines()
                self.ids.text_layout.data = [{"text": line, "color": (1, 1, 1, 1)} for line in lyrics]  # 흰색
            else:
                self.ids.text_layout.data = [{"text": "Lyrics not available.", "color": (1, 1, 1, 1)}]  # 흰색
        else:
            # 곡 목록을 표시
            self.ids.text_layout.data = [{"text": song, "color": (1, 1, 1, 1)} for song in self.song_list]
            # 곡 목록에서 선택된 항목은 노란색으로 표시
            selected_song = self.ids.track_label.text.strip()
            for item in self.ids.text_layout.data:
                if item["text"] == selected_song:
                    item["color"] = (1, 1, 0, 1)  # 노란색

class ChocopopApp(App):
    def build(self):
        return ChocopopPlayer()
    
class SelectableLabel(Label):
    """Custom selectable label for song list."""
    def on_touch_down(self, touch):
        app = App.get_running_app()
        # 가사 화면인지 확인
        if app.root.is_showLyrics:
            print("In lyrics view, song selection is disabled.")
            return False  # 이벤트 무시
        if self.collide_point(*touch.pos):
            app.root.load_song(self.text)  # 곡 선택 및 로드
            return True
        return super().on_touch_down(touch)

if __name__ == "__main__":
    ChocopopApp().run()
