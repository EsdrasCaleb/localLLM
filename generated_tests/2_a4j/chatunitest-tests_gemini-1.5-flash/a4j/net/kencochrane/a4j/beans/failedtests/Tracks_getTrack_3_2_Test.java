package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Tracks_getTrack_3_2_Test {

    private Tracks tracks;

    @BeforeEach
    void setUp() {
        tracks = new Tracks();
    }

    @Test
    void testGetTrackWithinBounds() {
        String[] trackList = { "Track 1", "Track 2", "Track 3" };
        tracks.setTrack(trackList);
        assertEquals("Track 2", tracks.getTrack(1));
    }

    @Test
    void testGetTrackOutOfBounds() {
        String[] trackList = { "Track 1", "Track 2", "Track 3" };
        tracks.setTrack(trackList);
        assertNull(tracks.getTrack(3));
    }

    @Test
    void testGetTrackEmptyList() {
        assertNull(tracks.getTrack(0));
    }

    @Test
    void testGetTrackZeroIndex() {
        String[] trackList = { "Track 1", "Track 2", "Track 3" };
        tracks.setTrack(trackList);
        assertEquals("Track 1", tracks.getTrack(0));
    }

    @Test
    void testGetTrackNegativeIndex() {
        String[] trackList = { "Track 1", "Track 2", "Track 3" };
        tracks.setTrack(trackList);
        assertNull(tracks.getTrack(-1));
    }

    @Test
    void testNullTrackList() {
        try {
            Field tracksField = Tracks.class.getDeclaredField("tracks");
            tracksField.setAccessible(true);
            tracksField.set(tracks, null);
            assertNull(tracks.getTrack(0));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Exception during reflection: " + e.getMessage());
        }
    }
}
