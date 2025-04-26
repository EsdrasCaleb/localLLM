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
    void testGetTrackNegativeIndex() {
        String[] trackList = { "Track 1", "Track 2", "Track 3" };
        tracks.setTrack(trackList);
        assertNull(tracks.getTrack(-1));
    }
}
