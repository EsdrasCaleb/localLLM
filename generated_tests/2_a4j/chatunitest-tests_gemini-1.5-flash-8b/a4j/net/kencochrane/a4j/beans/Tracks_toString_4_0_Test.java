package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Tracks_toString_4_0_Test {

    private Tracks tracks;

    @BeforeEach
    void setUp() {
        tracks = new Tracks();
    }

    @Test
    void testToStringEmptyTracks() {
        String expected = "Tracks is null or size 0\n";
        String actual = tracks.toString();
        assertEquals(expected, actual);
    }

    @Test
    void testToStringNonEmptyTracks() {
        String[] tracksArray = { "Track 1", "Track 2", "Track 3" };
        tracks.setTrack(tracksArray);
        String expected = "# of Tracks = 3\n" + "Track - Track 1\n" + "Track - Track 2\n" + "Track - Track 3\n";
        String actual = tracks.toString();
        assertEquals(expected, actual);
    }

    @Test
    void testToStringWithOneTrack() {
        String[] tracksArray = { "Track 1" };
        tracks.setTrack(tracksArray);
        String expected = "# of Tracks = 1\n" + "Track - Track 1\n";
        String actual = tracks.toString();
        assertEquals(expected, actual);
    }
}
