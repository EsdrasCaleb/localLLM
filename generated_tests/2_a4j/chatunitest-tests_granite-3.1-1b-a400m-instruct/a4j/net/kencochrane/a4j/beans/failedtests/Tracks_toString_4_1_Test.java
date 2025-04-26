package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Tracks_toString_4_1_Test {

    @Test
    void testToString() {
        Tracks tracks = new Tracks();
        tracks.setTrack(new String[] { "Track 1", "Track 2", "Track 3" });
        String expected = "# of Tracks = 3\nTrack - Track 1\nTrack - Track 2\nTrack - Track 3\n";
        String actual = tracks.toString();
        assertEquals(expected, actual);
    }

    @Test
    void testToStringWithNullTracks() {
        Tracks tracks = new Tracks();
        tracks.setTrack(null);
        String expected = "Tracks is null or size 0\n";
        String actual = tracks.toString();
        assertEquals(expected, actual);
    }

    @Test
    void testToStringWithEmptyTracks() {
        Tracks tracks = new Tracks();
        tracks.setTrack(new String[0]);
        String expected = "# of Tracks = 0\n";
        String actual = tracks.toString();
        assertEquals(expected, actual);
    }
}
