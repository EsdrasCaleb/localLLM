// Tracks_toString_4_1_Test.java
package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Tracks_toString_4_1_Test {

    @Mock
    private Tracks tracks;

    @BeforeEach
    public void setup() {
        when(tracks.getTrack()).thenReturn(new String[0]);
    }

    @Test
    public void testToString_EmptyTracks() {
        String result = tracks.toString();
        assertEquals("Tracks is null or size 0", result);
    }

    @Test
    public void testToString_SingleTrack() {
        when(tracks.getTrack()).thenReturn(new String[] { "Track1" });
        String result = tracks.toString();
        assertEquals("Track - Track1", result);
    }

    @Test
    public void testToString_MultipleTracks() {
        when(tracks.getTrack()).thenReturn(new String[] { "Track1", "Track2", "Track3" });
        String result = tracks.toString();
        assertEquals("# of Tracks = 3\nTrack - Track1\nTrack - Track2\nTrack - Track3", result);
    }

    @Test
    public void testToString_NullTracks() {
        when(tracks.getTrack()).thenReturn(null);
        String result = tracks.toString();
        assertEquals("Tracks is null or size 0", result);
    }
}
