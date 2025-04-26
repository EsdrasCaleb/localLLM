package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Tracks_toString_4_2_Test {

    @Test
    public void testToString() {
        Tracks tracks = new Tracks();
        String expected = "Tracks is null or size 0";
        assertEquals(expected, tracks.toString());
        tracks.setTrack(new String[] { "Track 1", "Track 2", "Track 3" });
        expected = "# of Tracks = 3\nTrack - Track 1\nTrack - Track 2\nTrack - Track 3";
        assertEquals(expected, tracks.toString());
        tracks.setTrack(new String[] {});
        expected = "# of Tracks = 0\n";
        assertEquals(expected, tracks.toString());
    }
}
