package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Tracks_toString_4_0_Test {

    @Test
    void testToString() {
        Tracks tracks = new Tracks();
        tracks.setTrack(new String[] { "Track 1", "Track 2" });
        String expected = "# of Tracks = 2\nTrack - Track 1\nTrack - Track 2";
        String actual = tracks.toString();
        assertEquals(expected, actual);
    }
}
