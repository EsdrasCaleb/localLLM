package net.kencochrane.a4j.beans;

import java.util.ArrayList;
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
    public void setUp() {
        tracks = new Tracks();
        tracks.setTrack(new String[] { "Track1", "Track2", "Track3" });
    }

    @Test
    public void testGetTrack() {
        assertEquals("Track1", tracks.getTrack(0));
        assertEquals("Track2", tracks.getTrack(1));
        assertEquals("Track3", tracks.getTrack(2));
    }
}
