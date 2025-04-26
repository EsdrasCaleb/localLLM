package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class Tracks_getTrack_3_0_Test {

    @Test
    public void testGetTrack() {
        Tracks tracks = new Tracks();
        tracks.setTrack(new String[] { "Track1", "Track2", "Track3" });
        assertEquals("Track1", tracks.getTrack(0));
        assertEquals("Track2", tracks.getTrack(1));
        assertEquals("Track3", tracks.getTrack(2));
        assertNull(tracks.getTrack(-1));
        assertNull(tracks.getTrack(3));
    }
}
