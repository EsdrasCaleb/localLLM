package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Tracks_getTrack_3_0_Test {

    @Test
    public void testGetTrack() {
        Tracks tracks = new Tracks();
        String[] testTracks = { "track1", "track2", "track3" };
        tracks.setTrack(testTracks);
        Assertions.assertEquals("track1", tracks.getTrack(0));
        Assertions.assertEquals("track2", tracks.getTrack(1));
        Assertions.assertEquals("track3", tracks.getTrack(2));
    }
}
