package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Tracks_getTrack_3_0_Test {

    @Test
    void testGetTrack() {
        Tracks tracks = new Tracks();
        tracks.setTrack(new String[] { "track1", "track2" });
        int index = 1;
        String expectedTrack = "track2";
        String actualTrack = tracks.getTrack(index);
        assert actualTrack.equals(expectedTrack);
    }
}
