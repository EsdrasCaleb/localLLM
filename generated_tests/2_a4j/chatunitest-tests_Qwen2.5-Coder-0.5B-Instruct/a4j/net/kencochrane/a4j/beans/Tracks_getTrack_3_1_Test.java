package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Tracks_getTrack_3_1_Test {

    @Test
    public void testGetTrack() {
        // Arrange
        Tracks tracks = mock(Tracks.class);
        ArrayList<String> tracksList = new ArrayList<>();
        tracksList.add("Track 1");
        tracksList.add("Track 2");
        // Act
        tracks.setTrack(tracksList.toArray(new String[tracksList.size()]));
        // Assert
        assertEquals("Track 2", tracks.getTrack(1));
    }
}
