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

public class Tracks_toString_4_0_Test {

    @Test
    public void testToString() {
        // Arrange
        ArrayList<String> tracks = new ArrayList<>(Arrays.asList("Track 1", "Track 2", "Track 3"));
        Tracks trackObject = new Tracks();
        trackObject.setTrack(tracks.toArray(new String[0]));
        // Act
        String result = trackObject.toString();
        // Assert
        assertEquals("3 tracks: Track 1\nTrack 2\nTrack 3", result);
    }
}
