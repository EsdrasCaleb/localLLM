// Test method
package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class Tracks_toString_4_0_Test {

    Tracks tracks;

    @BeforeEach
    public void setUp() {
        tracks = new Tracks();
    }

    @Test
    public void testToString() {
        // Arrange
        tracks.setTrack(new String[] { "Track 1", "Track 2", "Track 3" });
        // Act
        String result = tracks.toString();
        // Assert
        assertEquals("Tracks is null or size 0\nTrack - Track 1\nTrack - Track 2\nTrack - Track 3", result);
    }
}
