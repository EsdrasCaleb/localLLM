// Test class
package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class Tracks_getTrack_3_0_Test {

    @ExtendWith(MockitoExtension.class)
    public class TestClass {

        @Mock
        private Tracks mockTracks;

        @Test
        void testGetTrack() {
            // Arrange
            ArrayList<String> tracks = new ArrayList<>();
            tracks.add("track1");
            tracks.add("track2");
            // <Buggy Line>: non-static method getTrack(int) cannot be referenced from a static context
            String result = mockTracks.getTrack(1);
            // Assert
            assertEquals("track2", result);
        }
    }
}
