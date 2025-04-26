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

class Tracks_toString_4_0_Test {

    @InjectMocks
    private Tracks tracks;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testToStringWithTracks() {
        ArrayList<String> mockTracks = new ArrayList<>(Arrays.asList("Track1", "Track2", "Track3"));
        tracks.setTrack(mockTracks.toArray(new String[0]));
        String expected = "# of Tracks = 3\n" + "Track - Track1\n" + "Track - Track2\n" + "Track - Track3\n";
        assertEquals(expected, tracks.toString());
    }

    @Test
    void testToStringWithEmptyTracks() {
        ArrayList<String> mockTracks = new ArrayList<>();
        tracks.setTrack(mockTracks.toArray(new String[0]));
        String expected = "Tracks is null or size 0\n";
        assertEquals(expected, tracks.toString());
    }
}
