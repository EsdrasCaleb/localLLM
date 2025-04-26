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

class Tracks_getTrack_3_1_Test {

    @InjectMocks
    private Tracks tracks;

    @Mock
    private ArrayList<String> mockTracks;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        tracks.setTrack(new String[] { "Track1", "Track2", "Track3" });
    }

    @Test
    void testGetTrackValidIndex() {
        int index = 1;
        when(mockTracks.get(index)).thenReturn("Track2");
        when(mockTracks.size()).thenReturn(3);
        String result = tracks.getTrack(index);
        assertEquals("Track2", result);
        verify(mockTracks).get(index);
    }

    @Test
    void testGetTrackInvalidIndex() {
        int index = 5;
        when(mockTracks.size()).thenReturn(3);
        String result = tracks.getTrack(index);
        assertNull(result);
        verify(mockTracks, never()).get(index);
    }

    @Test
    void testGetTrackEmptyList() {
        int index = 0;
        when(mockTracks.size()).thenReturn(0);
        String result = tracks.getTrack(index);
        assertNull(result);
        verify(mockTracks, never()).get(index);
    }
}
