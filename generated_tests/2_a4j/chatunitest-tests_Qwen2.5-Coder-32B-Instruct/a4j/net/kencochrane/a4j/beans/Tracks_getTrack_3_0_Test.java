package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Tracks_getTrack_3_0_Test {

    @InjectMocks
    private Tracks tracks;

    @Mock
    private ArrayList<String> mockTracks;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        Field tracksField = Tracks.class.getDeclaredField("tracks");
        tracksField.setAccessible(true);
        tracksField.set(tracks, mockTracks);
    }

    @Test
    public void testGetTrack_IndexLessThanZero() {
        when(mockTracks.size()).thenReturn(3);
        String result = tracks.getTrack(-1);
        assertNull(result);
        verify(mockTracks, never()).get(anyInt());
    }
}
